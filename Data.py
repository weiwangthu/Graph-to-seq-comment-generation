import torch
import random
import copy
import numpy as np
from graph_loader import *
from util.nlp_utils import split_chinese_sentence, remove_stopwords
from util.dict_utils import cosine_sim
from util.utils import bow

PAD = 0
BOS = 1
EOS = 2
UNK = 3
MASK = 4
TITLE = 5
BUFSIZE = 4096000

MAX_ARTICLE_LENGTH = 600
MAX_TITLE_LENGTH = 30
MAX_COMMENT_LENGTH = 50
# MAX_COMMENT_NUM = 5


class Vocab:
    def __init__(self, vocab_file, vocab_size=50000):
        self._word2id = {'[PADDING]': 0, '[START]': 1, '[END]': 2, '[OOV]': 3, '[MASK]': 4, '_TITLE_': 5}
        self._id2word = ['[PADDING]', '[START]', '[END]', '[OOV]', '[MASK]', '_TITLE_']
        self._wordcount = {'[PADDING]': 1, '[START]': 1, '[END]': 1, '[OOV]': 1, '[MASK]': 1, '_TITLE_': 1}
        self.load_vocab(vocab_file, vocab_size)
        self.voc_size = len(self._word2id)
        self.UNK_token = 3
        self.PAD_token = 0
        self.stop_words = {word.strip() for word in open('stop_words.txt').readlines()}

    def load_vocab(self, vocab_file, vocab_size):
        for line in open(vocab_file):
            term_ = line.strip().split('\t')
            if len(term_) != 2:
                continue
            word, count = term_
            assert word not in self._word2id
            self._word2id[word] = len(self._word2id)
            self._id2word.append(word)
            self._wordcount[word] = int(count)
            if len(self._word2id) >= vocab_size:
                break
        assert len(self._word2id) == len(self._id2word)

    def word2id(self, word):
        if word in self._word2id:
            return self._word2id[word]
        return self._word2id['[OOV]']

    def sent2id(self, sent, add_start=False, add_end=False, remove_stop=False):
        if not remove_stop:
            result = [self.word2id(word) for word in sent]
        else:
            result = [self.word2id(word) for word in sent if word not in self.stop_words]
        if add_start:
            result = [self._word2id['[START]']] + result

        if add_end:
            result = result + [self._word2id['[END]']]
        return result

    def id2word(self, word_id):
        return self._id2word[word_id]

    def id2sent(self, sent_id):
        result = []
        for id in sent_id:
            if id == self._word2id['[END]']:
                break
            elif id == self._word2id['[PADDING]']:
                continue
            result.append(self._id2word[id])
        return result

class Example:
    """
    Each example is one data pair
        src: title (has oov)
        tgt: comment (oov has extend ids if use_oov else has oov)
        memory: tag (oov has extend ids)
    """

    def __init__(self, original_content, title, target, vocab, is_train, news_id, model, content_label=None):
        self.ori_title = title[:MAX_TITLE_LENGTH]
        self.ori_original_content = original_content[:MAX_ARTICLE_LENGTH]
        self.ori_news_id = news_id
        self.content_label = content_label
        if content_label is not None:
            min_title = min(len(title), MAX_TITLE_LENGTH)
            temp = [l for l in content_label if l < min_title]
            min_body = min(len(original_content), MAX_ARTICLE_LENGTH)
            temp2 = [l for l in content_label if 0 <= l - len(title) < min_body]
            if MAX_TITLE_LENGTH < len(title):
                temp2 = [l - (len(title) - MAX_TITLE_LENGTH) for l in temp2]
            temp = temp + temp2
            if len(temp) == 0:
                temp = [-1]
            self.content_label = temp

        if is_train:
            self.ori_target = target[:MAX_COMMENT_LENGTH]
        else:
            self.ori_targets = [tar[:MAX_COMMENT_LENGTH] for tar in target]

        self.title = vocab.sent2id(self.ori_title)
        self.original_content = vocab.sent2id(self.ori_original_content)
        if is_train:
            self.target = vocab.sent2id(self.ori_target, add_start=True, add_end=True)

        if model == 'h_attention':
            self.sentence_content = split_chinese_sentence(self.ori_original_content)
            self.sentence_content = [vocab.sent2id(sen[:MAX_ARTICLE_LENGTH]) for sen in self.sentence_content]
            self.sentence_content_max_len = Batch.get_length(self.sentence_content, MAX_ARTICLE_LENGTH)
            self.sentence_content, self.sentence_content_mask = Batch.padding_list_to_tensor(self.sentence_content, self.sentence_content_max_len.max().item())

        if model == 'bow2seq':
            self.bow = self.bow_vec(self.original_content, MAX_ARTICLE_LENGTH)

        if model == 'autoenc_vae_bow' or model == 'user_autoenc_vae_bow':
            if is_train:
                content_words = self.target
                self.tgt_bow = np.bincount(content_words, minlength=vocab.voc_size)
        if model == 'autoenc_vae_bow_norm' or model == 'user_autoenc_vae_bow_norm' \
                or model == 'user2seq_test_new' or model == 'var_select_user2seq_new' \
                or model == 'var_select2seq_test_new' or model == 'var_select2seq_test_span' \
                or model == 'var_select2seq_test_span2' or model == 'var_select_user2seq_new2' \
                or model == 'var_select2seq_test_span3' or model == 'var_select_user2seq_label' \
                or model == 'user_autoenc_vae_bow2' or model == 'var_select_user2seq_new3' \
                or model == 'user_autoenc_vae_bow3' or model == 'user2seq_test_new2' \
                or model == 'user2seq_test_new22' or model == 'var_select_user2seq_new22':
            if is_train:
                content_words = vocab.sent2id(self.ori_target, add_start=True, add_end=True, remove_stop=True)
                self.tgt_bow = np.bincount(content_words, minlength=vocab.voc_size)
                # normal
                self.tgt_bow[vocab.UNK_token] = 0
                self.tgt_bow = self.tgt_bow / np.sum(self.tgt_bow)

    def bow_vec(self, content, max_len):
        bow = {}
        for word_id in content:
            if word_id not in bow:
                bow[word_id] = 0
            bow[word_id] += 1
        bow = list(bow.items())
        bow.sort(key=lambda k: k[1], reverse=True)
        bow.insert(0, (UNK, 1))
        return [word_id[0] for word_id in bow[:max_len]]

    @staticmethod
    def bag_of_word(words, length):
        vec = [0 for _ in range(length)]
        for w in words:
            vec[w] += 1
        return vec


class Batch:
    """
    Each batch is a mini-batch of data

    """

    def __init__(self, example_list, is_train, model):
        self.model = model
        self.is_train = is_train
        self.examples = example_list
        if model == 'h_attention':
            self.sentence_content = [e.sentence_content for e in example_list]
            self.sentence_content_mask = [e.sentence_content_mask for e in example_list]

            self.sentence_content_len = self.get_length(self.sentence_content)
            self.sentence_mask, _ = self.padding_list_to_tensor([[1 for _ in range(d)] for d in self.sentence_content_len], self.sentence_content_len.max().item())
            self.sentence_mask = self.sentence_mask.to(torch.uint8)
        elif model == 'bow2seq':
            bow_list = [e.bow for e in example_list]
            self.bow_len = self.get_length(bow_list, MAX_ARTICLE_LENGTH)
            self.bow, self.bow_mask = self.padding_list_to_tensor(bow_list, self.bow_len.max().item())
        elif model == 'autoenc_vae_bow' or model == 'user_autoenc_vae_bow'\
                or model == 'autoenc_vae_bow_norm' or model == 'user_autoenc_vae_bow_norm':
            if is_train:
                self.tgt_bow = torch.FloatTensor([e.tgt_bow for e in example_list])
            else:
                title_list = [e.title for e in example_list]
                self.title_len = self.get_length(title_list, MAX_TITLE_LENGTH)
                self.title, self.title_mask = self.padding_list_to_tensor(title_list, self.title_len.max().item())

        # seq2seq, select_diverse2seq, select2seq and so on.
        else:
            content_list = [e.original_content for e in example_list]
            self.content_len = self.get_length(content_list, MAX_ARTICLE_LENGTH)
            self.content, self.content_mask = self.padding_list_to_tensor(content_list, self.content_len.max().item())

            title_list = [e.title for e in example_list]
            self.title_len = self.get_length(title_list, MAX_TITLE_LENGTH)
            self.title, self.title_mask = self.padding_list_to_tensor(title_list, self.title_len.max().item())

            title_content_list = [e.title + e.original_content for e in example_list]
            self.title_content_len = self.get_length(title_content_list, MAX_TITLE_LENGTH + MAX_ARTICLE_LENGTH)
            self.title_content, self.title_content_mask = self.padding_list_to_tensor(title_content_list, self.title_content_len.max().item())

            if model == 'user2seq_test_new' or model == 'var_select_user2seq_new' \
                    or model == 'var_select2seq_test_new' or model == 'var_select2seq_test_span' \
                    or model == 'var_select2seq_test_span2' or model == 'var_select_user2seq_new2'\
                    or model == 'var_select2seq_test_span3' or model == 'var_select_user2seq_label'\
                    or model == 'user_autoenc_vae_bow2' or model == 'var_select_user2seq_new3' \
                    or model == 'user_autoenc_vae_bow3' or model == 'user2seq_test_new2' \
                    or model == 'user2seq_test_new22' or model == 'var_select_user2seq_new22':
                if is_train:
                    self.tgt_bow = torch.FloatTensor([e.tgt_bow for e in example_list])

        if is_train:
            self.tgt_len = self.get_length([e.target for e in example_list])
            self.tgt, self.tgt_mask = self.padding_list_to_tensor([e.target for e in example_list], self.tgt_len.max().item())

            if example_list[0].content_label is not None:
                self.content_label = [e.content_label for e in example_list]

    @staticmethod
    def get_length(examples, max_len=1000):
        length = []
        for e in examples:
            if len(e) > max_len:
                length.append(max_len)
            else:
                length.append(len(e))
        assert len(length) == len(examples)
        length = torch.LongTensor(length)
        return length

    @staticmethod
    def padding_list_to_tensor(batch, max_len):
        padded_batch = []
        mask_batch = []
        for x in batch:
            y = x + [PAD] * (max_len - len(x))
            m = [1] * len(x) + [0] * (max_len - len(x))
            padded_batch.append(y)
            mask_batch.append(m)
        padded_batch = torch.LongTensor(padded_batch)
        mask_batch = torch.LongTensor(mask_batch).to(torch.uint8)
        return padded_batch, mask_batch

    @staticmethod
    def padding_2d_list_to_tensor(batch, max_len):
        padded_batch = []
        mask_batch = []
        for x in batch:
            y = x + [PAD] * (max_len - len(x))
            m = [1] * len(x) + [0] * (max_len - len(x))
            padded_batch.append(y)
            mask_batch.append(m)
        padded_batch = torch.LongTensor(padded_batch)
        mask_batch = torch.LongTensor(mask_batch).to(torch.uint8)
        return padded_batch, mask_batch


class DataLoader:
    def __init__(self, filename, config, vocab, adj_type, use_gnn, model, is_train=True, debug=False, train_num=0, extra_file=None):
        self.batch_size = config.batch_size if is_train else config.max_generator_batches
        self.vocab = vocab
        self.config = config
        # self.max_len = MAX_LENGTH
        self.filename = filename
        self.stream = open(self.filename, encoding='utf8')
        self.epoch_id = 0

        self.is_train = is_train
        self.debug = debug
        self.train_num = train_num

        self.adj_type = adj_type
        self.use_gnn = use_gnn
        self.model = model

        if self.config.dataset_name == 'yahoo':
            self.create_comments_from_article = lambda x,y: self.create_comments_from_yahoo_article(x, y)
        elif self.config.dataset_name == '163':
            self.create_comments_from_article = lambda x, y: self.create_comments_from_163_article(x, y)
        else:
            self.create_comments_from_article = lambda x,y: self.create_comments_from_tencent_article(x, y)

        if extra_file is not None:
            self.extra_file = extra_file
            self.stream_extra = open(self.extra_file, encoding='utf8')
        else:
            self.stream_extra = None

    def __iter__(self):
        lines = self.stream.readlines()
        if self.stream_extra is not None:
            lines_ext = self.stream_extra.readlines()

        # next epoch
        if not lines:
            self.epoch_id += 1
            self.stream.close()
            self.stream = open(self.filename, encoding='utf8')
            lines = self.stream.readlines()

            if self.stream_extra is not None:
                self.stream_extra.close()
                self.stream_extra = open(self.extra_file, encoding='utf8')
                lines_ext = self.stream_extra.readlines()

        articles = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            articles.append(json.loads(line))

            if len(articles) > 100 and self.debug:
                break
            if self.train_num > 0:
                if len(articles) >= self.train_num:
                    break
        # random.shuffle(articles)
        # read extra data
        if self.stream_extra is not None:
            articles_ext = []
            for line in lines_ext:
                line = line.strip()
                articles_ext.append(json.loads(line))

                if len(articles_ext) > 100 and self.debug:
                    break
                if self.train_num > 0:
                    if len(articles_ext) >= self.train_num:
                        break
            assert len(articles) == len(articles_ext)

        data = []
        if self.stream_extra is None:
            for idx, doc in enumerate(articles):
                data.extend(self.create_comments_from_article(doc, None))
        else:
            for idx, (doc, doc_extra) in enumerate(zip(articles, articles_ext)):
                data.extend(self.create_comments_from_article(doc, doc_extra))
        if self.is_train:
            random.shuffle(data)

        idx = 0
        while idx < len(data):
            example_list = self.covert_json_to_example(data[idx:idx + self.batch_size])
            yield Batch(example_list, self.is_train, self.model)
            idx += self.batch_size

    def create_comments_from_tencent_article(self, article, article_ext=None):
        comments = []
        if self.is_train:
            for i in range(len(article['comment'])):
                item = dict()
                item['title'] = article['title']
                item['body'] = article['body']
                item['comment'] = article['comment'][i][0]
                comments.append(item)

                if article_ext is not None:
                    item['content_label'] = article_ext['com_labels'][i]

                if 0 < self.config.max_comment_num <= len(comments):
                    break
        else:
            # contain one article and multi comments
            item = dict()
            item['id'] = article['id']
            item['title'] = article['title']
            item['body'] = article['body']
            item['comment'] = [c[0] for c in article['comment'][:5]]
            comments.append(item)
        return comments

    def create_comments_from_yahoo_article(self, article, article_ext=None):
        comments = []
        if self.is_train:
            for i in range(len(article['cmts'])):
                item = dict()
                item['title'] = article['title']
                item['body'] = ' '.join(article['paras'])
                item['comment'] = article['cmts'][i]['cmt']
                comments.append(item)

                if article_ext is not None:
                    item['content_label'] = article_ext['com_labels'][i]

                if 0 < self.config.max_comment_num <= len(comments):
                    break
        else:
            # contain one article and multi comments
            item = dict()
            item['id'] = article['_id']
            item['title'] = article['title']
            item['body'] = ' '.join(article['paras'])
            item['comment'] = [c['cmt'] for c in article['cmts'][:5]]
            comments.append(item)
        return comments

    def create_comments_from_163_article(self, article, article_ext=None):
        comments = []
        if self.is_train:
            for i in range(len(article['cmts'])):
                item = dict()
                item['title'] = article['title']
                item['body'] = article['body']
                item['comment'] = article['cmts'][i]['cmt']
                comments.append(item)

                if article_ext is not None:
                    item['content_label'] = article_ext['com_labels'][i]

                if 0 < self.config.max_comment_num <= len(comments):
                    break
        else:
            # contain one article and multi comments
            item = dict()
            item['id'] = article['_id']
            item['title'] = article['title']
            item['body'] = article['body']
            item['comment'] = [c['cmt'] for c in article['cmts'][:5]]
            comments.append(item)
        return comments

    def covert_json_to_example(self, json_list):
        results = []
        for g in json_list:
            if self.is_train:
                target = g['comment'].split()
            else:
                # multi comments for each article
                target = [s.split() for s in g['comment']]

            title = g["title"].split()
            original_content = g["body"].split()

            content_label = g['content_label'] if 'content_label' in g else None

            news_id = None if self.is_train else g["id"]

            e = Example(original_content, title, target, self.vocab, self.is_train, news_id=news_id, model=self.model, content_label=content_label)
            results.append(e)
        return results


def data_stats(fname, is_test):
    content_word_num = []
    content_char_num = []
    title_word_num = []
    title_char_num = []
    comment_word_num = []
    comment_char_num = []
    keyword_num = []
    urls = {}

    for line in open(fname, "r"):
        g = json.loads(line)
        url = g["url"]
        if url not in urls:
            urls[url] = 0
        if is_test:
            targets = [s.split() for s in g["label"].split("$$")]
            urls[url] += len(targets)
            for target in targets:
                comment_word_num.append(len(target))
                comment_char_num.append(len("".join(target)))
        else:
            urls[url] += 1
            target = g["label"].split()
            comment_word_num.append(len(target))
            comment_char_num.append(len("".join(target)))
        title = g["title"].split()
        title_word_num.append(len(title))
        title_char_num.append(len("".join(title)))
        original_content = g["text"].split()
        content_word_num.append(len(original_content))
        content_char_num.append(len("".join(original_content)))

        # betweenness = g["g_vertices_betweenness_vec"]
        # pagerank = g["g_vertices_pagerank_vec"]
        # katz = g["g_vertices_katz_vec"]
        concept_names = g["v_names"]
        keyword_num.append(len(concept_names))
        text_features = g["v_text_features_mat"]
        content = []

        adj_numsent = g["adj_mat_numsent"]
        # adj_numsent is a list(list)
        adj_tfidf = g["adj_mat_tfidf"]
    print('number of documents', len(urls))
    print('number of total comments', sum(list(urls.values())))
    print('average number of comments', np.mean(list(urls.values())))
    content_word_num = np.mean(content_word_num)
    content_char_num = np.mean(content_char_num)
    title_word_num = np.mean(title_word_num)
    title_char_num = np.mean(title_char_num)
    comment_word_num = np.mean(comment_word_num)
    comment_char_num = np.mean(comment_char_num)
    keyword_num = np.mean(keyword_num)
    print(
        'average content word number: %.2f, average content character number: %.2f, average title word number: %.2f, '
        % (content_word_num, content_char_num, title_word_num),
        'average title character numerb: %.2f, average comment word number %.2f, average comment character number %.2f'
        % (title_char_num, comment_word_num, comment_char_num),
        'average keyword number %.2f' % keyword_num)


def eval_bow(feature_file, cand_file):
    stop_words = {word.strip() for word in open('stop_words.txt').readlines()}
    contents = []
    for line in open(feature_file, "r"):
        g = json.loads(line)
        contents.append(remove_stopwords(g["text"].split(), stop_words))
    candidates = []
    for line in open(cand_file):
        words = line.strip().split()
        candidates.append(remove_stopwords(words, stop_words))
    assert len(contents) == len(candidates), (len(contents), len(candidates))
    results = []
    for content, candidate in zip(contents, candidates):
        results.append(cosine_sim(bow(content), bow(candidate)))
    return results, np.mean(results)


def eval_unique_words(cand_file):
    stop_words = {word.strip() for word in open('stop_words.txt').readlines()}
    result = set()
    for line in open(cand_file):
        words = set(line.strip().split())
        result.update(words)
    result = result.difference(stop_words)
    return result


def eval_distinct(cand_file):
    unigram, bigram, trigram = set(), set(), set()
    sentence = set()
    for line in open(cand_file):
        words = line.strip().split()
        sentence.add(line)
        unigram.update(set(words))
        for i in range(len(words) - 1):
            bigram.add((words[i], words[i + 1]))
        for i in range(len(words) - 2):
            trigram.add((words[i], words[i + 1], words[i + 2]))
    return unigram, bigram, trigram, sentence


def calc_diversity(texts):
    unigram, bigram, trigram, qugram = set(), set(), set(), set()
    num_tok = 0
    for vec in texts:
        v_len = len(vec)
        num_tok += v_len
        unigram.update(vec)
        bigram.update([tuple(vec[i:i+2]) for i in range(v_len-1)])
        trigram.update([tuple(vec[i:i + 3]) for i in range(v_len - 2)])
        qugram.update([tuple(vec[i:i + 4]) for i in range(v_len - 3)])
    metrics = OrderedDict()
    metrics['d_1'] = round(len(unigram) * 1.0 / num_tok * 100, 6)
    metrics['d_2'] = round(len(bigram) * 1.0 / num_tok * 100, 6)
    metrics['d_3'] = round(len(trigram) * 1.0 / num_tok * 100, 6)
    metrics['d_4'] = round(len(qugram) * 1.0 / num_tok * 100, 6)
    metrics['num_d1'] = len(unigram)
    metrics['num_d2'] = len(bigram)
    metrics['num_d3'] = len(trigram)
    metrics['num_d4'] = len(qugram)
    metrics['num_tok'] = num_tok
    metrics['sen_len'] = round(num_tok * 1.0 / len(texts), 6)
    return metrics


if __name__ == '__main__':
    '''
    print('entertainment')
    data_stats('./data/train_graph_features.json', False)
    data_stats('./data/dev_graph_features.json', True)
    print('sport')
    data_stats('./sport_data/train_graph_features.json', False)
    data_stats('./sport_data/dev_graph_features.json', True)
    '''
    topic = sys.argv[1]
    cand_log = sys.argv[2]
    # print(eval_bow(os.path.join(topic, 'dev_graph_features.json'), os.path.join(topic, 'log', cand_log, 'candidate.txt'))[1])
    unigram, bigram, trigram, sentence = eval_distinct(os.path.join(topic, 'log', cand_log, 'candidate.txt'))
    print('unigram', len(unigram), 'bigram', len(bigram), 'trigram', len(trigram), 'sentence', len(sentence))
    print(len(eval_unique_words(os.path.join(topic, 'log', cand_log, 'candidate.txt'))))
