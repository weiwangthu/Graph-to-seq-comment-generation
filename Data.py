import torch
import random
import copy
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
MAX_COMMENT_NUM = 5


class Vocab:
    def __init__(self, vocab_file, vocab_size=50000):
        self._word2id = {'[PADDING]': 0, '[START]': 1, '[END]': 2, '[OOV]': 3, '[MASK]': 4, '_TITLE_': 5}
        self._id2word = ['[PADDING]', '[START]', '[END]', '[OOV]', '[MASK]', '_TITLE_']
        self._wordcount = {'[PADDING]': 1, '[START]': 1, '[END]': 1, '[OOV]': 1, '[MASK]': 1, '_TITLE_': 1}
        self.load_vocab(vocab_file, vocab_size)
        self.voc_size = len(self._word2id)
        self.UNK_token = 3
        self.PAD_token = 0

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

    def sent2id(self, sent, add_start=False, add_end=False):
        result = [self.word2id(word) for word in sent]
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

    def __init__(self, original_content, title, target, vocab, is_train):
        self.ori_title = title[:MAX_TITLE_LENGTH]
        self.ori_original_content = original_content[:MAX_ARTICLE_LENGTH]
        if is_train:
            self.ori_target = target[:MAX_COMMENT_LENGTH]
        else:
            self.ori_targets = [tar[:MAX_COMMENT_LENGTH] for tar in target]

        self.title = vocab.sent2id(self.ori_title)
        self.original_content = vocab.sent2id(self.ori_original_content)
        if is_train:
            self.target = vocab.sent2id(self.ori_target, add_start=True, add_end=True)

        self.sentence_content = split_chinese_sentence(self.ori_original_content)
        self.sentence_content = [vocab.sent2id(sen[:MAX_ARTICLE_LENGTH]) for sen in self.sentence_content]
        self.sentence_content_max_len = Batch.get_length(self.sentence_content, MAX_ARTICLE_LENGTH)
        self.sentence_content, self.sentence_content_mask = Batch.padding_list_to_tensor(self.sentence_content, self.sentence_content_max_len.max().item())

        self.bow = self.bow_vec(self.original_content, MAX_ARTICLE_LENGTH)

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

        if is_train:
            self.tgt_len = self.get_length([e.target for e in example_list])
            self.tgt, self.tgt_mask = self.padding_list_to_tensor([e.target for e in example_list], self.tgt_len.max().item())

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
    def __init__(self, filename, batch_size, vocab, adj_type, use_gnn, model, is_train=True, debug=False):
        self.batch_size = batch_size
        self.vocab = vocab
        # self.max_len = MAX_LENGTH
        self.filename = filename
        self.stream = open(self.filename, encoding='utf8')
        self.epoch_id = 0

        self.is_train = is_train
        self.debug = debug

        self.adj_type = adj_type
        self.use_gnn = use_gnn
        self.model = model

    def __iter__(self):
        lines = self.stream.readlines()
        if not lines:
            self.epoch_id += 1
            self.stream.close()
            self.stream = open(self.filename, encoding='utf8')
            lines = self.stream.readlines()

        articles = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            articles.append(json.loads(line))

            if len(articles) > 100 and self.debug:
                break
        # random.shuffle(articles)

        data = []
        for idx, doc in enumerate(articles):
            data.extend(self.create_comments_from_article(doc))
        if self.is_train:
            random.shuffle(data)

        idx = 0
        while idx < len(data):
            example_list = self.covert_json_to_example(data[idx:idx + self.batch_size])
            yield Batch(example_list, self.is_train, self.model)
            idx += self.batch_size

    def create_comments_from_article(self, article):
        comments = []
        if self.is_train:
            for i in range(len(article['comment'])):
                item = dict()
                item['title'] = article['title']
                item['body'] = article['body']
                item['comment'] = article['comment'][i][0]
                comments.append(item)

                if len(comments) >= MAX_COMMENT_NUM:
                    break
        else:
            # contain one article and multi comments
            article['comment'] = article['comment'][:5]
            comments.append(article)
        return comments

    def covert_json_to_example(self, json_list):
        results = []
        for g in json_list:
            if self.is_train:
                target = g['comment'].split()
            else:
                # multi comments for each article
                target = [s[0].split()for s in g['comment']]

            title = g["title"].split()
            original_content = g["body"].split()

            e = Example(original_content, title, target, self.vocab, self.is_train)
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
