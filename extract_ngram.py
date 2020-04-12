import argparse
import os
import json
import time
import codecs

import util

stop_words = {word.strip() for word in open('stop_words.txt').readlines()}
stop_words.add(',')

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', default='config.yaml', type=str,
                        help="config file")
    return parser.parse_args()

def extract_one_article(article):
    result = {}
    result['id'] = article['id']
    result['com_labels'] = []
    result['com_texts'] = []

    title_words = article['title'].split()
    body_words = article['body'].split()
    content_words = title_words + body_words
    len_content = len(content_words)

    # collect ngram
    for com in article['comment']:
        com_words = com[0].split()
        len_com = len(com_words)
        n_grams_set = [set() for _ in range(6)]
        for ng_i in range(6):
            gram_len = ng_i + 1
            if len_com >= gram_len:
                if gram_len == 1:
                    n_grams_set[ng_i].update([w for w in com_words if w not in stop_words])
                else:
                    temp = [tuple(com_words[i:i+gram_len]) for i in range(len_com-gram_len+1)
                            if com_words[i] not in stop_words and com_words[i+gram_len-1] not in stop_words]
                    n_grams_set[ng_i].update(temp)

        # give label
        cur_label = []
        cur_text = []
        start = 0
        ngram_lens = [6, 5, 4, 3, 2, 1]
        # skip empty ngram check
        no_empty_ngram_lens = [ng_len for ng_len in ngram_lens if len(n_grams_set[ng_len-1]) > 0]
        if len(no_empty_ngram_lens) == 0:
            cur_label = [-1]
            cur_text = ['null']
        else:
            while start < len_content:
                for ng_len in no_empty_ngram_lens:
                    end = start+ng_len
                    if end <= len_content:
                        span = tuple(content_words[start: end]) if ng_len > 1 else content_words[start]
                        if span in n_grams_set[ng_len-1]:
                            cur_label.extend([ind for ind in range(start, end)])
                            span_text = ' '.join(content_words[start: end]) if ng_len > 1 else content_words[start]
                            cur_text.append(span_text)
                            start = end
                            break
                else:
                    # cur_label.append(0)
                    start = start + 1

            # assert len(cur_label) == len_content
            if len(cur_label) == 0:
                cur_label = [-1]
                cur_text = ['null']
            # else:
            #     cur_label = list(map(lambda x: str(x), cur_label))
            #     cur_label = ','.join(cur_label)
            #     cur_text = ','.join(cur_text)
        result['com_labels'].append(cur_label)
        result['com_texts'].append(cur_text)
    assert len(result['com_labels']) == len(article['comment'])
    return result

def extract_ngram(articles):
    results = []
    start_time = time.time()
    for ar in articles:
        res = extract_one_article(ar)
        results.append(json.dumps(res))
        if len(results) % 1000 == 0:
            span = time.time() - start_time
            print(span, len(results))
        # if len(results) > 10:
        #     break
    return results

def read_article(path):
    with open(path, encoding='utf8') as fin:
        for line in fin:
            item = json.loads(line.strip(), encoding='utf-8')
            yield item

def save_ngram(lines, path):
    with codecs.open(path, 'w', 'utf-8') as fout:
        fout.write('\n'.join(lines))
        fout.write('\n')


if __name__ == "__main__":
    args = parse_config()
    config = util.utils.read_config(args.config)

    corpus_files = [config.train_file, config.valid_file, config.test_file]
    for corpus in corpus_files:
        articles = read_article(corpus)
        ngrams = extract_ngram(articles)
        save_ngram(ngrams, corpus + '.ngram')
