import argparse
import os
import json


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='data_path', type=str, help="config file")
    return parser.parse_args()


def build_vocab(corpus_files, vocab_file):
    word2count = {}
    for corpus_file in corpus_files:
        for line in open(corpus_file):
            # words = line.strip().split()
            g = json.loads(line)
            words = g["body"].split()
            words.extend(g["title"].split())
            words.extend([w for com in g["comment"] for w in com[0].split()])
            for word in words:
                if word not in word2count:
                    word2count[word] = 0
                word2count[word] += 1
    word2count = list(word2count.items())
    word2count.sort(key=lambda k: k[1], reverse=True)
    write = open(vocab_file, 'w')
    for word_pair in word2count:
        write.write(word_pair[0] + '\t' + str(word_pair[1]) + '\n')
    write.close()


if __name__ == "__main__":
    args = parse_config()
    data_path = args.data
    # corpus_files = ['data.train.json']
    corpus_files = ['data.train.json', 'data.dev.json', 'data.test.json']
    corpus_files = [os.path.join(data_path, split) for split in corpus_files]
    vocab_file = os.path.join(data_path, 'vocab.txt')
    build_vocab(corpus_files, vocab_file)
