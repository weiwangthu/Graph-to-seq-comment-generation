# coding: utf-8
from gensim.models import KeyedVectors

input_path = '../ft_local/data-with-body/data.train.json.word_vec'
word_vectors = KeyedVectors.load(input_path, mmap='r')
result = word_vectors.similar_by_word("青春")
for item in result:
    print("{}: {:.4f}".format(*item))
# word_vectors.save_word2vec_format(fname='../Data/vec.txt', binary=False)