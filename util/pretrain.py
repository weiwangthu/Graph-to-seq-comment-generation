import gensim, logging
import json

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_doc(input_path):
    all_line = []
    for line in open(input_path, 'r'):
        item = json.loads(line.strip())
        comments = [c[0].split() for c in item['comment']]
        all_line.extend(comments)
    return all_line


input_path = '../ft_local/data-with-body/data.train.json'
documents = get_doc(input_path)
model = gensim.models.Word2Vec(documents, sg=0, size=300, window=8, negative=25, hs=0, sample=1e-4, iter=15, min_count=5, workers=8)
model.train(documents, total_examples=len(documents), epochs=15)
model.wv.save(input_path + '.word_vec')
