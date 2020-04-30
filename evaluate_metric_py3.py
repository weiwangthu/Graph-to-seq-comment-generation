# -*- coding:utf-8 -*-
import sys
import codecs
from nlgeval import NLGEval
import json
from collections import OrderedDict

base_path = '/apdcephfs/share_916081/shared_info/rickwwang_ckpt/Multi-comment-generation'
ref_info = {
    'tencent': base_path + '/ft_local/data-with-body/data.test.json',
    'yahoo': base_path + '/ft_local/release_v1/test.data_v1',
    '163': base_path + '/ft_local/163/test.json',
}

def load_reference_data(path, data_set='tencent'):
    references = []
    for line in open(path, encoding='utf8'):
        g = json.loads(line.strip())
        if data_set == 'tencent':
            comments = [com[0] for com in g["comment"]]
        else:
            comments = [com['cmt'] for com in g["cmts"]]
        references.append(comments)
    return references


def calc_diversity(texts):
    unigram, bigram, trigram, qugram = set(), set(), set(), set()
    num_tok = 0
    for vec in texts:
        vec = vec.split()
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

def has_two(pair):
    return len(pair[0].strip().split('\t')) > 1

def convert_hyp(text):
    text = text.strip().split('\t')[1]
    return text


hyp_path = sys.argv[1]
data_set = sys.argv[2]
hypos = codecs.open(hyp_path, 'r', encoding='utf8').readlines()
total = len(hypos)
print('total number of test example', total)
refers = load_reference_data(ref_info[data_set], data_set)
assert len(refers) == len(hypos)

pair = zip(hypos, refers)
pair = list(filter(has_two, pair))
hypos, refers = zip(*pair)
print('break test number:', total - len(hypos))
hypos = list(map(convert_hyp, hypos))
print('correct number of test example', len(hypos))
print('hyp')
print(hypos[0])


nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)  # loads the models
metrics_dict = nlgeval.compute_metrics(refers, hypos)
over_lap_text_result = ','.join('{:s}={:.6f}'.format(key, metrics_dict[key]) for key in metrics_dict.keys())
print(over_lap_text_result)

metrics = calc_diversity(hypos)
dis_text_result = ','.join('{:s}={:.6f}'.format(key, metrics[key]) for key in metrics.keys())
print(dis_text_result)

with open(hyp_path + '.metric', 'a') as f:
    f.write('\n'.join([over_lap_text_result, dis_text_result]))
    f.write('\n')




