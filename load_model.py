from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import time
import codecs
from collections import Counter

import torch
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np

import util
from Data import Vocab, DataLoader
from util import utils
from util.nlp_utils import *
from util.misc_utils import move_to_cuda
from models import *
from gensim.models import KeyedVectors


# config
def parse_args():
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-gpus', default=[1], type=int, nargs='+',
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-seed', type=int, default=1234,
                        help="Random seed")

    parser.add_argument('-config', default='config.yaml', type=str,
                        help="config file")
    parser.add_argument('-model', default='graph2seq', type=str,
                        choices=['seq2seq', 'graph2seq', 'bow2seq', 'h_attention', 'select_diverse2seq',
                                 'select2seq', 'select_var_diverse2seq',
                                 'var_select_var_diverse2seq', 'var_select_var_user_diverse2seq',
                                 'select2seq_test', 'var_select_var_user_diverse2seq_test',
                                 'var_select_var_user_diverse2seq_test2', 'var_select_var_user_diverse2seq_test3',
                                 'var_select_var_user_diverse2seq_test4',
                                 'var_select2seq_test', 'user2seq_test', 'var_select_user2seq_test',
                                 'autoenc', 'user_autoenc', 'user_autoenc_vae', 'user_autoenc_near',
                                 'autoenc_lm', 'autoenc_vae', 'autoenc_vae_bow', 'autoenc_vae_cat',
                                 'user_autoenc_vae_bow',
                                 ])
    parser.add_argument('-adj', type=str, default="numsent",
                        help='adjacent matrix')
    parser.add_argument('-use_copy', default=False, action="store_true",
                        help='whether to use copy mechanism')
    parser.add_argument('-use_bert', default=False, action="store_true",
                        help='whether to use bert in the encoder')
    parser.add_argument('-use_content', default=False, action="store_true",
                        help='whether to use title in the seq2seq')
    parser.add_argument('-word_level_model', default='bert', choices=['bert', 'memory', 'word'],
                        help='whether to use bert or memory network or nothing in the word level of encoder')
    parser.add_argument('-graph_model', default='none', choices=['GCN', 'GNN', 'none'],
                        help='whether to use gcn in the encoder')
    parser.add_argument('-drop_dec_input', default=False, action="store_true",
                        help='whether to use title in the seq2seq')

    parser.add_argument('-notrain', default=False, action='store_true',
                        help="train or not")
    parser.add_argument('-restore', type=str, default=None,
                        help="restore checkpoint")
    parser.add_argument('-beam_search', default=False, action='store_true',
                        help="beam_search")
    parser.add_argument('-n_best', type=int, default=5,
                        help="beam_search")
    parser.add_argument('-n_topic', type=int, default=5,
                        help="beam_search")

    parser.add_argument('-log', default='', type=str,
                        help="log directory")
    parser.add_argument('-verbose', default=False, action='store_true',
                        help="verbose")
    parser.add_argument('-debug', default=False, action="store_true",
                        help='whether to use debug mode')
    parser.add_argument('-train_num', type=int, default=0,
                        help='whether to use debug mode')

    group = parser.add_argument_group('Hyperparameter')
    group.add_argument('-n_z', type=int, default=64, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('-n_topic_num', type=int, default=10, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('-tau', type=float, default=0.5, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('-gama1', type=float, default=0.0, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('-gama_kld', type=float, default=0.05, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('-gama_select', type=float, default=0.0, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('-gama_rank', type=float, default=1.0, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('-gama_reg', type=float, default=1.0, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('-min_select', type=float, default=0.0, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('-topic', default=False, action="store_true",
                       help='save a checkpoint every N epochs')
    group.add_argument('-one_user', default=False, action="store_true",
                       help='save a checkpoint every N epochs')
    group.add_argument('-dynamic1', default=False, action="store_true",
                       help='save a checkpoint every N epochs')
    group.add_argument('-dynamic2', default=False, action="store_true",
                       help='save a checkpoint every N epochs')
    group.add_argument('-mid_max', type=float, default=20.0, metavar='N',
                       help='save a checkpoint every N epochs')

    opt = parser.parse_args()
    config = util.utils.read_config(opt.config)

    # overwrite config file
    d = vars(opt)
    for k,v in d.items():
        if k in config:
            print('overwirte: ', k, v)
            config[k] = v

    return opt, config


# set opt and config as global variables
args, config = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)


# Training settings
def set_up_logging():
    # log为记录文件
    # config.log是记录的文件夹, 最后一定是/
    # opt.log是此次运行时记录的文件夹的名字
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    if args.log == '':
        log_path = config.log_dir + utils.format_time(time.localtime()) + '/'
    else:
        log_path = config.log_dir + args.log + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logging = utils.logging(log_path + 'log.txt')  # 往这个文件里写记录
    logging_csv = utils.logging_csv(log_path + 'record.csv')  # 往这个文件里写记录
    for k, v in config.items():
        logging("%s:\t%s\n" % (str(k), str(v)))
    logging("\n")
    return logging, logging_csv, log_path


logging, logging_csv, log_path = set_up_logging()
use_cuda = torch.cuda.is_available()


def get_sentence_embedding(list_sentences, words_dict):
    embeds = []
    for words in list_sentences:
        semb = []
        for word in words:
            if word in words_dict.vocab:
                embed = words_dict[word]
                semb.append(embed)
        if len(semb) == 0:
            emb = np.zeros(300, dtype=np.float32)
        else:
            emb = np.sum(semb, axis=0) / len(semb)
        embeds.append(emb)
    return np.array(embeds, dtype=np.float32)


def get_input_data_from_emb(data_loader, comment_num):
    data_path = log_path + 'data_for_cluster.num%d.emb' % comment_num
    word2vector_path = './ft_local/data-with-body/data.train.json.word_vec'
    x_texts = []
    # get comment vectors
    for batch in data_loader:
        # collect comment texts
        batch_texts = [batch.examples[bid].ori_target for bid in range(len(batch.examples))]
        x_texts.extend(batch_texts)
        if len(x_texts) > comment_num:
            break

    is_reload_data = os.path.exists(data_path + '.npy')
    if not is_reload_data:
        print('load pre trained word vector: %s' % word2vector_path)
        words_dict = KeyedVectors.load(word2vector_path, mmap='r')
        print('convert data')
        X = get_sentence_embedding(x_texts, words_dict)
        print('save input data: %s' % data_path)
        np.save(data_path, X)
    else:
        print('reload input data: %s' % data_path)
        X = np.load(data_path + '.npy').astype(np.float32)

    return X, x_texts

def get_input_data_from_model(data_loader, model, epoch, comment_num):
    data_path = log_path + 'data_for_cluster.num%d.e%d' % (comment_num, epoch)
    x_texts = []
    is_reload_data = os.path.exists(data_path + '.npy')
    if is_reload_data:
        print('reload input data: %s' % data_path)
        X = np.load(data_path + '.npy').astype(np.float32)
    else:
        X = []
    # get comment vectors
    for batch in data_loader:
        if not is_reload_data:
            # model vetor
            z = model.get_comment_rep(batch, use_cuda)
            # org bow vector
            # z = batch.tgt_bow

            X.extend(z.tolist())

        # collect comment texts
        batch_texts = [batch.examples[bid].ori_target for bid in range(len(batch.examples))]
        x_texts.extend(batch_texts)
        if len(x_texts) > comment_num:
            break
    if not is_reload_data:
        print('save input data: %s' % data_path)
        X = np.array(X, dtype=np.float32)
        np.save(data_path, X)

    return X, x_texts

def cluster(data_loader, model, epoch):
    print('get input data from model')
    comment_num = 100000
    X, x_texts = get_input_data_from_model(data_loader, model, epoch, comment_num)
    # X, x_texts = get_input_data_from_emb(data_loader, comment_num)

    print('start k-means')
    # k-means cluster
    y_pred = KMeans(n_clusters=config.n_topic_num, random_state=0).fit_predict(X)
    ch_score = metrics.calinski_harabasz_score(X, y_pred)
    # score_str = 'ch_score: %s' % str(ch_score)
    sc_score = metrics.silhouette_score(X, y_pred, metric='euclidean')
    score_str = 'ch_score: %s, sc_score: %s' % (str(ch_score), str(sc_score))

    print('save cluster result')
    collect_result = [[] for _ in range(config.n_topic_num)]
    for text, label in zip(x_texts, y_pred):
        collect_result[label].append(text)

    # debug for selected_user
    result_str = []
    result_str.append(score_str)
    collect_ids = []
    # save average comment lengths of every topic
    for ii in range(len(collect_result)):
        comment_count = len(collect_result[ii])
        if comment_count > 0:
            len_list = [len(com) for com in collect_result[ii]]
            comment_len = sum(len_list) / comment_count
            comment_min = min(len_list)
            comment_max = max(len_list)
            result_str.append('\t'.join([str(ii), str(comment_count),
                                         str(comment_len), str(comment_min), str(comment_max)]))
            collect_ids.append(ii)
    with codecs.open(log_path + 'topic_comment.nu%d.nt%s.e%s.statistic' % (comment_num, str(config.n_topic_num), str(epoch)), 'w', 'utf-8') as f:
        f.write('\n'.join(result_str))
        f.write('\n')

    collect_num = 40
    for uid in collect_ids[:collect_num]:
        # save topic comments
        topic_comments = list(map(lambda x: ''.join(x), collect_result[uid]))[:500]
        with codecs.open(log_path + 'topic_comment.nu%d.nt%s.e%s.i%s' % (comment_num, str(config.n_topic_num), str(epoch), str(uid)), 'w', 'utf-8') as f:
            f.write('\n'.join(topic_comments))
            f.write('\n')

        # save topic words
        topic_words = [w for com in collect_result[uid] for w in com]
        topic_words = Counter(topic_words)
        topic_words = topic_words.most_common(1000)
        topic_words = list(map(lambda x: '\t'.join([str(x[0]), str(x[1])]), topic_words))
        with codecs.open(log_path + 'topic_comment.nu%d.nt%s.e%s.i%s.word' % (comment_num, str(config.n_topic_num), str(epoch), str(uid)), 'w', 'utf-8') as f:
            f.write('\n'.join(topic_words))
            f.write('\n')
    exit(0)


def main():
    vocab = Vocab(config.vocab_file, config.vocab_size)

    # model
    print('building model...\n')
    # configure the model
    # Model and optimizer
    if args.model == 'graph2seq':
        model = graph2seq(config, vocab, use_cuda, args.use_copy, args.use_bert, args.word_level_model, args.graph_model)
    elif args.model == 'seq2seq':
        model = seq2seq(config, vocab, use_cuda, use_content=args.use_content)
    elif args.model == 'bow2seq':
        model = bow2seq(config, vocab, use_cuda)
    elif args.model == 'h_attention':
        model = hierarchical_attention(config, vocab, use_cuda)
    elif args.model == 'select_diverse2seq':
        model = select_diverse2seq(config, vocab, use_cuda)
    elif args.model == 'select2seq':
        model = select2seq(config, vocab, use_cuda)
    elif args.model == 'select_var_diverse2seq':
        model = select_var_diverse2seq(config, vocab, use_cuda)
    elif args.model == 'var_select_var_diverse2seq':
        model = var_select_var_diverse2seq(config, vocab, use_cuda)
    elif args.model == 'var_select_var_user_diverse2seq':
        model = var_select_var_user_diverse2seq(config, vocab, use_cuda)
    elif args.model == 'select2seq_test':
        model = select2seq_test(config, vocab, use_cuda)
    elif args.model == 'var_select_var_user_diverse2seq_test':
        model = var_select_var_user_diverse2seq_test(config, vocab, use_cuda)
    elif args.model == 'var_select_var_user_diverse2seq_test2':
        model = var_select_var_user_diverse2seq_test2(config, vocab, use_cuda)
    elif args.model == 'var_select_var_user_diverse2seq_test3':
        model = var_select_var_user_diverse2seq_test3(config, vocab, use_cuda)
    elif args.model == 'var_select_var_user_diverse2seq_test4':
        model = var_select_var_user_diverse2seq_test4(config, vocab, use_cuda)
    elif args.model == 'var_select2seq_test':
        model = var_select2seq_test(config, vocab, use_cuda)
    elif args.model == 'var_select_user2seq_test':
        model = var_select_user2seq_test(config, vocab, use_cuda)
    elif args.model == 'user2seq_test':
        model = user2seq_test(config, vocab, use_cuda)
    elif args.model == 'autoenc':
        model = autoenc(config, vocab, use_cuda)
    elif args.model == 'user_autoenc':
        model = user_autoenc(config, vocab, use_cuda)
    elif args.model == 'user_autoenc_vae':
        model = user_autoenc_vae(config, vocab, use_cuda)
    elif args.model == 'user_autoenc_near':
        model = user_autoenc_near(config, vocab, use_cuda)
    elif args.model == 'autoenc_lm':
        model = autoenc_lm(config, vocab, use_cuda)
    elif args.model == 'autoenc_vae':
        model = autoenc_vae(config, vocab, use_cuda)
    elif args.model == 'autoenc_vae_bow':
        model = autoenc_vae_bow(config, vocab, use_cuda)
    elif args.model == 'autoenc_vae_cat':
        model = autoenc_vae_cat(config, vocab, use_cuda)
    elif args.model == 'user_autoenc_vae_bow':
        model = user_autoenc_vae_bow(config, vocab, use_cuda)

    # total number of parameters
    print(repr(model))
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    print('total number of parameters: %d\n\n' % param_count)

    print('loading checkpoint...')
    checkpoints = torch.load(os.path.join(log_path, args.restore))
    model.load_state_dict(checkpoints['model'])
    model.eval()
    epoch = checkpoints['epcoh']
    # set cuda
    if use_cuda:
        model.cuda()

    use_gnn = False
    if args.graph_model == 'GNN':
        use_gnn = True
    # load data
    train_data = DataLoader(config.train_file, config.batch_size, vocab, args.adj, use_gnn, args.model, True, args.debug, args.train_num)
    valid_data = DataLoader(config.valid_file, config.batch_size, vocab, args.adj, use_gnn, args.model, True, args.debug, args.train_num)
    test_data = DataLoader(config.test_file, config.max_generator_batches, vocab, args.adj, use_gnn, args.model, False, args.debug, args.train_num)

    cluster(train_data, model, epoch)

if __name__ == '__main__':
    main()
