from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import time
import collections
import codecs
import math
from collections import Counter

import torch
import torch.nn as nn
from tqdm import tqdm
from torch import autograd

import lr_scheduler as L
import util
from optims import Optim
from util import utils
from Data import Vocab, DataLoader, calc_diversity
from models import *

from util.nlp_utils import *
from util.misc_utils import AverageMeter
from util.misc_utils import load_pretrained_weight


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
                        choices=['seq2seq', 'graph2seq', 'bow2seq', 'h_attention', 'seq2gateseq', 'cvae',
                                 'select_diverse2seq', 'select2seq', 'select_var_diverse2seq',
                                 'var_select_var_diverse2seq', 'var_select_var_user_diverse2seq',
                                 'select2seq_test', 'var_select_var_user_diverse2seq_test',
                                 'var_select_var_user_diverse2seq_test2', 'var_select_var_user_diverse2seq_test3',
                                 'var_select_var_user_diverse2seq_test4',
                                 'var_select2seq_test', 'user2seq_test', 'var_select_user2seq_test',
                                 'autoenc', 'user_autoenc', 'user_autoenc_vae', 'user_autoenc_near',
                                 'autoenc_lm', 'autoenc_vae', 'autoenc_vae_bow', 'autoenc_vae_cat',
                                 'user_autoenc_vae_bow', 'autoenc_vae_bow_norm', 'user_autoenc_vae_bow_norm',
                                 'user2seq_test_new', 'var_select_user2seq_new', 'var_select2seq_test_new',
                                 'user2seq_expand', 'var_select_expand_user2seq',
                                 'var_select2seq_align', 'var_select2seq_test_span', 'var_select2seq_test_span2',
                                 'var_select_user2seq_new2', 'var_select2seq_test_span3',
                                 'select2seq_label', 'var_select_user2seq_label',
                                 'select2seq_encode', 'select2seq_encode2', 'user_autoenc_vae_bow2',
                                 'var_select_user2seq_new3', 'user_autoenc_vae_bow3',
                                 'user2seq_test_new2', 'user2seq_test_new22', 'var_select_user2seq_new22'
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
    parser.add_argument('-use_label', default=False, action="store_true",
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
    parser.add_argument('-load_pre', type=str, default=None,
                        help="restore checkpoint")
    parser.add_argument('-gate_prob', type=float, default=0.5,
                        help="beam_search")
    parser.add_argument('-debug_select', default=False, action="store_true",
                        help='save a checkpoint every N epochs')
    parser.add_argument('-topic', default=False, action="store_true",
                        help='save a checkpoint every N epochs')
    parser.add_argument('-topic_content', default=False, action="store_true",
                        help='save a checkpoint every N epochs')
    parser.add_argument('-no_topk', default=False, action="store_true",
                        help='save a checkpoint every N epochs')
    parser.add_argument('-topk_num', type=int, default=5, metavar='N',
                        help='save a checkpoint every N epochs')
    parser.add_argument('-topic_vec', default=False, action="store_true",
                        help='save a checkpoint every N epochs')
    parser.add_argument('-debug_select_topic', default=False, action="store_true",
                        help='save a checkpoint every N epochs')
    parser.add_argument('-collect_num', type=int, default=20, metavar='N',
                        help='save a checkpoint every N epochs')

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
    group.add_argument('-gama_kld_select', type=float, default=0.0, metavar='N',
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
    group.add_argument('-one_user', default=False, action="store_true",
                       help='save a checkpoint every N epochs')
    group.add_argument('-con_one_user', default=False, action="store_true",
                       help='save a checkpoint every N epochs')
    group.add_argument('-dynamic1', default=False, action="store_true",
                       help='save a checkpoint every N epochs')
    group.add_argument('-dynamic2', default=False, action="store_true",
                       help='save a checkpoint every N epochs')
    group.add_argument('-dynamic3', default=False, action="store_true",
                       help='save a checkpoint every N epochs')
    group.add_argument('-mid_max', type=float, default=20.0, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('-opt_join', default=False, action="store_true",
                       help='save a checkpoint every N epochs')
    group.add_argument('-opt_con', default=False, action="store_true",
                       help='save a checkpoint every N epochs')
    group.add_argument('-topic_min_select', type=float, default=0.0, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('-dynamic4', default=False, action="store_true",
                       help='save a checkpoint every N epochs')
    group.add_argument('-gama_con_select', type=float, default=0.0, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('-use_post_user', default=False, action="store_true",
                       help='save a checkpoint every N epochs')
    group.add_argument('-use_post_gate', default=False, action="store_true",
                       help='save a checkpoint every N epochs')
    group.add_argument('-dynamic_tau', default=False, action="store_true",
                       help='save a checkpoint every N epochs')
    group.add_argument('-content_span', type=int, default=20, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('-gama_label', type=float, default=0.0, metavar='N',
                       help='save a checkpoint every N epochs')
    group.add_argument('-fix_gate', default=False, action="store_true",
                       help='save a checkpoint every N epochs')
    group.add_argument('-dynamic5', default=False, action="store_true",
                       help='save a checkpoint every N epochs')
    group.add_argument('-gama_kld_cvae', type=float, default=0.0, metavar='N',
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


def train(model, vocab, train_data, valid_data, scheduler, optim, org_epoch, updates, org_best_score=None):
    scores = []
    best_score = org_best_score if org_best_score is not None else 10000.
    tau = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    tau += [0.05] * 31
    for epoch in range(org_epoch + 1, org_epoch + config.epoch + 1):
        total_acc = 0.
        total_loss = 0.
        local_updates = 0
        start_time = time.time()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        # debug for selected_user
        collect_result = [[] for _ in range(10)]

        if config.schedule:
            scheduler.step(epoch-1)
            print("Decaying learning rate to %g" % scheduler.get_lr()[0])

        model.train()

        for batch in tqdm(train_data, disable=not args.verbose):

            if args.dynamic1:
                # loss = opt.rec_coef * rec_loss + kld_coef * kld
                model.gama_kld = config.gama_kld * min(1.0, (math.tanh((updates - 15000.0 * args.mid_max) / 15000.0) + 1) / 2)
            if args.dynamic2:
                model.gama_kld = config.gama_kld * min(1.0, updates/(15000.0 * args.mid_max))
            if args.dynamic3:
                model.gama_select = max(config.gama_select, 1 - updates/(15000.0 * args.mid_max))
            if args.dynamic4:
                model.gama_con_select = max(config.gama_con_select, 1 - updates/(15000.0 * args.mid_max))
            if args.dynamic_tau:
                model.config.tau = tau[epoch]
            if args.dynamic5:
                model.gama_kld_cvae = config.gama_kld_cvae * min(1.0, updates/(15000.0 * args.mid_max))

            # with autograd.detect_anomaly():
            model.zero_grad()
            outputs = model(batch, use_cuda)
            target = batch.tgt
            if use_cuda:
                target = target.cuda()
            if isinstance(outputs, dict):
                result = model.compute_loss(outputs, target.transpose(0, 1)[1:])
                loss = result['loss']
                acc = result['acc']

                # get other loss information
                for k, v in result.items():
                    if k in ['loss', 'acc'] or (hasattr(v, 'numel') and v.numel() > 1):
                        continue  # these are already logged above
                    else:
                        extra_meters[k].update(v.item())
            else:
                loss, acc = model.compute_loss(outputs.transpose(0, 1), target.transpose(0, 1)[1:])
            if torch.isnan(loss):
                raise Exception('nan error')

            # optimizer
            loss.backward()
            optim.step()

            total_loss += loss.data.item()
            total_acc += acc.data.item()
            updates += 1
            local_updates += 1

            # # debug, for saving selected_user of each comment
            # selected_user = result['selected_user'].tolist()
            # for bid in range(len(selected_user)):
            #     collect_result[selected_user[bid]].append(''.join(batch.examples[bid].ori_target))
            # if sum([len(uu) for uu in collect_result]) > 10000:
            #     break

            if updates % config.print_interval == 0 or args.debug:
                logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.3f, train acc: %.3f\n"
                        % (time.time() - start_time, epoch, updates, total_loss / local_updates, total_acc / local_updates))

                # log other loss
                if len(extra_meters) > 0:
                    other_information = ','.join('{:s}={:.3f}'.format(key, extra_meters[key].avg) for key in extra_meters.keys())
                    logging(other_information + '\n')
                if hasattr(model, 'gama_kld'):
                    logging("kld weight: %.6f\n" % model.gama_kld)
                if hasattr(model, 'gama_select'):
                    logging("select weight: %.6f\n" % model.gama_select)
                if hasattr(model, 'gama_con_select'):
                    logging("con select weight: %.6f\n" % model.gama_con_select)

            # if updates % config.eval_interval == 0 or args.debug:
            #     print('evaluating after %d updates...' % updates)
            #     score = eval_loss(model, vocab, valid_data, epoch, updates)
            #     scores.append(score)
            #     if score <= max_bleu:
            #         save_model(log_path + 'checkpoint_best_%d_%d_%f.pt'%(epoch, updates, score), model, optim, epoch, updates)
            #         max_bleu = score
            #
            #     model.train()
            #     # total_loss = 0.
            #     # total_acc = 0.
            #     # start_time = time.time()
            #
            # if updates % config.save_interval == 0:
            #     save_model(log_path + 'checkpoint_%d_%d.pt'%(epoch, updates), model, optim, epoch, updates)

        # # debug for selected_user
        # for uid in range(10):
        #     with codecs.open(log_path + 'topic_comment.' + str(uid), 'w', 'utf-8') as f:
        #         f.write('\n'.join(collect_result[uid]))
        #         f.write('\n')
        # exit(0)

        # log information
        logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.3f, train acc: %.3f\n"
                % (time.time() - start_time, epoch, updates, total_loss / local_updates, total_acc / local_updates))
        # log other loss
        if len(extra_meters) > 0:
            other_information = ','.join('{:s}={:.3f}'.format(key, extra_meters[key].avg) for key in extra_meters.keys())
            logging(other_information + '\n')

        # eval and save model after each epoch
        print('evaluating after %d updates...' % updates)
        score = eval_loss(model, vocab, valid_data, epoch, updates)
        scores.append(score)

        if score <= best_score:
            save_model(log_path + 'checkpoint_best.pt', model, optim, epoch, updates, best_score)
            best_score = score

        # save every epoch
        save_model(log_path + 'checkpoint_last.pt', model, optim, epoch, updates, best_score)
        save_model(log_path + 'checkpoint_%d.pt' % epoch, model, optim, epoch, updates, best_score)
    return best_score


def eval_topic(model, train_data, epoch):
    # debug for selected_user
    collect_result = [[] for _ in range(config.n_topic_num)]

    model.train()
    for batch in tqdm(train_data, disable=not args.verbose):
        # with autograd.detect_anomaly():
        outputs = model(batch, use_cuda)
        target = batch.tgt
        if use_cuda:
            target = target.cuda()
        if isinstance(outputs, dict):
            result = model.compute_loss(outputs, target.transpose(0, 1)[1:])
            loss = result['loss']
        else:
            loss, acc = model.compute_loss(outputs.transpose(0, 1), target.transpose(0, 1)[1:])
        if torch.isnan(loss):
            raise Exception('nan error')

        # debug, for saving selected_user of each comment
        if args.topic_content:
            selected_user = result['con_sel_user'].tolist()
        else:
            selected_user = result['selected_user'].tolist()

        for bid in range(len(selected_user)):
            collect_result[selected_user[bid]].append(batch.examples[bid].ori_target)
        if sum([len(uu) for uu in collect_result]) > min(config.n_topic_num * 1000, 200000):
            break

    # debug for selected_user
    result_str = []
    collect_ids = []
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
    with codecs.open(log_path + 'topic_comment.%s.statistic' % str(epoch), 'w', 'utf-8') as f:
        f.write('\n'.join(result_str))
        f.write('\n')

    collect_num = args.collect_num
    for uid in collect_ids[:collect_num]:
        topic_comments = list(map(lambda x: ''.join(x), collect_result[uid]))
        with codecs.open(log_path + 'topic_comment.%s.%s' % (str(epoch), str(uid)), 'w', 'utf-8') as f:
            f.write('\n'.join(topic_comments))
            f.write('\n')
        topic_words = [w for com in collect_result[uid] for w in com]
        topic_words = Counter(topic_words)
        topic_words = topic_words.most_common(1000)
        topic_words = list(map(lambda x: '\t'.join([str(x[0]), str(x[1])]), topic_words))
        with codecs.open(log_path + 'topic_comment.%s.%s.word' % (str(epoch), str(uid)), 'w', 'utf-8') as f:
            f.write('\n'.join(topic_words))
            f.write('\n')
    exit(0)


def eval_bleu(model, vocab, valid_data, epoch, updates):
    model.eval()
    multi_ref, reference, candidate, source, tags, alignments = [], [], [], [], [], []
    select_content = []

    for batch in tqdm(valid_data, disable=not args.verbose):
        if len(args.gpus) > 1 or not args.beam_search:
            samples, alignment = model.sample(batch, use_cuda)
        else:
            result = model.beam_sample(batch, use_cuda, beam_size=config.beam_size, n_best=args.n_best)
            samples, alignment = result[0], result[1]
            if config.debug_select:
                select_words = result[2]
        '''
        if i == 0:
            print(batch.examples[27].ori_title)
            print(alignment.shape)
            print([d for d in alignment.tolist()[27]])
            return
        '''
        candidate += [[vocab.id2sent(si) for si in s]for s in samples]
        source += [example for example in batch.examples]
        # reference += [example.ori_target for example in batch.examples]
        multi_ref += [example.ori_targets for example in batch.examples]
        if config.debug_select:
            select_content += [vocab.id2sent(s) for s in select_words]
    utils.write_multi_result_to_file(source, candidate, log_path, epoch)
    if config.debug_select:
        utils.write_topic_result_to_file(source, select_content, log_path, epoch, 0, data_type='beam.swords')

    # bleu, best 1
    single_candidate = [c[0] for c in candidate]
    text_result, bleu = utils.eval_multi_bleu(multi_ref, single_candidate, log_path)
    logging_csv([epoch, updates, text_result])
    print(text_result, flush=True)

    # distinct, best 1 and best n
    metrics_best_1 = calc_diversity(single_candidate)
    text_result = ','.join('{:s}={:.6f}'.format(key, metrics_best_1[key]) for key in metrics_best_1.keys())
    logging_csv([epoch, updates, text_result])
    print(text_result, flush=True)

    flatten_candidate = [si for can in candidate for si in can]
    metrics_best_n = calc_diversity(flatten_candidate)
    text_result = ','.join('{:s}={:.6f}'.format(key, metrics_best_n[key]) for key in metrics_best_n.keys())
    logging_csv([epoch, updates, text_result])
    print(text_result, flush=True)
    return bleu


def eval_bleu_with_topic(model, vocab, valid_data, epoch, updates):
    model.eval()
    multi_ref, reference, candidate, source, tags, alignments = [], [], [], [], [], []

    candidate = [[] for _ in range(args.n_topic)]
    select_content = [[] for _ in range(args.n_topic)]
    select_topic_content = [[] for _ in range(args.n_topic)]
    for i in range(args.n_topic):
        print('decode topic %d' % i)
        for batch in tqdm(valid_data, disable=not args.verbose):
            model.get_user.topic_id = i
            if len(args.gpus) > 1 or not args.beam_search:
                samples, alignment = model.sample(batch, use_cuda)
            else:
                result = model.beam_sample(batch, use_cuda, beam_size=config.beam_size)
                samples, alignment = result[0], result[1]
                if config.debug_select:
                    select_words = result[2]['select_words']
                if config.debug_select_topic:
                    select_topics = result[2]['select_topics']
            '''
            if i == 0:
                print(batch.examples[27].ori_title)
                print(alignment.shape)
                print([d for d in alignment.tolist()[27]])
                return
            '''
            # first topic
            if i == 0:
                candidate[i] += [vocab.id2sent(s[0]) for s in samples]
                source += [example for example in batch.examples]
                # reference += [example.ori_target for example in batch.examples]
                multi_ref += [example.ori_targets for example in batch.examples]
            else:
                candidate[i] += [vocab.id2sent(s[0]) for s in samples]

            if config.debug_select:
                select_content[i] += [vocab.id2sent(s) for s in select_words]
            if config.debug_select_topic:
                select_topic_content[i] += [str(s.item()) for s in select_topics]
        # save to file
        utils.write_topic_result_to_file(source, candidate[i], log_path, epoch, i)

        if config.debug_select:
            utils.write_topic_result_to_file(source, select_content[i], log_path, epoch, i, data_type='topic.swords')
        if config.debug_select_topic:
            utils.write_topic_result_to_file(source, select_topic_content[i], log_path, epoch, i, data_type='topic.stopics')

    # bleu, best 1
    text_result, bleu = utils.eval_multi_bleu(multi_ref, candidate[0], log_path)
    logging_csv([epoch, updates, text_result])
    print(text_result, flush=True)

    # distinct, best 1 and best n
    metrics_best_1 = calc_diversity(candidate[0])
    text_result = ','.join('{:s}={:.6f}'.format(key, metrics_best_1[key]) for key in metrics_best_1.keys())
    logging_csv([epoch, updates, text_result])
    print(text_result, flush=True)

    flatten_candidate = [si for tt in range(5) for si in candidate[tt]]
    metrics_best_n = calc_diversity(flatten_candidate)
    text_result = ','.join('{:s}={:.6f}'.format(key, metrics_best_n[key]) for key in metrics_best_n.keys())
    logging_csv([epoch, updates, text_result])
    print(text_result, flush=True)

    # save to one file
    candidate = list(zip(*candidate))
    utils.write_observe_to_file(source, candidate, log_path, epoch)
    return bleu


def eval_topic_vec(model, vocab, valid_data, epoch, updates):
    model.eval()
    multi_ref, reference, candidate, source, tags, alignments = [], [], [], [], [], []

    candidate = [[] for _ in range(args.n_topic)]
    for i in range(args.n_topic):
        print('decode topic %d' % i)
        for batch in tqdm(valid_data, disable=not args.verbose):
            model.get_user.topic_id = i
            samples = model.generate_with_topic(batch, use_cuda)

            # first topic
            if i == 0:
                candidate[i] += [vocab.id2sent(s) for s in samples]
                source += [example for example in batch.examples]
                # reference += [example.ori_target for example in batch.examples]
                multi_ref += [example.ori_targets for example in batch.examples]
            else:
                candidate[i] += [vocab.id2sent(s) for s in samples]

            if len(candidate[i]) > 100:
                break

        # save to file
        utils.write_topic_result_to_file(source, candidate[i], log_path, epoch, i)

    # bleu, best 1
    text_result, bleu = utils.eval_multi_bleu(multi_ref, candidate[0], log_path)
    logging_csv([epoch, updates, text_result])
    print(text_result, flush=True)

    # distinct, best 1 and best n
    metrics_best_1 = calc_diversity(candidate[0])
    text_result = ','.join('{:s}={:.6f}'.format(key, metrics_best_1[key]) for key in metrics_best_1.keys())
    logging_csv([epoch, updates, text_result])
    print(text_result, flush=True)

    flatten_candidate = [si for tt in range(5) for si in candidate[tt]]
    metrics_best_n = calc_diversity(flatten_candidate)
    text_result = ','.join('{:s}={:.6f}'.format(key, metrics_best_n[key]) for key in metrics_best_n.keys())
    logging_csv([epoch, updates, text_result])
    print(text_result, flush=True)

    # save to one file
    candidate = list(zip(*candidate))
    utils.write_observe_to_file(source, candidate, log_path, epoch)
    return bleu

def eval_loss(model, vocab, valid_data, epoch, updates):
    model.eval()

    total_acc = 0.
    total_loss = 0.
    local_updates = 0
    extra_meters = collections.defaultdict(lambda: AverageMeter())
    for batch in tqdm(valid_data, disable=not args.verbose):
        outputs = model(batch, use_cuda)
        target = batch.tgt
        if use_cuda:
            target = target.cuda()
        if isinstance(outputs, dict):
            result = model.compute_loss(outputs, target.transpose(0, 1)[1:])
            loss = result['loss']
            acc = result['acc']

            # get other loss information
            for k, v in result.items():
                if k in ['loss', 'acc'] or v.numel() > 1:
                    continue  # these are already logged above
                else:
                    extra_meters[k].update(v.item())
        else:
            loss, acc = model.compute_loss(outputs.transpose(0, 1), target.transpose(0, 1)[1:])

        total_loss += loss.data.item()
        total_acc += acc.data.item()
        local_updates += 1

    avg_loss, avg_acc = total_loss/local_updates, total_acc/local_updates
    text_result = 'loss=%f, acc=%f' % (avg_loss, avg_acc)
    # log other loss
    if len(extra_meters) > 0:
        other_information = ','.join('{:s}={:.3f}'.format(key, extra_meters[key].avg) for key in extra_meters.keys())
        text_result = ','.join([text_result, other_information])

    logging_csv([epoch, updates, text_result])
    print(text_result, flush=True)
    return avg_loss

def save_model(path, model, optim, epoch, updates, score):
    model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'epcoh': epoch,
        'updates': updates,
        'best_eval_score': score
    }
    torch.save(checkpoints, path)


def main():
    # set seed
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    # autograd.set_detect_anomaly(True)

    vocab = Vocab(config.vocab_file, config.vocab_size)

    # load data
    use_gnn = False
    if args.graph_model == 'GNN':
        use_gnn = True
    train_data = DataLoader(config.train_file, config, vocab, args.adj, use_gnn, args.model,
                            True, args.debug, args.train_num, config.train_label_file if args.use_label else None)
    valid_data = DataLoader(config.valid_file, config, vocab, args.adj, use_gnn, args.model,
                            True, args.debug, args.train_num, config.dev_label_file if args.use_label else None)

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
    elif args.model == 'seq2gateseq':
        model = seq2gateseq(config, vocab, use_cuda)
    elif args.model == 'cvae':
        model = cvae(config, vocab, use_cuda)
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
    elif args.model == 'autoenc_vae_bow' or args.model == 'autoenc_vae_bow_norm':
        model = autoenc_vae_bow(config, vocab, use_cuda)
    elif args.model == 'autoenc_vae_cat':
        model = autoenc_vae_cat(config, vocab, use_cuda)
    elif args.model == 'user_autoenc_vae_bow' or args.model == 'user_autoenc_vae_bow_norm':
        model = user_autoenc_vae_bow(config, vocab, use_cuda)
    elif args.model == 'user2seq_test_new':
        model = user2seq_test_new(config, vocab, use_cuda)
    elif args.model == 'var_select_user2seq_new':
        model = var_select_user2seq_new(config, vocab, use_cuda)
    elif args.model == 'var_select2seq_test_new':
        model = var_select2seq_test_new(config, vocab, use_cuda)
    elif args.model == 'user2seq_expand':
        model = user2seq_expand(config, vocab, use_cuda)
    elif args.model == 'var_select_expand_user2seq':
        model = var_select_expand_user2seq(config, vocab, use_cuda)
    elif args.model == 'var_select2seq_align':
        model = var_select2seq_align(config, vocab, use_cuda)
    elif args.model == 'var_select2seq_test_span':
        model = var_select2seq_test_span(config, vocab, use_cuda)
    elif args.model == 'var_select2seq_test_span2':
        model = var_select2seq_test_span2(config, vocab, use_cuda)
    elif args.model == 'var_select_user2seq_new2':
        model = var_select_user2seq_new2(config, vocab, use_cuda)
    elif args.model == 'var_select2seq_test_span3':
        model = var_select2seq_test_span3(config, vocab, use_cuda)
    elif args.model == 'select2seq_label':
        model = select2seq_label(config, vocab, use_cuda)
    elif args.model == 'var_select_user2seq_label':
        model = var_select_user2seq_label(config, vocab, use_cuda)
    elif args.model == 'select2seq_encode':
        model = select2seq_encode(config, vocab, use_cuda)
    elif args.model == 'select2seq_encode2':
        model = select2seq_encode2(config, vocab, use_cuda)
    elif args.model == 'user_autoenc_vae_bow2':
        model = user_autoenc_vae_bow2(config, vocab, use_cuda)
    elif args.model == 'var_select_user2seq_new3':
        model = var_select_user2seq_new3(config, vocab, use_cuda)
    elif args.model == 'user_autoenc_vae_bow3':
        model = user_autoenc_vae_bow3(config, vocab, use_cuda)
    elif args.model == 'user2seq_test_new2':
        model = user2seq_test_new2(config, vocab, use_cuda)
    elif args.model == 'user2seq_test_new22':
        model = user2seq_test_new22(config, vocab, use_cuda)
    elif args.model == 'var_select_user2seq_new22':
        model = var_select_user2seq_new22(config, vocab, use_cuda)

    # total number of parameters
    logging(repr(model) + "\n\n")
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    logging('total number of parameters: %d\n\n' % param_count)

    if args.load_pre is not None:
        load_pretrained_weight(model, os.path.join(config.log_dir, args.load_pre))

    # load best or last checkpoint
    if args.restore:
        print('loading checkpoint...')
        checkpoints = torch.load(os.path.join(log_path, args.restore))
        model.load_state_dict(checkpoints['model'])

        # load other information
        updates = checkpoints['updates']
        epoch = checkpoints['epcoh']
        best_score = checkpoints['best_eval_score'] if 'best_eval_score' in checkpoints else 10.0
        logging('restore from: %d epcoh, %d update\n\n' % (epoch, updates))
    else:
        # set other information
        updates = 0
        epoch = 0
        best_score = None

    # set cuda
    if use_cuda:
        model.cuda()
    if len(args.gpus) > 1:
        model = nn.DataParallel(model, device_ids=args.gpus, dim=1)

    # optimizer
    if args.restore:
        optim = checkpoints['optim']
    else:
        optim = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                      lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)

    optim.set_parameters(model.parameters())
    if config.schedule:
        scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)
    else:
        scheduler = None

    if not args.notrain:
        if not args.topic:
            best_score = train(model, vocab, train_data, valid_data, scheduler, optim, epoch, updates, best_score)
            logging("Best score: %.6f\n" % best_score)
        else:
            eval_topic(model, train_data, epoch)
    else:
        assert args.restore is not None
        test_data = DataLoader(config.test_file, config, vocab, args.adj, use_gnn, args.model, False, args.debug, args.train_num)
        if args.topic:
            # utils.write_embedding(model.get_user.use_emb.weight.detach().cpu().numpy(), log_path, epoch)
            eval_bleu_with_topic(model, vocab, test_data, epoch, updates)
        elif args.topic_vec:
            eval_topic_vec(model, vocab, test_data, epoch, updates)
        else:
            eval_bleu(model, vocab, test_data, epoch, updates)


if __name__ == '__main__':
    main()
