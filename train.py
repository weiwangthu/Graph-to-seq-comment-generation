from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import time
import collections

import torch
import torch.nn as nn
from tqdm import tqdm
from torch import autograd

import lr_scheduler as L
import util
from optims import Optim
from util import utils
from Data import Vocab, DataLoader
from models import *

from util.nlp_utils import *
from util.misc_utils import AverageMeter


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
                                 'var_select_var_user_diverse2seq_test2', 'var_select_var_user_diverse2seq_test3'])
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

    parser.add_argument('-notrain', default=False, action='store_true',
                        help="train or not")
    parser.add_argument('-restore', type=str, default=None,
                        help="restore checkpoint")
    parser.add_argument('-beam_search', default=False, action='store_true',
                        help="beam_search")

    parser.add_argument('-log', default='', type=str,
                        help="log directory")
    parser.add_argument('-verbose', default=False, action='store_true',
                        help="verbose")
    parser.add_argument('-debug', default=False, action="store_true",
                        help='whether to use debug mode')

    group = parser.add_argument_group('Hyperparameter')
    group.add_argument('-n_z', type=int, default=64, metavar='N',
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
    best_score = org_best_score if org_best_score is not None else 10.
    for epoch in range(org_epoch + 1, org_epoch + config.epoch + 1):
        total_acc = 0.
        total_loss = 0.
        local_updates = 0
        start_time = time.time()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        if config.schedule:
            scheduler.step(epoch-1)
            print("Decaying learning rate to %g" % scheduler.get_lr()[0])

        model.train()

        for batch in tqdm(train_data, disable=not args.verbose):
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
                    if k in ['loss', 'acc']:
                        continue  # these are already logged above
                    else:
                        extra_meters[k].update(v.item())
            else:
                loss, acc = model.compute_loss(outputs.transpose(0, 1), target.transpose(0, 1)[1:])
            if torch.isnan(loss):
                raise Exception('nan error')

            loss.backward()
            total_loss += loss.data.item()
            total_acc += acc.data.item()

            optim.step()
            updates += 1
            local_updates += 1

            if updates % config.print_interval == 0 or args.debug:
                logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.3f, train acc: %.3f\n"
                        % (time.time() - start_time, epoch, updates, total_loss / local_updates, total_acc / local_updates))

                # log other loss
                if len(extra_meters) > 0:
                    other_information = ','.join('{:s}={:.3f}'.format(key, extra_meters[key].avg) for key in extra_meters.keys())
                    logging(other_information + '\n')

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


def eval_bleu(model, vocab, valid_data, epoch, updates):
    model.eval()
    multi_ref, reference, candidate, source, tags, alignments = [], [], [], [], [], []

    for batch in tqdm(valid_data, disable=not args.verbose):
        if len(args.gpus) > 1 or not args.beam_search:
            samples, alignment = model.sample(batch, use_cuda)
        else:
            samples, alignment = model.beam_sample(batch, use_cuda, beam_size=config.beam_size)
        '''
        if i == 0:
            print(batch.examples[27].ori_title)
            print(alignment.shape)
            print([d for d in alignment.tolist()[27]])
            return
        '''
        candidate += [vocab.id2sent(s) for s in samples]
        source += [example for example in batch.examples]
        # reference += [example.ori_target for example in batch.examples]
        multi_ref += [example.ori_targets for example in batch.examples]
    utils.write_result_to_file(source, candidate, log_path, epoch)
    # text_result, bleu = utils.eval_bleu(reference, candidate, log_path)
    text_result, bleu = utils.eval_multi_bleu(multi_ref, candidate, log_path)
    logging_csv([epoch, updates, text_result])
    print(text_result, flush=True)
    # print(multi_text_result, flush=True)
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
                if k in ['loss', 'acc']:
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
    train_data = DataLoader(config.train_file, config.batch_size, vocab, args.adj, use_gnn, args.model, True, args.debug)
    valid_data = DataLoader(config.valid_file, config.batch_size, vocab, args.adj, use_gnn, args.model, True, args.debug)

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

    # total number of parameters
    logging(repr(model) + "\n\n")
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    logging('total number of parameters: %d\n\n' % param_count)

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
        best_score = train(model, vocab, train_data, valid_data, scheduler, optim, epoch, updates, best_score)
        logging("Best score: %.6f\n" % best_score)
    else:
        assert args.restore is not None
        test_data = DataLoader(config.test_file, config.max_generator_batches, vocab, args.adj, use_gnn, args.model, False, args.debug)
        eval_bleu(model, vocab, test_data, epoch, updates)


if __name__ == '__main__':
    main()
