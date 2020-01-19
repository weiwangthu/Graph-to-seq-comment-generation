from __future__ import division
from __future__ import print_function

import argparse
import codecs

import util
from Data import Vocab, DataLoader


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
                                 'var_select_var_user_diverse2seq_test4'])
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
    group.add_argument('-topic', default=False, action="store_true",
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


def main():
    vocab = Vocab(config.vocab_file, config.vocab_size)
    train_data = DataLoader(config.train_file, config.batch_size, vocab, args.adj, False, args.model, True, args.debug)
    valid_data = DataLoader(config.valid_file, config.batch_size, vocab, args.adj, False, args.model, True, args.debug)
    test_data = DataLoader(config.test_file, config.batch_size, vocab, args.adj, False, args.model, False, args.debug)

    lines = []
    for batch in train_data:
        for e in batch.examples:
            src_words = e.ori_title + e.ori_original_content
            tgt_words = e.ori_target
            # tgt_words = e.ori_targets[0]
            lines.append(' '.join(src_words) + '\t' + ' '.join(tgt_words))
    train_file = 'train.txt'
    with codecs.open(train_file, 'w', 'utf-8') as f:
        f.write("\n".join(lines))
        f.write("\n")

    lines = []
    for batch in valid_data:
        for e in batch.examples:
            src_words = e.ori_title + e.ori_original_content
            tgt_words = e.ori_target
            # tgt_words = e.ori_targets[0]
            lines.append(' '.join(src_words) + '\t' + ' '.join(tgt_words))
    train_file = 'dev.txt'
    with codecs.open(train_file, 'w', 'utf-8') as f:
        f.write("\n".join(lines))
        f.write("\n")

    # lines = []
    # for batch in test_data:
    #     for e in batch.examples:
    #         src_words = e.ori_title + e.ori_original_content
    #         # tgt_words = e.ori_target
    #         tgt_words = e.ori_targets[0]
    #         lines.append(' '.join(src_words) + '\t' + ' '.join(tgt_words))
    # train_file = 'test.txt'
    # with codecs.open(train_file, 'w', 'utf-8') as f:
    #     f.write("\n".join(lines))
    #     f.write("\n")


if __name__ == '__main__':
    main()
