#!/usr/bin/env bash
# generate
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 0 -model seq2seq -log 5c_seq2seq_t -notrain -restore checkpoint_best_11_4.637043.pt -beam_search -n_best 5
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 0 -model seq2seq -use_content -log 5c_seq2seq_tc -notrain -restore checkpoint_best_12_4.602771.pt -beam_search -n_best 5
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 0 -model bow2seq -log 5c_bow2seq_b -notrain -restore checkpoint_best_14_4.549125.pt -beam_search -n_best 5
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 3 -model h_attention -log 5c_h_attention -notrain -restore checkpoint_best_15_4.590873.pt -beam_search -n_best 5

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 0 -model seq2seq -log seq2seq_t -notrain -restore checkpoint_best_25_4.496477.pt -beam_search
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -model seq2seq -use_content -log seq2seq_tc -notrain -restore checkpoint_best_25_4.467218.pt -beam_search
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 0 -model bow2seq -log bow2seq_b -notrain -restore checkpoint_best_21_4.419056.pt -beam_search


#yahoo
# seq2seq-T
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 0 -config config_yahoo.yaml \
  -notrain -beam_search -restore checkpoint_best.pt \
  -model seq2seq -log seq2seq_t
# seq2seq-TC
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -config config_yahoo.yaml \
  -notrain -beam_search -restore checkpoint_best.pt \
  -model seq2seq -use_content -log seq2seq_tc
# bow2seq-b
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 2 -config config_yahoo.yaml \
  -notrain -beam_search -restore checkpoint_best.pt \
  -model bow2seq -log bow2seq_b
## bow2seq-k
#CUDA_VISIBLE_DEVICES=0 python train.py -gpus 0 -model bow2seq -log bow2seq_k
# h_attention
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 3 -config config_yahoo.yaml \
  -notrain -beam_search -restore checkpoint_best.pt \
  -model h_attention -log h_attention

#tencent
# seq2seq-T
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 0 -config config_tencent.yaml \
  -notrain -beam_search -restore checkpoint_best.pt \
  -model seq2seq -log seq2seq_t
# seq2seq-TC
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -config config_tencent.yaml \
  -notrain -beam_search -restore checkpoint_best.pt \
  -model seq2seq -use_content -log seq2seq_tc
# bow2seq-b
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 2 -config config_tencent.yaml \
  -notrain -beam_search -restore checkpoint_best.pt \
  -model bow2seq -log bow2seq_b
## bow2seq-k
#CUDA_VISIBLE_DEVICES=0 python train.py -gpus 0 -model bow2seq -log bow2seq_k
# h_attention
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 3 -config config_tencent.yaml \
  -notrain -beam_search -restore checkpoint_best.pt \
  -model h_attention -log h_attention
