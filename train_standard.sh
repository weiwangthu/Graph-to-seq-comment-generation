#!/usr/bin/env bash
# seq2seq-T
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 0 -model seq2seq -log seq2seq_t
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 0 -model seq2seq -log 5c_seq2seq_t
## seq2seq-C
#CUDA_VISIBLE_DEVICES=0 python train.py -gpus 0 -model seq2seq -log seq2seq_c
# seq2seq-TC
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -model seq2seq -use_content -log seq2seq_tc
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -model seq2seq -use_content -log 5c_seq2seq_tc
# bow2seq-b
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 2 -model bow2seq -log bow2seq_b
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 2 -model bow2seq -log 5c_bow2seq_b
## bow2seq-k
#CUDA_VISIBLE_DEVICES=0 python train.py -gpus 0 -model bow2seq -log bow2seq_k
# h_attention
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 3 -model h_attention -log h_attention
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 3 -model h_attention -log 5c_h_attention

# graph2seq
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 0 -model graph2seq -graph_model GCN -use_copy -log graph2seq
# without copy
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -model graph2seq -graph_model GCN  -log graph2seq_wo_copy
# without GCN
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 2 -model graph2seq -use_copy -log graph2seq_wo_gcn
# without copy and GCN
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 3 -model graph2seq -log graph2seq_wo_copy_gcn


# restore
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 0 -model seq2seq -log seq2seq_t -restore checkpoint_11.pt
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -model seq2seq -use_content -log seq2seq_tc -restore checkpoint_5.pt
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 2 -model bow2seq -log bow2seq_b -restore checkpoint_9.pt
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 3 -model h_attention -log h_attention
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 3 -model h_attention -log 5c_h_attention -restore checkpoint_1.pt
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 0 -model seq2seq -log 5c_seq2seq_t -restore checkpoint_19.pt
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -model seq2seq -use_content -log 5c_seq2seq_tc -restore checkpoint_10.pt
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 2 -model bow2seq -log 5c_bow2seq_b -restore checkpoint_16.pt


#yahoo
# seq2seq-T
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 0 -config config_yahoo.yaml \
  -model seq2seq -log seq2seq_t
# seq2seq-TC
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -config config_yahoo.yaml \
  -model seq2seq -use_content -log seq2seq_tc -restore checkpoint_last.pt
# bow2seq-b
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 2 -config config_yahoo.yaml \
  -model bow2seq -log bow2seq_b
## bow2seq-k
#CUDA_VISIBLE_DEVICES=0 python train.py -gpus 0 -model bow2seq -log bow2seq_k
# h_attention
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 3 -config config_yahoo.yaml \
  -model h_attention -log h_attention
# gann
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 0 -config config_yahoo.yaml \
  -model seq2gateseq -log seq2gateseq_t
# cvae
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 0 -config config_yahoo.yaml \
  -model cvae -log cvae \
  -dynamic2 -mid_max 10 -gama_kld 1

# 163
# seq2seq-T
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 0 -config config_163.yaml \
  -model seq2seq -log seq2seq_t
# seq2seq-TC
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -config config_163.yaml \
  -model seq2seq -use_content -log seq2seq_tc -restore checkpoint_last.pt
# bow2seq-b
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 2 -config config_163.yaml \
  -model bow2seq -log bow2seq_b
## bow2seq-k
#CUDA_VISIBLE_DEVICES=0 python train.py -gpus 0 -model bow2seq -log bow2seq_k
# gann
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 0 -config config_163.yaml \
  -model seq2gateseq -log seq2gateseq_t
# cvae
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 0 -config config_163.yaml \
  -model cvae -log cvae \
  -dynamic2 -mid_max 10 -gama_kld 1

#tencent
# seq2seq-T
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 0 -config config_tencent.yaml \
  -model seq2seq -log seq2seq_t
# seq2seq-TC
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -config config_tencent.yaml \
  -model seq2seq -use_content -log seq2seq_tc
# bow2seq-b
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 2 -config config_tencent.yaml \
  -model bow2seq -log bow2seq_b
## bow2seq-k
#CUDA_VISIBLE_DEVICES=0 python train.py -gpus 0 -model bow2seq -log bow2seq_k
# h_attention
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 3 -config config_tencent.yaml \
  -model h_attention -log h_attention
# gann
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 0 -config config_tencent.yaml \
  -model seq2gateseq -log seq2gateseq_t
# cvae
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 0 -config config_tencent.yaml \
  -model cvae -log cvae \
  -dynamic2 -mid_max 10 -gama_kld 1