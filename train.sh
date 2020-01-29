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


# generate
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 0 -model seq2seq -log 5c_seq2seq_t -notrain -restore checkpoint_best_11_4.637043.pt -beam_search -n_best 5
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 0 -model seq2seq -use_content -log 5c_seq2seq_tc -notrain -restore checkpoint_best_12_4.602771.pt -beam_search -n_best 5
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 0 -model bow2seq -log 5c_bow2seq_b -notrain -restore checkpoint_best_14_4.549125.pt -beam_search -n_best 5
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 3 -model h_attention -log 5c_h_attention -notrain -restore checkpoint_best_15_4.590873.pt -beam_search -n_best 5

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 0 -model seq2seq -log seq2seq_t -notrain -restore checkpoint_best_25_4.496477.pt -beam_search
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -model seq2seq -use_content -log seq2seq_tc -notrain -restore checkpoint_best_25_4.467218.pt -beam_search
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 0 -model bow2seq -log bow2seq_b -notrain -restore checkpoint_best_21_4.419056.pt -beam_search


CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model select2seq -log 5c_select2seq_tau0.5_gama0.5_e2

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model select_var_diverse2seq -log 5c_select_var_diverse2seq_tau0.5_gama0_kld0.05 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0 -gama_rank 0 -gama_reg 0

CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model var_select_var_diverse2seq -log 5c_var_select_var_diverse2seq_tau0.5_gama0_kld0.05_sel0.05 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.05 -gama_rank 0 -gama_reg 0

CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model var_select_var_user_diverse2seq -log 5c_var_select_var_user_diverse2seq_tau0.5_gama0_kld0.05_sel0_r1_re0.5 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.0 -gama_rank 1.0 -gama_reg 0.5

# test model
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model var_select_var_user_diverse2seq_test -log 5c_var_select_var_user_diverse2seq_test_tau0.5_gama0_kld0.05_sel0_r1_re0.01_test \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.0 -gama_rank 1.0 -gama_reg 0.01 -topic
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model var_select_var_user_diverse2seq_test -log 5c_var_select_var_user_diverse2seq_test2_tau0.5_gama0_kld0.05_sel0_r1_re0.01 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.0 -gama_rank 1.0 -gama_reg 0.01
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model var_select_var_user_diverse2seq_test -log 5c_var_select_var_user_diverse2seq_test2_tau0.5_gama0_kld0.05_sel0_r0.1_re0.01 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.0 -gama_rank 0.1 -gama_reg 0.01 -topic
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model var_select_var_user_diverse2seq_test -log 5c_var_select_var_user_diverse2seq_test2_tau0.5_gama0.005_kld0.05_sel0.05_r1_re0.01 \
  -tau 0.5 -gama1 0.005 -gama_kld 0.05 -gama_select 0.05 -gama_rank 1.0 -gama_reg 0.01 -topic -debug

# test2 model
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model var_select_var_user_diverse2seq_test2 -log 5c_var_select_var_user_diverse2seq_test22_tau0.5_gama0_kld0.05_sel0_r1_re0.01 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.0 -gama_rank 1.0 -gama_reg 0.01 -n_z 256 -topic
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model var_select_var_user_diverse2seq_test2 -log 5c_var_select_var_user_diverse2seq_test22_tau0.5_gama0.005_kld0.05_sel0.05_r1_re0.01 \
  -tau 0.5 -gama1 0.005 -gama_kld 0.05 -gama_select 0.05 -gama_rank 1.0 -gama_reg 0.01 -n_z 256

# test3 model
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model var_select_var_user_diverse2seq_test3 -log 5c_var_select_var_user_diverse2seq_test3_tau0.5_gama0_kld0.05_sel0_r1_re0.01 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.0 -gama_rank 1.0 -gama_reg 0.01 -n_z 256 -topic
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_11.pt -n_best 5 \
  -model var_select_var_user_diverse2seq_test3 -log 5c_var_select_var_user_diverse2seq_test3_tau0.5_gama0.005_kld0.05_sel0.05_r1_re0.01 \
  -tau 0.5 -gama1 0.005 -gama_kld 0.05 -gama_select 0.05 -gama_rank 1.0 -gama_reg 0.01 -n_z 256 -topic

# test4 model
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model var_select_var_user_diverse2seq_test4 -log 5c_var_select_var_user_diverse2seq_test4_tau0.5_gama0_kld0.05_sel0_r1_re0.01 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.0 -gama_rank 1.0 -gama_reg 0.01 -n_z 256 -topic
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model var_select_var_user_diverse2seq_test4 -log 5c_var_select_var_user_diverse2seq_test4_tau0.5_gama0.005_kld0.05_sel0.05_r1_re0.01 \
  -tau 0.5 -gama1 0.005 -gama_kld 0.05 -gama_select 0.05 -gama_rank 1.0 -gama_reg 0.01 -n_z 256



# restore
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 0 -model seq2seq -log seq2seq_t -restore checkpoint_11.pt
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -model seq2seq -use_content -log seq2seq_tc -restore checkpoint_5.pt
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 2 -model bow2seq -log bow2seq_b -restore checkpoint_9.pt
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 3 -model h_attention -log h_attention
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 3 -model h_attention -log 5c_h_attention -restore checkpoint_1.pt
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 0 -model seq2seq -log 5c_seq2seq_t -restore checkpoint_19.pt
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -model seq2seq -use_content -log 5c_seq2seq_tc -restore checkpoint_10.pt
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 2 -model bow2seq -log 5c_bow2seq_b -restore checkpoint_16.pt


# debug own model
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -model select2seq -log 5c_select2seq
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -model select2seq -log 5c_select2seq_tau1.0
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -model select2seq -log 5c_select2seq_tau0.5_gama0.5_e4
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content -model select2seq -log 5c_select2seq_tau0.5_gama0.5_e2
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -model select2seq -log 5c_select2seq_tau0.5_gama0.5
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -model select2seq_test -log 5c_select2seq_test_tau0.5_gama0.5_e2
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -model select2seq_test -log 5c_select2seq_test_tau0.5_gama0


CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -model select_var_diverse2seq -log 5c_select_var_diverse2seq_tau0.5_gama0_kld0
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content -model select_var_diverse2seq -log 5c_select_var_diverse2seq_tau0.5_gama0_kld0.4
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -model select_var_diverse2seq -log 5c_select_var_diverse2seq_tau0.5_gama0_kld0.05
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -model select_var_diverse2seq -log 5c_select_var_diverse2seq_tau0.5_gama0_kld0.005
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -model select_var_diverse2seq -log 5c_select_var_diverse2seq_tau0.5_gama0_kld0.0005



CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content -model var_select_var_diverse2seq -log 5c_var_select_var_diverse2seq_tau0.5_gama0_kld0_sel0
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -model var_select_var_diverse2seq -log 5c_var_select_var_diverse2seq_tau0.5_gama0_kld0_sel0.4
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -model var_select_var_diverse2seq -log 5c_var_select_var_diverse2seq_tau0.5_gama0_kld0.05_sel0.05
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content -model var_select_var_diverse2seq -log 5c_var_select_var_diverse2seq_tau0.5_gama0_kld0.05_sel0.005
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -model var_select_var_diverse2seq -log 5c_var_select_var_diverse2seq_tau0.5_gama0_kld0.05_sel0

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq -log 5c_var_select_var_user_diverse2seq_tau0.5_gama0_kld0.05_sel0_r1_re1
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq -log 5c_var_select_var_user_diverse2seq_tau0.5_gama0_kld0.05_sel0_r1_re0.5
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq -log 5c_var_select_var_user_diverse2seq_tau0.5_gama0_kld0.05_sel0_r0.5_re1
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq -log 5c_var_select_var_user_diverse2seq_tau0.5_gama0_kld0.05_sel0_r0.05_re1

# test
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test -log 5c_var_select_var_user_diverse2seq_test_tau0.5_gama0_kld0.05_sel0_r1_re1
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test -log 5c_var_select_var_user_diverse2seq_test_tau0.5_gama0_kld0.05_sel0_r1_re0.1
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test -log 5c_var_select_var_user_diverse2seq_test_tau0.5_gama0_kld0.05_sel0_r1_re0.01 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.0 -gama_rank 1.0 -gama_reg 0.01 -restore checkpoint_last.pt
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test -log 5c_var_select_var_user_diverse2seq_test_tau0.5_gama0_kld0.05_sel0_r1_re0.01_test \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.0 -gama_rank 1.0 -gama_reg 0.01

# test model
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test2 -log 5c_var_select_var_user_diverse2seq_test2_tau0.5_gama0_kld0.05_sel0_r1_re0.01 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.0 -gama_rank 1.0 -gama_reg 0.01 -restore checkpoint_last.pt
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test2 -log 5c_var_select_var_user_diverse2seq_test2_tau0.5_gama0_kld0.05_sel0_r0.1_re0.01 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.0 -gama_rank 0.1 -gama_reg 0.01 -restore checkpoint_last.pt
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test -log 5c_var_select_var_user_diverse2seq_test2_tau0.5_gama0_kld0.05_sel0_r1_re0.01_test \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.0 -gama_rank 1.0 -gama_reg 0.01
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test -log 5c_var_select_var_user_diverse2seq_test2_tau0.5_gama0.005_kld0.05_sel0.05_r1_re0.01 \
  -tau 0.5 -gama1 0.005 -gama_kld 0.05 -gama_select 0.05 -gama_rank 1.0 -gama_reg 0.01
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test -log 5c_var_select_var_user_diverse2seq_test2_tau0.5_gama0.005_kld0.05_sel0.1_r1_re0.01 \
  -tau 0.5 -gama1 0.005 -gama_kld 0.05 -gama_select 0.1 -gama_rank 1.0 -gama_reg 0.01
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -train_num 100000 \
  -model var_select_var_user_diverse2seq_test -log m_5c_var_select_var_user_diverse2seq_test2_tau0.5_gama0_kld0.05_sel0.1_r1_re0.01 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.1 -gama_rank 1.0 -gama_reg 0.01 -min_select 0.0
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -train_num 100000 \
  -model var_select_var_user_diverse2seq_test -log m_5c_var_select_var_user_diverse2seq_test2_tau0.5_gama0_kld0.05_sel0.1_r1_re0.01_m0.1 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.1 -gama_rank 1.0 -gama_reg 0.01 -min_select 0.1
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content -train_num 100000 \
  -model var_select_var_user_diverse2seq_test -log m_5c_var_select_var_user_diverse2seq_test2_tau0.5_gama0_kld0.1_sel0.1_r1_re0.01_m0.1 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.1 -gama_select 0.1 -gama_rank 1.0 -gama_reg 0.01 -min_select 0.1

# test2 model
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test2 -log 5c_var_select_var_user_diverse2seq_test22_tau0.5_gama0_kld0.05_sel0_r1_re0.01 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.0 -gama_rank 1.0 -gama_reg 0.01 -n_z 256
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test2 -log 5c_var_select_var_user_diverse2seq_test22_tau0.5_gama0.005_kld0.05_sel0.05_r1_re0.01 \
  -tau 0.5 -gama1 0.005 -gama_kld 0.05 -gama_select 0.05 -gama_rank 1.0 -gama_reg 0.01 -n_z 256
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test2 -log 5c_var_select_var_user_diverse2seq_test22_tau0.5_gama0.005_kld0.05_sel0.1_r1_re0.01 \
  -tau 0.5 -gama1 0.005 -gama_kld 0.05 -gama_select 0.1 -gama_rank 1.0 -gama_reg 0.01 -n_z 256
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content -train_num 100000 \
  -model var_select_var_user_diverse2seq_test2 -log m_5c_var_select_var_user_diverse2seq_test22_tau0.5_gama0_kld0.05_sel0.1_r1_re0.01_m0.1 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.1 -gama_rank 1.0 -gama_reg 0.01 -n_z 256 -min_select 0.1

# test3 model
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test3 -log 5c_var_select_var_user_diverse2seq_test3_tau0.5_gama0_kld0.05_sel0_r1_re0.01 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.0 -gama_rank 1.0 -gama_reg 0.01 -n_z 256
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test3 -log 5c_var_select_var_user_diverse2seq_test3_tau0.5_gama0.005_kld0.05_sel0.05_r1_re0.01\
  -tau 0.5 -gama1 0.005 -gama_kld 0.05 -gama_select 0.05 -gama_rank 1.0 -gama_reg 0.01 -n_z 256
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test3 -log 5c_var_select_var_user_diverse2seq_test3_tau0.5_gama0.005_kld0.1_sel0.05_r1_re0.01\
  -tau 0.5 -gama1 0.005 -gama_kld 0.1 -gama_select 0.05 -gama_rank 1.0 -gama_reg 0.01 -n_z 256
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -train_num 100000 \
  -model var_select_var_user_diverse2seq_test3 -log m_5c_var_select_var_user_diverse2seq_test3_tau0.5_gama0_kld0.1_sel0.1_r1_re0.01_m0.1 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.1 -gama_select 0.1 -gama_rank 1.0 -gama_reg 0.01 -n_z 256 -min_select 0.1
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -train_num 100000 \
  -model var_select_var_user_diverse2seq_test3 -log m_5c_var_select_var_user_diverse2seq_test3_tau0.5_gama0_kld0.1_sel0.1_r1_re0.01_m0.0 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.1 -gama_select 0.1 -gama_rank 1.0 -gama_reg 0.01 -n_z 256 -min_select 0.0

# test4 model
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test4 -log 5c_var_select_var_user_diverse2seq_test4_tau0.5_gama0_kld0.05_sel0_r1_re0.01 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.0 -gama_rank 1.0 -gama_reg 0.01 -n_z 256
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test4 -log 5c_var_select_var_user_diverse2seq_test4_tau0.5_gama0.005_kld0.05_sel0.05_r1_re0.01 \
  -tau 0.5 -gama1 0.005 -gama_kld 0.05 -gama_select 0.05 -gama_rank 1.0 -gama_reg 0.01 -n_z 256



# debug

CUDA_VISIBLE_DEVICES=0 python -m pudb.run  train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test -log 5c_var_select_var_user_diverse2seq_test2_tau0.5_gama0_kld0.05_sel0_r0.1_re0.01 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.0 -gama_rank 0.1 -gama_reg 0.01 -restore checkpoint_last.pt