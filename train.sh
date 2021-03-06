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
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test -log 5c_var_select_var_user_diverse2seq_test2_tau0.5_gama0.005_kld0.05_sel0.05_r1_re0.01_fix \
  -tau 0.5 -gama1 0.005 -gama_kld 0.05 -gama_select 0.05 -gama_rank 1.0 -gama_reg 0.01

# var select, user
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model var_select2seq_test -log 5c_var_select2seq_test_tau0.5_gama0.005_sel0.1 \
  -tau 0.5 -gama1 0.005 -gama_select 0.1
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model var_select2seq_test -log 5c_var_select2seq_test_tau0.5_gama0_sel0.1 \
  -tau 0.5 -gama1 0 -gama_select 0.1
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model var_select2seq_test -log 5c_var_select2seq_test_tau0.5_gama0_sel0.01 \
  -tau 0.5 -gama1 0 -gama_select 0.01

CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model var_select_user2seq_test -log 5c_var_select_user2seq_test_tau0.5_gama0.005_sel0.1_re0.01 \
  -tau 0.5 -gama1 0.005 -gama_select 0.1 -gama_reg 0.01 -n_z 256
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model var_select_user2seq_test -log 5c_var_select_user2seq_test_tau0.5_gama0_sel0.1_re0.01 \
  -tau 0.5 -gama1 0 -gama_select 0.1 -gama_reg 0.01 -n_z 256
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model var_select_user2seq_test -log 5c_var_select_user2seq_test_tau0.5_gama0_sel0.1_re0 \
  -tau 0.5 -gama1 0 -gama_select 0.1 -gama_reg 0 -n_z 256
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model var_select_user2seq_test -log 5c_var_select_user2seq_test_tau0.5_gama0_sel0.1_re0_m0.1 \
  -tau 0.5 -gama1 0 -gama_select 0.1 -gama_reg 0 -n_z 256 -min_select 0.1
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model var_select_user2seq_test -log 5c_var_select_user2seq_test_tau0.5_gama0_sel0.5_re0_m0.1 \
  -tau 0.5 -gama1 0 -gama_select 0.5 -gama_reg 0 -n_z 256 -min_select 0.1

CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model var_select_user2seq_new -log 5c_var_select_user2seq_new_tau0.5_re0_kld0.2_t100_dy3_0.2_gama0_klds0.5_m0.1  \
  -tau 0.5 -gama_reg 0 -gama_kld 0.2 -n_z 64 -n_topic_num 100 -dynamic3 -gama_select 0.2 -mid_max 20 \
  -gama1 0 -gama_kld_select 0.5 -min_select 0.1
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model var_select_user2seq_new -log 5c_var_select_user2seq_new_tau0.5_re0_kld0.2_t100_dy3_0.2_gama0_klds0.5_m0.1_cone  \
  -tau 0.5 -gama_reg 0 -gama_kld 0.2 -n_z 64 -n_topic_num 100 -dynamic3 -gama_select 0.2 -mid_max 20 \
  -gama1 0 -gama_kld_select 0.5 -min_select 0.1 -con_one_user

CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model user2seq_test -log 5c_user2seq_test_re0 \
  -gama_reg 0 -n_z 256
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user2seq_test -log 5c_user2seq_test_re0.01 \
  -gama_reg 0.01 -n_z 256
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user2seq_test -log 5c_user2seq_test_re0.001 \
  -gama_reg 0.001 -n_z 256
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model user2seq_test -log 5c_user2seq_test_tau0.5_re0.01_one \
  -tau 0.5 -gama_reg 0.01 -n_z 256 -one_user
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model user2seq_test -log 5c_user2seq_test_re1 \
  -gama_reg 1 -n_z 256
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user2seq_test -log 5c_user2seq_test_tau0.5_re1_one \
  -tau 0.5 -gama_reg 1 -n_z 256 -one_user
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model user2seq_test -log 5c_user2seq_test_tau0.5_re0_one \
  -tau 0.5 -gama_reg 0 -n_z 256 -one_user
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user2seq_test -log 5c_user2seq_test_tau0.5_re0_r1_one \
  -tau 0.5 -gama_reg 0 -gama_rank 1 -n_z 256 -one_user
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user2seq_test -log 5c_user2seq_test_tau0.5_re0_r1_one_loss2 \
  -tau 0.5 -gama_reg 0 -gama_rank 1 -n_z 256 -one_user

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user2seq_test_new -log 5c_user2seq_test_new_tau0.5_re0_kld0.2_t100_dy3_0.2  \
  -tau 0.5 -gama_reg 0 -gama_kld 0.2 -n_z 64 -n_topic_num 100 -dynamic3 -gama_select 0.2 -mid_max 20
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user2seq_test_new -log 5c_user2seq_test_new_tau0.5_re0_kld0.2_t100_dy3_0.2_cone  \
  -tau 0.5 -gama_reg 0 -gama_kld 0.2 -n_z 64 -n_topic_num 100 -dynamic3 -gama_select 0.2 -mid_max 20 -con_one_user
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user2seq_test_new -log 5c_user2seq_test_new_tau0.5_re0_kld0.2_t100_dy3_0.2_10  \
  -tau 0.5 -gama_reg 0 -gama_kld 0.2 -n_z 64 -n_topic_num 100 -dynamic3 -gama_select 0.2 -mid_max 10

CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model autoenc -log 5c_autoenc
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model autoenc_lm -log 5c_autoenc_lm

CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model autoenc_vae -log 5c_autoenc_vae_kld0.1 \
  -gama_kld 0.1 -n_z 256
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model autoenc_vae -log 5c_autoenc_vae_kld0.01  \
  -gama_kld 0.01 -n_z 256
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model autoenc_vae -log 5c_autoenc_vae_dy1  \
  -dynamic1 -n_z 256
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model autoenc_vae -log 5c_autoenc_vae_dy2  \
  -dynamic2 -n_z 256
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model autoenc_vae -log 5c_autoenc_vae_kld0.1_drop \
  -gama_kld 0.1 -n_z 256 -drop_dec_input
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model autoenc_vae -log 5c_autoenc_vae_dy2_0.05  \
  -dynamic2 -gama_kld 0.05 -n_z 256
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model autoenc_vae -log 5c_autoenc_vae_dy1_0.1_10  \
  -dynamic1 -gama_kld 0.1 -n_z 256 -mid_max 10.0
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model autoenc_vae -log 5c_autoenc_vae_dy2_0.1_20  \
  -dynamic2 -gama_kld 0.1 -n_z 256 -mid_max 20.0
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model autoenc_vae -log 5c_autoenc_vae_kld0.05  \
  -gama_kld 0.05 -n_z 256
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model autoenc_vae -log 5c_autoenc_vae_kld0.05_drop \
  -gama_kld 0.05 -n_z 256 -drop_dec_input
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model autoenc_vae -log 5c_autoenc_vae_dy1_0.05_10  \
  -dynamic1 -gama_kld 0.05 -n_z 256 -mid_max 10.0
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model autoenc_vae -log 5c_autoenc_vae_dy2_0.05_20  \
  -dynamic2 -gama_kld 0.05 -n_z 256 -mid_max 20.0
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model autoenc_vae -log 5c_autoenc_vae_dy2_0.05_20_drop  \
  -dynamic2 -gama_kld 0.05 -n_z 256 -mid_max 20.0 -drop_dec_input

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model autoenc_vae_cat -log 5c_autoenc_vae_cat_kld0.1  \
  -gama_kld 0.1 -n_z 256
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model autoenc_vae_cat -log 5c_autoenc_vae_cat_kld0.1_drop  \
  -gama_kld 0.1 -n_z 256 -drop_dec_input

CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model autoenc_vae_bow -log 5c_autoenc_vae_bow_kld0.01  \
  -gama_kld 0.01 -n_z 64
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model autoenc_vae_bow -log 5c_autoenc_vae_bow_dy1  \
  -dynamic1 -n_z 64
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model autoenc_vae_bow -log 5c_autoenc_vae_bow_dy2  \
  -dynamic2 -n_z 64
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model autoenc_vae_bow -log 5c_autoenc_vae_bow_kld0  \
  -gama_kld 0 -n_z 64
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model autoenc_vae_bow -log 5c_autoenc_vae_bow_kld1  \
  -gama_kld 1 -n_z 64
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model autoenc_vae_bow -log 5c_autoenc_vae_bow_dy1_0.1  \
  -dynamic1 -gama_kld 0.1 -n_z 64
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model autoenc_vae_bow -log 5c_autoenc_vae_bow_kld0.1  \
  -gama_kld 0.1 -n_z 64
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model autoenc_vae_bow -log 5c_autoenc_vae_bow_dy2_0.1   \
  -dynamic2 -gama_kld 0.1 -n_z 64
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model autoenc_vae_bow -log 5c_autoenc_vae_bow_dy1_0.2_10  \
  -dynamic1 -gama_kld 0.2 -n_z 64 -mid_max 10.0
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model autoenc_vae_bow -log 5c_autoenc_vae_bow_dy2_0.2_20   \
  -dynamic2 -gama_kld 0.2 -n_z 64 -mid_max 20.0
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model autoenc_vae_bow -log 5c_autoenc_vae_bow_kld0.2  \
  -gama_kld 0.2 -n_z 64
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model autoenc_vae_bow -log 5c_autoenc_vae_bow_kld0.4  \
  -gama_kld 0.4 -n_z 64
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model autoenc_vae_bow_norm -log 5c_autoenc_vae_bow_norm_kld0.2  \
  -gama_kld 0.2 -n_z 64



CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model user_autoenc -log 5c_user_autoenc \
  -tau 0.5 -gama_reg 0 -gama_select 1 -n_z 256  -one_user
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc -log 5c_user_autoenc_tau0.5_re0_sel0_one \
  -tau 0.5 -gama_reg 0 -gama_select 0 -n_z 256  -one_user
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc -log 5c_user_autoenc_tau0.5_re0_sel1_one \
  -tau 0.5 -gama_reg 0 -gama_select 1 -n_z 256  -one_user
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc -log 5c_user_autoenc_tau0.5_re0_sel1 \
  -tau 0.5 -gama_reg 0 -gama_select 1 -n_z 256

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc -log 5c_user_autoenc_tau0.5_re0_sel1_t100_one \
  -tau 0.5 -gama_reg 0 -gama_select 1 -n_z 256  -n_topic_num 100 -one_user
CUDA_VISIBLE_DEVICES=6 python train.py -gpus 1 -use_content \
  -model user_autoenc -log 5c_user_autoenc_tau0.5_re0_sel1_t1000_one \
  -tau 0.5 -gama_reg 0 -gama_select 1 -n_z 256  -n_topic_num 1000 -one_user
CUDA_VISIBLE_DEVICES=5 python train.py -gpus 1 -use_content \
  -model user_autoenc -log 5c_user_autoenc_tau0.5_re0_sel1_t100_z64_one \
  -tau 0.5 -gama_reg 0 -gama_select 1 -n_z 64  -n_topic_num 100 -one_user
CUDA_VISIBLE_DEVICES=4 python train.py -gpus 1 -use_content \
  -model user_autoenc -log 5c_user_autoenc_tau0.5_re0_sel1_t100_z128_one \
  -tau 0.5 -gama_reg 0 -gama_select 1 -n_z 128  -n_topic_num 100 -one_user
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc -log 5c_user_autoenc_tau0.5_re0_sel1_t100 \
  -tau 0.5 -gama_reg 0 -gama_select 1 -n_z 256  -n_topic_num 100
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc -log 5c_user_autoenc_tau0.5_re0_sel1_t1000 \
  -tau 0.5 -gama_reg 0 -gama_select 1 -n_z 256  -n_topic_num 1000

CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model user_autoenc -log 5c_user_autoenc_tau0.5_re0_sel1_r1_t100_one \
  -tau 0.5 -gama_reg 0 -gama_select 1 -gama_rank 1 -n_z 256  -n_topic_num 100 -one_user
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc -log 5c_user_autoenc_tau0.5_re0_sel10_r1_t100_one \
  -tau 0.5 -gama_reg 0 -gama_select 10 -gama_rank 1 -n_z 256  -n_topic_num 100 -one_user
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc -log 5c_user_autoenc_tau0.5_re0_sel100_r1_t100_one \
  -tau 0.5 -gama_reg 0 -gama_select 100 -gama_rank 1 -n_z 256  -n_topic_num 100 -one_user
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc -log 5c_user_autoenc_tau0.5_re0_sel10_r0.1_t100_one \
  -tau 0.5 -gama_reg 0 -gama_select 10 -gama_rank 0.1 -n_z 256  -n_topic_num 100 -one_user
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc -log 5c_user_autoenc_tau0.5_re0_sel10_r0.01_t100_one \
  -tau 0.5 -gama_reg 0 -gama_select 10 -gama_rank 0.01 -n_z 256  -n_topic_num 100 -one_user

CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_near -log 5c_user_autoenc_near_re0_r1_t100 \
  -gama_reg 0 -gama_rank 1 -n_z 256  -n_topic_num 100
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_near -log 5c_user_autoenc_near_re0_r0.1_t100 \
  -gama_reg 0 -gama_rank 0.1 -n_z 256  -n_topic_num 100
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model user_autoenc_near -log 5c_user_autoenc_near_re0_r0.01_t100 \
  -gama_reg 0 -gama_rank 0.01 -n_z 256  -n_topic_num 100
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_near -log 5c_user_autoenc_near_re0_r0.001_t100 \
  -gama_reg 0 -gama_rank 0.001 -n_z 256  -n_topic_num 100
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model user_autoenc_near -log 5c_user_autoenc_near_re0_r0_t100 \
  -gama_reg 0 -gama_rank 0 -n_z 256  -n_topic_num 100

CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld1_sel0_one \
  -tau 0.5 -gama_reg 0 -gama_kld 1 -gama_select 0 -n_z 256  -one_user
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld0.1_sel0_one \
  -tau 0.5 -gama_reg 0 -gama_kld 0.1 -gama_select 0 -n_z 256  -one_user
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld1_sel0.1_one \
  -tau 0.5 -gama_reg 0 -gama_kld 1 -gama_select 0.1 -n_z 256  -one_user
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld1_sel0.5_one \
  -tau 0.5 -gama_reg 0 -gama_kld 1 -gama_select 0.5 -n_z 256  -one_user
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld1_sel1_one \
  -tau 0.5 -gama_reg 0 -gama_kld 1 -gama_select 1 -n_z 256  -one_user
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld1_sel1 \
  -tau 0.5 -gama_reg 0 -gama_kld 1 -gama_select 1 -n_z 256
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld1_sel1_fixl \
  -tau 0.5 -gama_reg 0 -gama_kld 1 -gama_select 1 -n_z 256
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld1_sel10_fixl \
  -tau 0.5 -gama_reg 0 -gama_kld 1 -gama_select 10 -n_z 256
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld1_sel100_fixl \
  -tau 0.5 -gama_reg 0 -gama_kld 1 -gama_select 100 -n_z 256
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld0.1_sel1_fixl \
  -tau 0.5 -gama_reg 0 -gama_kld 0.1 -gama_select 1 -n_z 256

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld0_sel0_fixl_fixu \
  -tau 0.5 -gama_reg 0 -gama_kld 0 -gama_select 0 -n_z 256
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld0_sel1_fixl_fixu \
  -tau 0.5 -gama_reg 0 -gama_kld 0 -gama_select 1 -n_z 256
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld1_sel0_fixl_fixu \
  -tau 0.5 -gama_reg 0 -gama_kld 1 -gama_select 0 -n_z 256
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld1_sel1_fixl_fixu \
  -tau 0.5 -gama_reg 0 -gama_kld 1 -gama_select 1 -n_z 256
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld0.001_sel1_fixl_fixu \
  -tau 0.5 -gama_reg 0 -gama_kld 0.001 -gama_select 1 -n_z 256
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld0.0001_sel1_fixl_fixu \
  -tau 0.5 -gama_reg 0 -gama_kld 0.0001 -gama_select 1 -n_z 256
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld0.01_sel1_fixl_fixu \
  -tau 0.5 -gama_reg 0 -gama_kld 0.01 -gama_select 1 -n_z 256
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld0.1_sel1_fixl_fixu \
  -tau 0.5 -gama_reg 0 -gama_kld 0.1 -gama_select 1 -n_z 256
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld0.05_sel1_fixl_fixu \
  -tau 0.5 -gama_reg 0 -gama_kld 0.05 -gama_select 1 -n_z 256
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld0.02_sel1_fixl_fixu \
  -tau 0.5 -gama_reg 0 -gama_kld 0.02 -gama_select 1 -n_z 256
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld0.01_sel10_fixl_fixu \
  -tau 0.5 -gama_reg 0 -gama_kld 0.01 -gama_select 10 -n_z 256
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_kld0.01_sel100_fixl_fixu \
  -tau 0.5 -gama_reg 0 -gama_kld 0.01 -gama_select 100 -n_z 256

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_dy1_sel100_fixl_fixu \
  -tau 0.5 -gama_reg 0 -dynamic1 -gama_select 100 -n_z 256
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_dy2_sel100_fixl_fixu \
  -tau 0.5 -gama_reg 0 -dynamic2 -gama_select 100 -n_z 256

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_sel10_fixl_fixu_dy1_0.05_10 \
  -tau 0.5 -gama_reg 0 -gama_select 10 -n_z 256 -dynamic1 -gama_kld 0.05 -mid_max 10.0
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_sel10_fixl_fixu_dy2_0.05_20 \
  -tau 0.5 -gama_reg 0 -gama_select 10 -n_z 256 -dynamic2 -gama_kld 0.05 -mid_max 20.0
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_sel10_fixl_fixu_dy1_0.1_10 \
  -tau 0.5 -gama_reg 0 -gama_select 10 -n_z 256 -dynamic1 -gama_kld 0.1 -mid_max 10.0
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_sel10_fixl_fixu_dy2_0.1_20 \
  -tau 0.5 -gama_reg 0 -gama_select 10 -n_z 256 -dynamic2 -gama_kld 0.1 -mid_max 20.0
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_sel10_fixl_fixu_dy1_0.01_10 \
  -tau 0.5 -gama_reg 0 -gama_select 10 -n_z 256 -dynamic1 -gama_kld 0.01 -mid_max 10.0
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_sel10_fixl_fixu_dy2_0.01_20 \
  -tau 0.5 -gama_reg 0 -gama_select 10 -n_z 256 -dynamic2 -gama_kld 0.01 -mid_max 20.0
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_sel10_fixl_fixu_dy1_0.2_10 \
  -tau 0.5 -gama_reg 0 -gama_select 10 -n_z 256 -dynamic1 -gama_kld 0.2 -mid_max 10.0
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_sel10_fixl_fixu_dy2_0.2_20 \
  -tau 0.5 -gama_reg 0 -gama_select 10 -n_z 256 -dynamic2 -gama_kld 0.2 -mid_max 20.0

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_sel1_fixl_fixu_dy1_0.05_10_t100 \
  -tau 0.5 -gama_reg 0 -gama_select 1 -n_z 256 -dynamic1 -gama_kld 0.05 -mid_max 10.0 -n_topic_num 100
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0_sel1_fixl_fixu_dy1_0.05_10_t1000 \
  -tau 0.5 -gama_reg 0 -gama_select 1 -n_z 256 -dynamic1 -gama_kld 0.05 -mid_max 10.0 -n_topic_num 1000
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0.1_sel1_fixl_fixu_dy1_0.05_10 \
  -tau 0.5 -gama_reg 0.1 -gama_select 1 -n_z 256 -dynamic1 -gama_kld 0.05 -mid_max 10.0
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0.01_sel1_fixl_fixu_dy1_0.05_10 \
  -tau 0.5 -gama_reg 0.01 -gama_select 1 -n_z 256 -dynamic1 -gama_kld 0.05 -mid_max 10.0

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re1_sel1_fixl_fixu_dy1_0.05_10 \
  -tau 0.5 -gama_reg 1 -gama_select 1 -n_z 256 -dynamic1 -gama_kld 0.05 -mid_max 10.0
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re1_sel1_fixl_fixu_dy1_0.05_10_t100 \
  -tau 0.5 -gama_reg 1 -gama_select 1 -n_z 256 -dynamic1 -gama_kld 0.05 -mid_max 10.0 -n_topic_num 100
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re1_sel1_fixl_fixu_dy1_0.05_10_t1000 \
  -tau 0.5 -gama_reg 1 -gama_select 1 -n_z 256 -dynamic1 -gama_kld 0.05 -mid_max 10.0 -n_topic_num 1000


CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_sel10_dy1_0.2_10  \
  -tau 0.5 -gama_reg 0 -gama_select 10 -dynamic1 -gama_kld 0.2 -n_z 64 -mid_max 10.0
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_sel10_dy2_0.2_20   \
  -tau 0.5 -gama_reg 0 -gama_select 10 -dynamic2 -gama_kld 0.2 -n_z 64 -mid_max 20.0
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_sel10_dy1_0.1_10  \
  -tau 0.5 -gama_reg 0 -gama_select 10 -dynamic1 -gama_kld 0.1 -n_z 64 -mid_max 10.0
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_sel10_dy2_0.1_20   \
  -tau 0.5 -gama_reg 0 -gama_select 10 -dynamic2 -gama_kld 0.1 -n_z 64 -mid_max 20.0
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_sel10_kld0.2  \
  -tau 0.5 -gama_reg 0 -gama_select 10 -gama_kld 0.2 -n_z 64
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_sel10_kld0.1  \
  -tau 0.5 -gama_reg 0 -gama_select 10 -gama_kld 0.1 -n_z 64

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_sel0_kld0.2  \
  -tau 0.5 -gama_reg 0 -gama_select 0 -gama_kld 0.2 -n_z 64
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_sel1_kld0.2  \
  -tau 0.5 -gama_reg 0 -gama_select 1 -gama_kld 0.2 -n_z 64
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_sel0.1_kld0.2  \
  -tau 0.5 -gama_reg 0 -gama_select 0.1 -gama_kld 0.2 -n_z 64
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_sel0.01_kld0.2  \
  -tau 0.5 -gama_reg 0 -gama_select 0.01 -gama_kld 0.2 -n_z 64
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_sel0.5_kld0.2  \
  -tau 0.5 -gama_reg 0 -gama_select 0.5 -gama_kld 0.2 -n_z 64
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_sel0.2_kld0.2  \
  -tau 0.5 -gama_reg 0 -gama_select 0.2 -gama_kld 0.2 -n_z 64

CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_sel1_kld0.2_t100  \
  -tau 0.5 -gama_reg 0 -gama_select 1 -gama_kld 0.2 -n_z 64 -n_topic_num 100
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_sel1_kld0.2_t1000  \
  -tau 0.5 -gama_reg 0 -gama_select 1 -gama_kld 0.2 -n_z 64 -n_topic_num 1000
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0.1_sel1_kld0.2  \
  -tau 0.5 -gama_reg 0.1 -gama_select 1 -gama_kld 0.2 -n_z 64
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0.01_sel1_kld0.2  \
  -tau 0.5 -gama_reg 0.01 -gama_select 1 -gama_kld 0.2 -n_z 64

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0.1_sel100_kld0.2_t100  \
  -tau 0.5 -gama_reg 0.1 -gama_select 100 -gama_kld 0.2 -n_z 64 -n_topic_num 100
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0.1_sel100_kld0.2_t1000  \
  -tau 0.5 -gama_reg 0.1 -gama_select 100 -gama_kld 0.2 -n_z 64 -n_topic_num 1000
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0.1_sel10_kld0.2  \
  -tau 0.5 -gama_reg 0.1 -gama_select 10 -gama_kld 0.2 -n_z 64
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0.1_sel100_kld0.2  \
  -tau 0.5 -gama_reg 0.1 -gama_select 100 -gama_kld 0.2 -n_z 64

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_kld0.2_t100_dy3_0.2  \
  -tau 0.5 -gama_reg 0 -gama_kld 0.2 -n_z 64 -n_topic_num 100 -dynamic3 -gama_select 0.2 -mid_max 20
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow_norm -log 5c_user_autoenc_vae_bow_norm_re0_kld0.2_t100_dy3_0.2  \
  -tau 0.5 -gama_reg 0 -gama_kld 0.2 -n_z 64 -n_topic_num 100 -dynamic3 -gama_select 0.2 -mid_max 20
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_kld0.2_t100_sel0.4  \
  -tau 0.5 -gama_reg 0 -gama_kld 0.2 -n_z 64 -n_topic_num 100 -gama_select 0.4
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_kld0.2_t100_sel0.4_d128  \
  -tau 0.5 -gama_reg 0 -gama_kld 0.2 -n_z 128 -n_topic_num 100 -gama_select 0.4
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_kld0.2_t100_sel0.4_d256  \
  -tau 0.5 -gama_reg 0 -gama_kld 0.2 -n_z 256 -n_topic_num 100 -gama_select 0.4
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content \
  -model user_autoenc_vae_bow -log 5c_user_autoenc_vae_bow_re0_kld0.2_t1000_sel0.4  \
  -tau 0.5 -gama_reg 0 -gama_kld 0.2 -n_z 64 -n_topic_num 1000 -gama_select 0.4

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
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test2 -log 5c_var_select_var_user_diverse2seq_test22_tau0.5_gama0.005_kld0.05_sel0.05_r1_re0.01_fix \
  -tau 0.5 -gama1 0.005 -gama_kld 0.05 -gama_select 0.05 -gama_rank 1.0 -gama_reg 0.01 -n_z 256

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
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test3 -log 5c_var_select_var_user_diverse2seq_test3_tau0.5_gama0.005_kld0.05_sel0.05_r1_re0.01_fix \
  -tau 0.5 -gama1 0.005 -gama_kld 0.05 -gama_select 0.05 -gama_rank 1.0 -gama_reg 0.01 -n_z 256

# test4 model
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test4 -log 5c_var_select_var_user_diverse2seq_test4_tau0.5_gama0_kld0.05_sel0_r1_re0.01 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.0 -gama_rank 1.0 -gama_reg 0.01 -n_z 256
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test4 -log 5c_var_select_var_user_diverse2seq_test4_tau0.5_gama0.005_kld0.05_sel0.05_r1_re0.01 \
  -tau 0.5 -gama1 0.005 -gama_kld 0.05 -gama_select 0.05 -gama_rank 1.0 -gama_reg 0.01 -n_z 256
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test4 -log 5c_var_select_var_user_diverse2seq_test4_tau0.5_gama0.005_kld0.05_sel0.05_r1_re0.01_fix \
  -tau 0.5 -gama1 0.005 -gama_kld 0.05 -gama_select 0.05 -gama_rank 1.0 -gama_reg 0.01 -n_z 256



# debug

CUDA_VISIBLE_DEVICES=0 python -m pudb.run  train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test -log 5c_var_select_var_user_diverse2seq_test2_tau0.5_gama0_kld0.05_sel0_r0.1_re0.01 \
  -tau 0.5 -gama1 0.0 -gama_kld 0.05 -gama_select 0.0 -gama_rank 0.1 -gama_reg 0.01 -restore checkpoint_last.pt


CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content \
  -model var_select_var_user_diverse2seq_test -log 5c_var_select_testxx \
  -tau 0.5 -gama1 0.005 -gama_kld 0.05 -gama_select 0.05 -gama_rank 1.0 -gama_reg 0.01
