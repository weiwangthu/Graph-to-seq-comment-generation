#!/usr/bin/env bash
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
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model var_select_var_user_diverse2seq_test -log 5c_var_select_var_user_diverse2seq_test2_tau0.5_gama0.005_kld0.05_sel0.05_r1_re0.01_fix \
  -tau 0.5 -gama1 0.005 -gama_kld 0.05 -gama_select 0.05 -gama_rank 1.0 -gama_reg 0.01 -topic
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model var_select_var_user_diverse2seq_test -log 5c_var_select_var_user_diverse2seq_test2_tau0.5_gama0.005_kld0.05_sel0.1_r1_re0.01 \
  -tau 0.5 -gama1 0.005 -gama_kld 0.05 -gama_select 0.1 -gama_rank 1.0 -gama_reg 0.01 -topic

# var select and user
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model var_select2seq_test -log 5c_var_select2seq_test_tau0.5_gama0_sel0.1 \
  -tau 0.5 -gama1 0 -gama_select 0.1

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model var_select_user2seq_test -log 5c_var_select_user2seq_test_tau0.5_gama0_sel0.5_re0_m0.1 \
  -tau 0.5 -gama1 0 -gama_select 0.5 -gama_reg 0 -n_z 256 -min_select 0.1 -topic

CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_last.pt \
  -model var_select_user2seq_new -log 5c_var_select_user2seq_new_tau0.5_re0_kld0.2_t100_dy3_0.2_gama0_klds0.5_m0.1  \
  -tau 0.5 -gama_reg 0 -gama_kld 0.2 -n_z 64 -n_topic_num 100 -dynamic3 -gama_select 0.2 -mid_max 20 \
  -gama1 0 -gama_kld_select 0.5 -min_select 0.1 -topic
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_last.pt \
  -model var_select_user2seq_new -log 5c_var_select_user2seq_new_tau0.5_re0_kld0.2_t100_dy3_0.2_gama0_klds0.5_m0.1_cone  \
  -tau 0.5 -gama_reg 0 -gama_kld 0.2 -n_z 64 -n_topic_num 100 -dynamic3 -gama_select 0.2 -mid_max 20 \
  -gama1 0 -gama_kld_select 0.5 -min_select 0.1 -con_one_user -topic

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model user2seq_test -log 5c_user2seq_test_re0 \
  -gama_reg 0 -n_z 256 -topic
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model user2seq_test -log 5c_user2seq_test_re0.01 \
  -gama_reg 0.01 -n_z 256 -topic
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model user2seq_test -log 5c_user2seq_test_tau0.5_re0.01_one \
  -tau 0.5 -gama_reg 0.01 -n_z 256 -one_user -topic
CUDA_VISIBLE_DEVICES=3 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model user2seq_test -log 5c_user2seq_test_re1 \
  -gama_reg 1 -n_z 256 -topic
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_last.pt \
  -model user2seq_test -log 5c_user2seq_test_tau0.5_re0_r1_one \
  -tau 0.5 -gama_reg 0 -gama_rank 1 -n_z 256 -one_user -topic
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_last.pt \
  -model user2seq_test -log 5c_user2seq_test_tau0.5_re0_r1_one_loss2 \
  -tau 0.5 -gama_reg 0 -gama_rank 1 -n_z 256 -one_user -topic

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_last.pt \
  -model user2seq_test_new -log 5c_user2seq_test_new_tau0.5_re0_kld0.2_t100_dy3_0.2  \
  -tau 0.5 -gama_reg 0 -gama_kld 0.2 -n_z 64 -n_topic_num 100 -dynamic3 -gama_select 0.2 -mid_max 20 -topic
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_last.pt \
  -model user2seq_test_new -log 5c_user2seq_test_new_tau0.5_re0_kld0.2_t100_dy3_0.2_cone  \
  -tau 0.5 -gama_reg 0 -gama_kld 0.2 -n_z 64 -n_topic_num 100 -dynamic3 -gama_select 0.2 -mid_max 20 -con_one_user -topic


# vae, vae bow
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model autoenc_vae -log 5c_autoenc_vae_dy2_0.05  \
  -dynamic2 -gama_kld 0.05 -n_z 256

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_best.pt \
  -model autoenc_vae_bow -log 5c_autoenc_vae_bow_kld0.2  \
  -gama_kld 0.2 -n_z 64

CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_last.pt \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re0.1_sel1_fixl_fixu_dy1_0.05_10 \
  -tau 0.5 -gama_reg 0.1 -gama_select 1 -n_z 256 -dynamic1 -gama_kld 0.05 -mid_max 10.0 -topic
CUDA_VISIBLE_DEVICES=0 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_last.pt \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re1_sel1_fixl_fixu_dy1_0.05_10 \
  -tau 0.5 -gama_reg 1 -gama_select 1 -n_z 256 -dynamic1 -gama_kld 0.05 -mid_max 10.0 -topic
CUDA_VISIBLE_DEVICES=1 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_last.pt \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re1_sel1_fixl_fixu_dy1_0.05_10_t100 \
  -tau 0.5 -gama_reg 1 -gama_select 1 -n_z 256 -dynamic1 -gama_kld 0.05 -mid_max 10.0 -n_topic_num 100 -topic
CUDA_VISIBLE_DEVICES=2 python train.py -gpus 1 -use_content -notrain -beam_search -restore checkpoint_last.pt \
  -model user_autoenc_vae -log 5c_user_autoenc_vae_tau0.5_re1_sel1_fixl_fixu_dy1_0.05_10_t1000 \
  -tau 0.5 -gama_reg 1 -gama_select 1 -n_z 256 -dynamic1 -gama_kld 0.05 -mid_max 10.0 -n_topic_num 1000 -topic


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
