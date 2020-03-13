CUDA_VISIBLE_DEVICES=1 python load_model.py -gpus 1 -use_content \
  -model autoenc -log 5c_autoenc -restore checkpoint_best.pt -n_topic_num 100

CUDA_VISIBLE_DEVICES=0 python load_model.py -gpus 1 -use_content \
  -model autoenc_vae -log 5c_autoenc_vae_dy2_0.05  \
  -dynamic2 -gama_kld 0.05 -n_z 256 -restore checkpoint_best.pt -n_topic_num 100

CUDA_VISIBLE_DEVICES=1 python load_model.py -gpus 1 -use_content \
  -model autoenc_vae_bow -log 5c_autoenc_vae_bow_kld0.2  \
  -gama_kld 0.2 -n_z 64 -restore checkpoint_best.pt -n_topic_num 100