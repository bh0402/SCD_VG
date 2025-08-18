python train.py --labels age --dec_dist implicit --d_steps_per_iter 1 --sample_every_epoch 5 --latent_dim 100 --enc_arch resnet --save_model_every 5 --n_epochs 200 --prior scm_flow --sup_prop 1 --sup_coef 7 --sup_type ce

