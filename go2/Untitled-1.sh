./isaaclab.sh -p /home/user/IsaacLab/source/isaaclab/ABS-main/go2/camrec.py \
  --policy /home/user/IsaacLab/logs/rsl_rl/go2_velocity/2026-03-27_14-35-33/exported/policy.pt \
  --enable_cameras \
  --num_envs 16 \
  --n_steps 1000 \
  --sample_every 5 \
  --sample_offset 2


 
./isaaclab.sh -p /home/user/IsaacLab/source/isaaclab/ABS-main/go2/train_depth_resnet.py \
  --data_folder /home/user/IsaacLab/source/isaaclab/ABS-main/logs/rec_cam_go2 \
  --label_file label_obs.pkl \
  --num_rays 11 \
  --resnet_type resnet18 \
  --batch_size 320 \
  --num_epochs 302 \
  --save_interval 50 \
  --exp_name go2_depth_obs



  Epoch 298: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 62.77it/s]
Test Loss: nan
Epoch 299:   0%|                                                                                                                      | 0/9 [00:00<?, ?it/s]Epoch 299/302 [0/2880] Loss: nan
Epoch 299: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 58.78it/s]
Test Loss: nan
Epoch 300:   0%|                                                                                                                      | 0/9 [00:00<?, ?it/s]Epoch 300/302 [0/2880] Loss: nan
Epoch 300: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 65.85it/s]
Test Loss: nan
Model saved: /home/user/IsaacLab/source/isaaclab/ABS-main/logs/depth_logs_go2/20260327-184701-resnet18-go2_depth_obs/depth_lidar_model_20260327-184701_300.pt
Epoch 301:   0%|                                                                                                                      | 0/9 [00:00<?, ?it/s]Epoch 301/302 [0/2880] Loss: nan
Epoch 301: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 65.18it/s]
Test Loss: nan
(env_isaaclab) user@user:~/IsaacLab$ 


poch 296: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 60.64it/s]
Test Loss: 0.133982
Epoch 297:   0%|                                                                                                                      | 0/9 [00:00<?, ?it/s]Epoch 297/302 [0/2880] Loss: 0.128113
Epoch 297: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 65.02it/s]
Test Loss: 0.134495
Epoch 298:   0%|                                                                                                                      | 0/9 [00:00<?, ?it/s]Epoch 298/302 [0/2880] Loss: 0.125709
Epoch 298: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 57.38it/s]
Test Loss: 0.134402
Epoch 299:   0%|                                                                                                                      | 0/9 [00:00<?, ?it/s]Epoch 299/302 [0/2880] Loss: 0.123289
Epoch 299: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 65.41it/s]
Test Loss: 0.135612
Epoch 300:   0%|                                                                                                                      | 0/9 [00:00<?, ?it/s]Epoch 300/302 [0/2880] Loss: 0.126228
Epoch 300: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 57.55it/s]
Test Loss: 0.135548
Model saved: /home/user/IsaacLab/source/isaaclab/ABS-main/logs/depth_logs_go2/20260327-185536-resnet18-go2_depth_obs/depth_lidar_model_20260327-185536_300.pt
Epoch 301:   0%|                                                                                                                      | 0/9 [00:00<?, ?it/s]Epoch 301/302 [0/2880] Loss: 0.143110
Epoch 301: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 59.88it/s]
Test Loss: 0.136725
