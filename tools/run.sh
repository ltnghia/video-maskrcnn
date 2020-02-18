cd /mnt/sdb/ltnghia/ITS_project/Code/video_mask_rcnn/tools
python -m torch.distributed.launch --nproc_per_node=2 train_net.py --config-file /mnt/sdb/ltnghia/ITS_project/Code/video_mask_rcnn/tools/e2e_faster_rcnn_R_50_FPN_1x_gn.yaml
