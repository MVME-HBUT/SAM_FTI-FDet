# CUDA_VISIBLE_DEVICES=1 python tools/test.py \
#  configs/rsprompter/samseg-mask2former-nwpu.py \
#  work_dirs/coco_all_seg/samseg-mask2former-nwpu/best_coco_segm_mAP_epoch_96.pth \
#  --work-dir work_dirs/coco_all_seg/test 

CUDA_VISIBLE_DEVICES=0 python tools/test.py \
 configs/rsprompter/rsprompter_query-tfds_mask2former.py \
 /home/user/RSPrompter-release/work_dirs/coco_all_seg/rsprompter_query-tfds_10_5_mask2former_12_1024/best_coco_segm_mAP_epoch_90.pth \
 --work-dir work_dirs/coco_all_seg/test 