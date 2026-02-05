CUDA_VISIBLE_DEVICES=0,3  bash tools/dist_train.sh \
 configs/rsprompter/rsprompter_query-tfds.py \
 2 \
 --work-dir work_dirs/coco_all_seg/rsprompter_query-tfds_10_4_last