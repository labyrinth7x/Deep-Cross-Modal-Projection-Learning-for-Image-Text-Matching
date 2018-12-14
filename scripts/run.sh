GPUS=4
export CUDA_VISIBLE_DEVICES=${GPUS}

BASE_ROOT=/home/zhangqi/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching
DATASET_DIR=$BASE_ROOT/data/processed_data
CKPT_DIR=$BASE_ROOT/data/model_data_none
LOG_DIR=$BASE_ROOT/data/logs_start_none

python3.5 ${BASE_ROOT}/train.py \
    --checkpoint_dir ${CKPT_DIR} \
    --log_dir ${LOG_DIR} \
    --dataset_dir ${DATASET_DIR} \
    --gpus ${GPUS} \
    --num_epoches 10 \
    --resume \
    --pretrained_path ${CKPT_DIR}/49.pth.tar

python3.5 ${BASE_ROOT}/test.py \
    --checkpoint_dir ${CKPT_DIR} \
    --log_dir ${LOG_DIR} \
    --dataset_dir ${DATASET_DIR} \
    --gpus ${GPUS} \
    --epoch_start 0
