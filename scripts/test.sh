GPUS=3
export CUDA_VISIBLE_DEVICES=$GPUS

BASE_ROOT=/home/zhangqi/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching
IMAGE_DIR=/home/zhangqi/TriCrossModalV2/data/
ANNO_DIR=$BASE_ROOT/data/processed_data
CKPT_DIR=$BASE_ROOT/data/model_data
LOG_DIR=$BASE_ROOT/data/logs
IMAGE_MODEL=mobilenet_v1
lr=0.0002
batch_size=16
lr_decay_ratio=0.9
epoches_decay=80_150_200


python3.5 ${BASE_ROOT}/test.py \
    --bidirectional \
    --model_path $CKPT_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --image_model $IMAGE_MODEL \
    --log_dir $LOG_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --image_dir $IMAGE_DIR \
    --anno_dir $ANNO_DIR \
    --gpus $GPUS \
    --epoch_ema 0
