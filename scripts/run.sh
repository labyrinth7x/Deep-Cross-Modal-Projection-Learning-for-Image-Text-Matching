GPUS=6
export CUDA_VISIBLE_DEVICES=$GPUS

BASE_ROOT=/home/zhangqi/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching
DATASET_DIR=$BASE_ROOT/data/processed_data
CKPT_DIR=$BASE_ROOT/data/model_data
LOG_DIR=$BASE_ROOT/data/logs
PRETRAINED_PATH=$BASE_ROOT/mobilenet_sgd_rmsprop_69.526.tar
#PRETRAINED_PATH=$BASE_ROOT/resnet50.pth
IMAGE_MODEL=mobilenet_v1
lr=0.0002
num_epoches=120
batch_size=16

python3.5 $BASE_ROOT/train.py \
    --CMPC \
    --CMPM \
    --pretrained \
    --pretrained_path $PRETRAINED_PATH \
    --image_model $IMAGE_MODEL \
    --log_dir $LOG_DIR/$lr \
    --checkpoint_dir $CKPT_DIR/$lr \
    --dataset_dir $DATASET_DIR \
    --batch_size $batch_size \
    --gpus $GPUS \
    --num_epoches $num_epoches \
    --lr $lr

python3.5 ${BASE_ROOT}/test.py \
    --checkpoint_dir $CKPT_DIR/$lr \
    --log_dir $LOG_DIR/$lr \
    --dataset_dir $DATASET_DIR \
    --gpus $GPUS \
    --epoch_start 0
