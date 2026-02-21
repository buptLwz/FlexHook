DATASET_NAME=kitti-2
EXP_NAME=try

CFG_NAME=train-kitti2.yaml
TEST_CFG_NAME=kitti2.yaml

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/train/${CFG_NAME} \
    --data-path 12 \
    --output ${DATASET_NAME}/${EXP_NAME} \
    --batch-size 7 \
    --val-batch-size 40 \
    --visual rope-swin-tiny \
    --text roberta \
    --pretrained src \

for i in $(seq 1 5)
do
    b=$(expr ${i} - 1)

    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --data-path 12 \
    --output ./test-${DATASET_NAME}/${EXP_NAME}/best${i} \
    --val-batch-size 40 \
    --visual rope-swin-tiny \
    --text roberta \
    --eval \
    --resume ./${DATASET_NAME}/${EXP_NAME}/ckpt_epoch_best_${b}.pth
done


###########################################################

DATASET_NAME=kitti-1
EXP_NAME=try

CFG_NAME=train-kitti1.yaml
TEST_CFG_NAME=kitti1.yaml

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/train/${CFG_NAME} \
    --data-path 12 \
    --output ${DATASET_NAME}/${EXP_NAME} \
    --batch-size 7 \
    --val-batch-size 40 \
    --visual rope-swin-tiny \
    --text roberta \
    --pretrained src \

for i in $(seq 1 5)
do
    b=$(expr ${i} - 1)

    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --data-path 12 \
    --output ./test-${DATASET_NAME}/${EXP_NAME}/best${i} \
    --val-batch-size 40 \
    --visual rope-swin-tiny \
    --text roberta \
    --eval \
    --resume ./${DATASET_NAME}/${EXP_NAME}/ckpt_epoch_best_${b}.pth
done


##########################################################

DATASET_NAME=dance
EXP_NAME=try

CFG_NAME=train-dance.yaml
TEST_CFG_NAME=dance.yaml

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/train/${CFG_NAME} \
    --data-path 12 \
    --output ${DATASET_NAME}/${EXP_NAME} \
    --batch-size 9 \
    --val-batch-size 40 \
    --visual rope-swin-tiny \
    --text roberta \
    --pretrained src 

for i in $(seq 1 5)
do
    b=$(expr ${i} - 1)

    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --data-path 12 \
    --output ./test-${DATASET_NAME}/${EXP_NAME}/best${i} \
    --val-batch-size 40 \
    --visual rope-swin-tiny \
    --text roberta \
    --eval \
    --resume ./${DATASET_NAME}/${EXP_NAME}/ckpt_epoch_best_${b}.pth 
done

