DATASET_NAME=mix
EXP_NAME=try
CFG_NAME=train-mix.yaml

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/train/${CFG_NAME} \
    --data-path 12 \
    --output ${DATASET_NAME}/${EXP_NAME} \
    --batch-size 10 \
    --val-batch-size 40 \
    --pretrained src

for i in $(seq 1 5)
do
    b=$(expr ${i} - 1)
    TEST_CFG_NAME=aed-tao-infer-tiny.yaml

    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --data-path 12 \
    --output ./test-${DATASET_NAME}/${EXP_NAME}/best${i}/aed-tao/ \
    --val-batch-size 50 \
    --eval \
    --resume ./${DATASET_NAME}/${EXP_NAME}/ckpt_epoch_best_${b}.pth

    TEST_CFG_NAME=bot-mot17-infer-tiny.yaml
    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --data-path 12 \
    --output ./test-${DATASET_NAME}/${EXP_NAME}/best${i}/bot-mot17/ \
    --val-batch-size 50 \
    --eval \
    --resume ./${DATASET_NAME}/${EXP_NAME}/ckpt_epoch_best_${b}.pth

    TEST_CFG_NAME=u2-visdrone-infer-tiny.yaml
    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --data-path 12 \
    --output ./test-${DATASET_NAME}/${EXP_NAME}/best${i}/u2-drone/ \
    --val-batch-size 50 \
    --eval \
    --resume ./${DATASET_NAME}/${EXP_NAME}/ckpt_epoch_best_${b}.pth

    TEST_CFG_NAME=mcbyte-sports-infer-tiny.yaml
    OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --data-path 12 \
    --output ./test-${DATASET_NAME}/${EXP_NAME}/best${i}/mcbyte-sports/ \
    --val-batch-size 50 \
    --eval \
    --resume ./${DATASET_NAME}/${EXP_NAME}/ckpt_epoch_best_${b}.pth
    
done