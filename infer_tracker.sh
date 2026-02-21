TRACKER_ROOT=Temp-StrongSORT-kitti2


DATASET_NAME=kitti-2
CKPT_PATH=SOTA_ckpts/refer-kitti-v2-best
TEST_CFG_NAME=kitti2.yaml
EXP_NAME=refer-kitti-v2-best-${TRACKER_ROOT}

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --track-root tracker_outputs/${TRACKER_ROOT}\
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/ \
    --val-batch-size 80 \
    --visual rope-swin-tiny \
    --text roberta \
    --eval \
    --resume ${CKPT_PATH}.pth \



DATASET_NAME=kitti-2
CKPT_PATH=SOTA_ckpts/refer-kitti-v2-resnet34
TEST_CFG_NAME=kitti2.yaml
EXP_NAME=refer-kitti-v2-resnet34-${TRACKER_ROOT}

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --track-root tracker_outputs/${TRACKER_ROOT}\
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/ \
    --val-batch-size 80 \
    --visual resnet34 \
    --text roberta \
    --eval \
    --resume ${CKPT_PATH}.pth \



DATASET_NAME=kitti-2
CKPT_PATH=SOTA_ckpts/refer-kitti-v2-clip
TEST_CFG_NAME=kitti2.yaml
EXP_NAME=refer-kitti-v2-clip-${TRACKER_ROOT}

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --track-root tracker_outputs/${TRACKER_ROOT}\
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/ \
    --val-batch-size 80 \
    --visual clip \
    --text clip \
    --eval \
    --resume ${CKPT_PATH}.pth \

####################################################################################################################################################

TRACKER_ROOT=Temp-OCSORT-kitti2


DATASET_NAME=kitti-2
CKPT_PATH=SOTA_ckpts/refer-kitti-v2-best
TEST_CFG_NAME=kitti2.yaml
EXP_NAME=refer-kitti-v2-best-${TRACKER_ROOT}

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --track-root tracker_outputs/${TRACKER_ROOT}\
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/ \
    --val-batch-size 80 \
    --visual rope-swin-tiny \
    --text roberta \
    --eval \
    --resume ${CKPT_PATH}.pth \



DATASET_NAME=kitti-2
CKPT_PATH=SOTA_ckpts/refer-kitti-v2-resnet34
TEST_CFG_NAME=kitti2.yaml
EXP_NAME=refer-kitti-v2-resnet34-${TRACKER_ROOT}

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --track-root tracker_outputs/${TRACKER_ROOT}\
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/ \
    --val-batch-size 80 \
    --visual resnet34 \
    --text roberta \
    --eval \
    --resume ${CKPT_PATH}.pth \



DATASET_NAME=kitti-2
CKPT_PATH=SOTA_ckpts/refer-kitti-v2-clip
TEST_CFG_NAME=kitti2.yaml
EXP_NAME=refer-kitti-v2-clip-${TRACKER_ROOT}

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --track-root tracker_outputs/${TRACKER_ROOT}\
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/ \
    --val-batch-size 80 \
    --visual clip \
    --text clip \
    --eval \
    --resume ${CKPT_PATH}.pth \

####################################################################################################################################################
TRACKER_ROOT=DETR-NeuralSORT-kitti2


DATASET_NAME=kitti-2
CKPT_PATH=SOTA_ckpts/refer-kitti-v2-best
TEST_CFG_NAME=kitti2.yaml
EXP_NAME=refer-kitti-v2-best-${TRACKER_ROOT}

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --track-root tracker_outputs/${TRACKER_ROOT}\
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/ \
    --val-batch-size 80 \
    --visual rope-swin-tiny \
    --text roberta \
    --eval \
    --resume ${CKPT_PATH}.pth \



DATASET_NAME=kitti-2
CKPT_PATH=SOTA_ckpts/refer-kitti-v2-resnet34
TEST_CFG_NAME=kitti2.yaml
EXP_NAME=refer-kitti-v2-resnet34-${TRACKER_ROOT}

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --track-root tracker_outputs/${TRACKER_ROOT}\
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/ \
    --val-batch-size 80 \
    --visual resnet34 \
    --text roberta \
    --eval \
    --resume ${CKPT_PATH}.pth \



DATASET_NAME=kitti-2
CKPT_PATH=SOTA_ckpts/refer-kitti-v2-clip
TEST_CFG_NAME=kitti2.yaml
EXP_NAME=refer-kitti-v2-clip-${TRACKER_ROOT}

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --track-root tracker_outputs/${TRACKER_ROOT}\
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/ \
    --val-batch-size 80 \
    --visual clip \
    --text clip \
    --eval \
    --resume ${CKPT_PATH}.pth \

####################################################################################################################################################

TRACKER_ROOT=DETR-StrongSORT-kitti2


DATASET_NAME=kitti-2
CKPT_PATH=SOTA_ckpts/refer-kitti-v2-best
TEST_CFG_NAME=kitti2.yaml
EXP_NAME=refer-kitti-v2-best-${TRACKER_ROOT}

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --track-root tracker_outputs/${TRACKER_ROOT}\
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/ \
    --val-batch-size 80 \
    --visual rope-swin-tiny \
    --text roberta \
    --eval \
    --resume ${CKPT_PATH}.pth \



DATASET_NAME=kitti-2
CKPT_PATH=SOTA_ckpts/refer-kitti-v2-resnet34
TEST_CFG_NAME=kitti2.yaml
EXP_NAME=refer-kitti-v2-resnet34-${TRACKER_ROOT}

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --track-root tracker_outputs/${TRACKER_ROOT}\
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/ \
    --val-batch-size 80 \
    --visual resnet34 \
    --text roberta \
    --eval \
    --resume ${CKPT_PATH}.pth \



DATASET_NAME=kitti-2
CKPT_PATH=SOTA_ckpts/refer-kitti-v2-clip
TEST_CFG_NAME=kitti2.yaml
EXP_NAME=refer-kitti-v2-clip-${TRACKER_ROOT}

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --track-root tracker_outputs/${TRACKER_ROOT}\
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/ \
    --val-batch-size 80 \
    --visual clip \
    --text clip \
    --eval \
    --resume ${CKPT_PATH}.pth \

####################################################################################################################################################

TRACKER_ROOT=DETR-OCSORT-kitti2


DATASET_NAME=kitti-2
CKPT_PATH=SOTA_ckpts/refer-kitti-v2-best
TEST_CFG_NAME=kitti2.yaml
EXP_NAME=refer-kitti-v2-best-${TRACKER_ROOT}

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --track-root tracker_outputs/${TRACKER_ROOT}\
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/ \
    --val-batch-size 80 \
    --visual rope-swin-tiny \
    --text roberta \
    --eval \
    --resume ${CKPT_PATH}.pth \



DATASET_NAME=kitti-2
CKPT_PATH=SOTA_ckpts/refer-kitti-v2-resnet34
TEST_CFG_NAME=kitti2.yaml
EXP_NAME=refer-kitti-v2-resnet34-${TRACKER_ROOT}

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --track-root tracker_outputs/${TRACKER_ROOT}\
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/ \
    --val-batch-size 80 \
    --visual resnet34 \
    --text roberta \
    --eval \
    --resume ${CKPT_PATH}.pth \



DATASET_NAME=kitti-2
CKPT_PATH=SOTA_ckpts/refer-kitti-v2-clip
TEST_CFG_NAME=kitti2.yaml
EXP_NAME=refer-kitti-v2-clip-${TRACKER_ROOT}

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --track-root tracker_outputs/${TRACKER_ROOT}\
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/ \
    --val-batch-size 80 \
    --visual clip \
    --text clip \
    --eval \
    --resume ${CKPT_PATH}.pth \

