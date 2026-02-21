# refer-kitti-2-best TempRMOT+NeuralSORT 42.53 HOTA
# HOTA DetA AssA DetRe DetPr AssRe AssPr LocA RHOTA HOTA(0) LocA(0) HOTALocA(0) Dets GT_Dets IDs GT_IDs
# 42.526 30.627 59.189 58.552 38.307 64.693 85.047 89.088 58.843 48.331 84.486 40.833 750834 491222 30871 8043

DATASET_NAME=kitti-2
CKPT_PATH=SOTA_ckpts/refer-kitti-v2-best
TRACKER_ROOT=Temp-NeuralSORT-kitti2
TEST_CFG_NAME=kitti2.yaml
EXP_NAME=refer-kitti-v2-best

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


# # refer-kitti-2-resnet34 TempRMOT+NeuralSORT 41.50 HOTA
# HOTA DetA AssA DetRe DetPr AssRe AssPr LocA RHOTA HOTA(0) LocA(0) HOTALocA(0) Dets GT_Dets IDs GT_IDs
# 41.499 30.068 57.396 53.437 39.908 62.254 85.863 89.306 55.36 47.007 84.665 39.799 657761 491222 28637 8043

DATASET_NAME=kitti-2
CKPT_PATH=SOTA_ckpts/refer-kitti-v2-resnet34
TRACKER_ROOT=Temp-NeuralSORT-kitti2
TEST_CFG_NAME=kitti2.yaml
EXP_NAME=refer-kitti-v2-resnet34

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


# refer-kitti-2-clip TempRMOT+NeuralSORT 41.42 HOTA
# HOTA DetA AssA DetRe DetPr AssRe AssPr LocA RHOTA HOTA(0) LocA(0) HOTALocA(0) Dets GT_Dets IDs GT_IDs
# 41.424 29.057 59.195 60.342 35.246 65.759 84.23 89.1 59.738 46.972 84.526 39.704 840979 491222 34025 8043

DATASET_NAME=kitti-2
CKPT_PATH=SOTA_ckpts/refer-kitti-v2-clip
TRACKER_ROOT=Temp-NeuralSORT-kitti2
TEST_CFG_NAME=kitti2.yaml
EXP_NAME=refer-kitti-v2-clip

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


# refer-kitti-1-best TempRMOT+NeuralSORT 53.82 HOTA Some Typos in Paper
# HOTA DetA AssA DetRe DetPr AssRe AssPr LocA RHOTA HOTA(0) LocA(0) HOTALocA(0) Dets GT_Dets IDs GT_IDs
# 53.824 43.354 66.916 59.706 59.833 71.672 88.658 90.872 63.202 59.82 88.257 52.795 66320 66461 2838 1511

DATASET_NAME=kitti-1
CKPT_PATH=SOTA_ckpts/refer-kitti-best
TRACKER_ROOT=Temp-NeuralSORT-kitti1
TEST_CFG_NAME=kitti1.yaml
EXP_NAME=refer-kitti-best

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


# refer-kitti-1-resnet34 TempRMOT+NeuralSORT 53.61 HOTA Some Typos in the paper
# HOTA DetA AssA DetRe DetPr AssRe AssPr LocA RHOTA HOTA(0) LocA(0) HOTALocA(0) Dets GT_Dets IDs GT_IDs
# 53.605 42.762 67.277 63.531 55.402 73.005 87.826 90.74 65.373 59.704 87.95 52.51 76212 66461 3154 1511

DATASET_NAME=kitti-1
CKPT_PATH=SOTA_ckpts/refer-kitti-resnet34
TRACKER_ROOT=Temp-NeuralSORT-kitti1
TEST_CFG_NAME=kitti1.yaml
EXP_NAME=refer-kitti-resnet34

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


# refer-kitti-1-clip TempRMOT+NeuralSORT 53.45 HOTA
# HOTA DetA AssA DetRe DetPr AssRe AssPr LocA RHOTA HOTA(0) LocA(0) HOTALocA(0) Dets GT_Dets IDs GT_IDs
# 53.45 42.685 67.008 62.115 56.402 71.779 88.49 90.814 64.511 59.444 88.175 52.415 73193 66461 3133 1511

DATASET_NAME=kitti-1
CKPT_PATH=SOTA_ckpts/refer-kitti-clip
TRACKER_ROOT=Temp-NeuralSORT-kitti1
TEST_CFG_NAME=kitti1.yaml
EXP_NAME=refer-kitti-clip

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


# refer-dance-best DETR+NeuralSORT 32.17 HOTA
# HOTA DetA AssA DetRe DetPr AssRe AssPr LocA RHOTA HOTA(0) LocA(0) HOTALocA(0) Dets GT_Dets IDs GT_IDs
# 32.165 28.364 36.533 48.212 40.366 42.214 65.621 91.313 41.957 35.59 88.148 31.372 253853 212541 1412 252

DATASET_NAME=dance
CKPT_PATH=SOTA_ckpts/refer-dance-best
TRACKER_ROOT=DETR-NeuralSORT-Dance
TEST_CFG_NAME=dance.yaml
EXP_NAME=refer-dance-best

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

# mix AED(TAO)+BoT(MOT17)+u2mot(VisDrone)+McByte(SportsMOT) 56.76 HOTA Some Typos in the paper
# HOTA DetA AssA DetRe DetPr AssRe AssPr LocA RHOTA HOTA(0) LocA(0) HOTALocA(0) Dets GT_Dets IDs GT_IDs
# 56.756 47.234 68.239 75.242 55.056 70.39 92.459 91.96 71.652 61.857 89.308 55.243 2364489 1730146 123820 9378

DATASET_NAME=mix
CKPT_PATH=SOTA_ckpts/LaMOT-best
TEST_CFG_NAME=aed-tao-infer-tiny.yaml
EXP_NAME=LaMOT-best

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/aed-tao/ \
    --val-batch-size 50 \
    --eval \
    --resume ${CKPT_PATH}.pth \

TEST_CFG_NAME=bot-mot17-infer-tiny.yaml
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/bot-mot17/ \
    --val-batch-size 50 \
    --eval \
    --resume ${CKPT_PATH}.pth \

TEST_CFG_NAME=u2-visdrone-infer-tiny.yaml
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/u2-drone/ \
    --val-batch-size 50 \
    --eval \
    --resume ${CKPT_PATH}.pth \

TEST_CFG_NAME=mcbyte-sports-infer-tiny.yaml
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master-port 12345 main.py  \
    --cfg configs/infer/${TEST_CFG_NAME} \
    --data-path 12 \
    --output ./retest-${DATASET_NAME}/${EXP_NAME}/mcbyte-sports/ \
    --val-batch-size 50 \
    --eval \
    --resume ${CKPT_PATH}.pth \
