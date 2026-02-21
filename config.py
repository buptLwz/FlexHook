
import os
import torch
import yaml
from yacs.config import CfgNode as CN

# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
_C.DATA.VAL_BATCH_SIZE=40

# Dataset name
_C.DATA.DATASET = 'kitti-2'
_C.DATA.TRAIN = None
_C.DATA.TEST = None

# Input image size
_C.DATA.IMG_SIZE = (224,672)

_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

_C.freeze_text = False
_C.freeze_visual = False
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
# Model name
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = 'src'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []


# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0


# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''

# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False

# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# for acceleration
_C.FUSED_WINDOW_PROCESS = False
_C.FUSED_LAYERNORM = False

_C.sample_frame_len=8
_C.sample_frame_num=2
_C.sample_frame_stride=None
_C.sample_expression_num=1
_C.val_set_scale=None


_C.sample_alpha=2
_C.img_hw=[[224, 224], [448, 448], [448, 448]]
_C.random_crop_ratio=[1.0, 1.0]
_C.norm_mean=[0.48145466, 0.4578275, 0.40821073]
_C.norm_std=[0.26862954, 0.26130258, 0.27577711]

_C.data_root = ''
_C.val_data_root = ''

_C.track_root = ''
_C.convert = False
_C.freeze_text= False

_C.noise1 =False
_C.noise2 =False
_C.noise3 =False

_C.POSW = 0.1

_C.val_sample_frame_stride = None
_C.val_sample_expression_num = 8

_C.gauss_sample= True
_C.diff_eval = 208138 #16011
_C.text_len= 25
_C.entropy= -1

_C.VISUAL = 'rope-swin-tiny'
_C.TEXT = 'roberta'
_C.BERT_PATH = 'pretrained/bert-base-uncased'
_C.CLIP_PATH = 'pretrained/CLIP/RN50.pt'
_C.ROBERTA_PATH = 'pretrained/roberta-base'
_C.LRE = 1

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and (eval(f'args.{name}') is not None):
            return True
        return False
    # print(args)
    # assert 1==2
    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('val_batch_size'):
        config.DATA.VAL_BATCH_SIZE = args.val_batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path

    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
   
    if _check_args('output'):
        config.OUTPUT = args.output

    if _check_args('eval'):
        config.EVAL_MODE = args.eval
    # print(_check_args('lre'))
    if _check_args('lre'):
        config.LRE = args.lre
    # print(config.LRE)
    # print(hasattr(args, 'lre'))
    # print( eval(f'args.lre'))
    # assert 1==2
    if _check_args('dataset'):
        config.DATA.DATASET = args.dataset
    if _check_args('visual'):
        config.VISUAL = args.visual
    if _check_args('text'):
        config.TEXT = args.text

    if _check_args('freeze_text'):
        config.freeze_text = args.freeze_text
    if _check_args('freeze_visual'):
        config.freeze_visual = args.freeze_visual
    
    if _check_args('track_root'):
        config.track_root = args.track_root
    # for acceleration
    if _check_args('fused_window_process'):
        config.FUSED_WINDOW_PROCESS = True
    if _check_args('fused_layernorm'):
        config.FUSED_LAYERNORM = True
    ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    # set local rank for distributed training
    if PYTORCH_MAJOR_VERSION == 1:
        config.LOCAL_RANK = args.local_rank
    else:
        config.LOCAL_RANK = int(os.environ['LOCAL_RANK'])

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
