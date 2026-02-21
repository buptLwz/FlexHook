
import os

import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
# torch.autograd.set_detect_anomaly(True)
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained,load_ckpt_my, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor_sum,reduce_tensor,box_cxcywh_to_xyxy,generalized_box_iou
from tensorboardX import SummaryWriter
from test_utils import generate_final_results,generate_final_results_dance
from test_utils_mix import generate_final_results_mix
# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])
from collections import defaultdict
from time import sleep
import torch.nn.functional as F
from loss import FocalLoss
from fvcore.nn import sigmoid_focal_loss
#torch.backends.cudnn.benchmark = True
def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', default='',type=str, required=False, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', default='8',type=int, help="batch size for single GPU")
    parser.add_argument('--val-batch-size', default='40',type=int, help="batch size for single GPU")
    parser.add_argument('--visual',type=str, help="visual")
    parser.add_argument('--text',type=str, help="test")
    parser.add_argument('--dataset',type=str, help="dataset")


    parser.add_argument('--freeze-text',action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--freeze-visual',action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--track-root',type=str,help="dataset")




    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='default-try', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')
    
    parser.add_argument('--lre', type=int, help='LRE.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def main(config):

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.VISUAL}+{config.TEXT}")
    model = build_model(config)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    max_accuracy = 0.0

    if config.MODEL.RESUME:        
        if config.EVAL_MODE:
            # For multi-resolution evaluation
            config.defrost()
            config.MODEL.PRETRAINED = config.MODEL.RESUME
            config.freeze()
            load_ckpt_my(config, model_without_ddp, logger)
        else:
            max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        #acc1, acc5, loss = validate(config, data_loader_val, model)
        output_path = os.path.join(config.OUTPUT,'result.json')
        if not os.path.exists(output_path):

            if not os.path.exists(output_path.replace('result.json',f'result_{dist.get_rank()}.json')):
                logger.info(f"Start inference on {len(data_loader_val)} validation images")
                with torch.no_grad():
                    res_dict = inference(config, data_loader_val, model)
                torch.cuda.empty_cache()
                # os.makedirs(os.path.join(OUTPUT), exist_ok=True)
                json.dump(
                        res_dict,
                        open(output_path.replace('result.json',f'result_{dist.get_rank()}.json'), 'w'),
                        indent=1
                    )
            dist.barrier()
            
            
            if dist.get_rank() == 0:
                dlist=[]
                for i in range(dist.get_world_size()):
                    dname= output_path.replace('result.json',f'result_{i}.json')
                    if os.path.exists(dname):
                        dlist.append(json.load(open(dname)))
                    else:
                        sleep(10)
                        dlist.append(json.load(open(dname)))

                def deep_merge_dict(d1, d2):
                    for k, v in d2.items():
                        if k not in d1:
                            d1[k] = v
                        else:
                            if isinstance(v, dict) and isinstance(d1[k], dict):
                                deep_merge_dict(d1[k], v)
                            else:
                                d1[k] = v

                for i in range(1, len(dlist)):
                    deep_merge_dict(dlist[0], dlist[i])


                json.dump(
                    dlist[0],
                    open(output_path, 'w'),indent=1
                )
                CLS_DICT = dlist[0]
            # torch.cuda.synchronize()
        else:
            if dist.get_rank() == 0:
                CLS_DICT = json.load(open(output_path))
        dist.barrier()
        if dist.get_rank() == 0:
            SAVE_DIR = os.path.abspath(os.path.join(config.OUTPUT, 'results'))

            if not os.path.exists(SAVE_DIR):
                if config.DATA.DATASET=='kitti-1' or config.DATA.DATASET=='kitti-2':
                    generate_final_results(
                        config=config,
                        cls_dict=CLS_DICT,
                        data_dir=os.path.abspath(config.data_root),
                        track_dir=os.path.abspath(config.track_root),
                        save_dir=SAVE_DIR,
                        )
                elif config.DATA.DATASET=='dance':
                    generate_final_results_dance(
                        config=config,
                        cls_dict=CLS_DICT,
                        data_dir=os.path.abspath(config.data_root),
                        track_dir=os.path.abspath(config.track_root),
                        save_dir=SAVE_DIR,
                        )
                else:
                    generate_final_results_mix(
                        config=config,
                        cls_dict=CLS_DICT,
                        data_dir=os.path.abspath(config.data_root),
                        track_dir=os.path.abspath(config.track_root),
                        save_dir=SAVE_DIR,
                        )
            

            if config.DATA.TEST:
                seqfile = os.path.join('./seqmaps',config.DATA.TEST+'.txt')
            else:
                seqfile = os.path.join('./seqmaps',config.DATA.DATASET+'.txt')

            os.system('python3 ./TrackEval/scripts/run_mot_challenge.py '+\
            '--METRICS HOTA '+\
            f'--SEQMAP_FILE {seqfile} '+\
            '--SKIP_SPLIT_FOL True '+\
            '--GT_FOLDER ./datasets/refer-kitti/KITTI/training/image_02 '+\
            f'--TRACKERS_FOLDER {SAVE_DIR} '+\
            '--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt '+\
            f'--TRACKERS_TO_EVAL {SAVE_DIR} '+\
            '--USE_PARALLEL True '+\
            '--NUM_PARALLEL_CORES 2 '+\
            '--SKIP_SPLIT_FOL True '+\
            '--PLOT_CURVES False')
        # dist.barrier()
        

        if config.EVAL_MODE:
            return

    f1 = 0
    if dist.get_rank()==0:
        writer = SummaryWriter(log_dir=config.OUTPUT)
    else:
        writer = None

    if (not (config.MODEL.PRETRAINED == 'src')) and (not config.MODEL.RESUME):
        load_ckpt_my(config, model_without_ddp, logger)
        
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {f1:.1f}%")

    
    f1_list=[torch.tensor(-1.0)]
    gtf1_list=[torch.tensor(-1.0)]
    re_list=[torch.tensor(-1.0)]
    gtre_list=[torch.tensor(-1.0)]

    pr_list=[torch.tensor(-1.0)]


    best_list=[-1]
    logger.info("Start training")
    start_time = time.time()


    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        data_loader_train.sampler.set_epoch(epoch)

        if dist.get_rank() == 0 and epoch == 0: #for random init weight
            save_checkpoint(config, epoch-1, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger,F1=f1,bestnumber=None)
        torch.cuda.synchronize()

        cur_diff = train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,loss_scaler,writer,posweight=None)

        torch.cuda.empty_cache()

        if epoch >= 0:

            with torch.no_grad():
                gtf1,f1,gtrecall, recall, precision = validate(config, data_loader_val, model,epoch,writer,None,None,config.diff_eval)
            torch.cuda.empty_cache()
            if (gtf1>= gtf1_list[-1]):
                gtf1_list.append(gtf1)
                gtf1_list.sort(reverse=True)

                for i,value in enumerate(gtf1_list):
                    if value ==gtf1:
                        this_epoch_best_number=i
                        best_list=best_list[:i]+[epoch]+best_list[i:]
                        f1_list=f1_list[:i]+[f1]+f1_list[i:]
                        re_list=re_list[:i]+[recall]+re_list[i:]
                        gtre_list=gtre_list[:i]+[gtrecall]+gtre_list[i:]

                        pr_list=pr_list[:i]+[precision]+pr_list[i:]

                        break
                if len(gtf1_list)>5:
                    gtf1_list = gtf1_list[:-1]
                    f1_list = f1_list[:-1]
                    re_list = re_list[:-1]
                    gtre_list = gtre_list[:-1]

                    pr_list = pr_list[:-1]

                    best_list = best_list[:-1]
            else:
                this_epoch_best_number=None


            if dist.get_rank() == 0 and ((this_epoch_best_number is not None)):#and ((epoch+1) % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1) or (epoch+1) > config.DENSE_SAVE_EPOCH):
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger,F1=f1,bestnumber=this_epoch_best_number)

            
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {f1:.1f}%")
        max_accuracy = max(max_accuracy, f1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        logger.info(f'Best list: {best_list}')
        logger.info(f'Pr list: {[i.item() for i in pr_list]}')
        logger.info(f'gtRe list: {[i.item() for i in gtre_list]}')
        logger.info(f'gtF1 list: {[i.item() for i in gtf1_list]}')
        logger.info(f'Re list: {[i.item() for i in re_list]}')
        logger.info(f'F1 list: {[i.item() for i in f1_list]}')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler,writer,posweight=None):
    model.train()

    model.module.sample_expression_num = config.sample_expression_num
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_pos_meter = AverageMeter()
    loss_refer_meter = AverageMeter()

    loss_regular_meter = AverageMeter()

    loss_neg_meter = AverageMeter()



    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    sample_dict_for_save={}
    for idx, (samples, pes,bbox_gt,targets, expid,expma, data_key,sampled_indices,sampled_target_exp,index) in enumerate(data_loader):

        samples = samples.cuda(non_blocking=True) #btchw

        expid= expid.cuda(non_blocking=True) #
        expma = expma.cuda(non_blocking=True)#
        pes = pes.cuda(non_blocking=True) # bt2hw

        bbox_gt = bbox_gt.cuda(non_blocking=True) #bt4
        sampled_target_exp=list(map(list, zip(*sampled_target_exp)))

        if config.noise3:
            replace1=None
            for bi in range(bbox_gt.shape[0]):
                for ti in range(bbox_gt.shape[1]):
                    if torch.rand(1)<0.1:
                        replace1=[bi,ti]
                        break
                    
            if replace1 is not None:
                replace2 =None
                for bi in range(bbox_gt.shape[0]):
                    if bi==replace1[0]:
                        continue
                    for ti in range(bbox_gt.shape[1]):
                        if torch.rand(1)<0.1:
                            replace2=[bi,ti]
                            break
                if replace2 is not None:
                    temp = bbox_gt[replace1[0],replace1[1],:].clone()
                    bbox_gt[replace1[0],replace1[1],:]=bbox_gt[replace2[0],replace2[1],:]
                    bbox_gt[replace2[0],replace2[1],:]=temp

                    temp = samples[replace1[0],replace1[1],:,:,:].clone()
                    samples[replace1[0],replace1[1],:,:,:]=samples[replace2[0],replace2[1],:,:,:]
                    samples[replace2[0],replace2[1],:,:,:]=temp

                    temp = pes[replace1[0],replace1[1],:,:,:].clone()
                    pes[replace1[0],replace1[1],:,:,:]=pes[replace2[0],replace2[1],:,:,:]
                    pes[replace2[0],replace2[1],:,:,:]=temp

        
        outputs,regular = model(samples,pes,bbox_gt,expid,expma)

        f_,t_,n_,c_ = outputs.shape
        
        targets = targets.cuda(non_blocking=True)
        
        loss_refer= criterion(outputs.flatten(0,2), targets.unsqueeze(1).repeat(1,t_,1).flatten(0,2))

        with torch.no_grad():
            pos_weight = targets.unsqueeze(1).repeat(1,t_,1).flatten()
            if pos_weight.sum() == 0:
                pos_loss = 0
            else:
                pos_loss = ((loss_refer*pos_weight).sum())/(pos_weight.sum())
            
            if (1-pos_weight).sum() == 0:
                neg_loss = 0
            else:
                neg_loss = ((loss_refer*(1-pos_weight)).sum())/((1-pos_weight).sum())
            loss_pos_meter.update(pos_loss)
            loss_neg_meter.update(neg_loss)

            if config.POSW >0:
                pos_weight = config.POSW*pos_weight+(1-config.POSW)*(1-pos_weight)
            entropy = (-outputs.flatten(0,2).softmax(-1)*F.log_softmax(outputs.flatten(0,2))).sum(-1)

        if config.POSW == -2:
            loss = (loss_refer).mean()
            
        elif config.POSW == -1:
            loss = ((entropy)*loss_refer).mean()
        elif config.POSW >0:
            if config.entropy>0:
                loss_refer = (entropy**config.entropy)*loss_refer

            loss = (loss_refer*pos_weight).mean()
        
        loss = loss+regular

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        loss_scale_value = 0

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        loss_refer_meter.update(loss_refer.detach().mean().item(), targets.size(0))

        loss_regular_meter.update(regular.detach().mean().item(), targets.size(0))


        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            lr = optimizer.param_groups[0]['lr']
            if writer is not None:
                writer.add_scalar('Training/loss', loss_meter.val, epoch * (data_loader.__len__()) + idx)
                writer.add_scalar('Training/loss_avg', loss_meter.avg, epoch * (data_loader.__len__()) + idx)

                writer.add_scalar('Training/loss_refer', loss_refer_meter.val, epoch * (data_loader.__len__()) + idx)
                writer.add_scalar('Training/loss_refer_avg', loss_refer_meter.avg, epoch * (data_loader.__len__()) + idx)

                writer.add_scalar('Training/lr', lr, epoch * (data_loader.__len__()) + idx)

                writer.add_scalar('Training/posloss',loss_pos_meter.avg, epoch * (data_loader.__len__()) + idx)
                writer.add_scalar('Training/negloss', loss_neg_meter.avg, epoch * (data_loader.__len__()) + idx)

                writer.add_scalar('Training/regular', loss_regular_meter.avg, epoch * (data_loader.__len__()) + idx)

        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'loss_refer {loss_refer_meter.val:.4f} ({loss_refer_meter.avg:.4f})\t'
                f'loss_regular {loss_regular_meter.val:.4f} ({loss_regular_meter.avg:.4f})\t'

                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    lr_scheduler.step_update((epoch * num_steps + num_steps) // config.TRAIN.ACCUMULATION_STEPS)
        
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    diff = loss_neg_meter.avg/loss_pos_meter.avg
    if writer is not None:
        writer.add_scalar('Training/diff', diff, epoch)

    return diff


@torch.no_grad()
def validate(config, data_loader, model,epoch=0, writer=None,evaljson=None,gtjson=None,diff_eval=0.0):
    
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    model.module.sample_expression_num = config.val_sample_expression_num
    beta = 1.5
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    sumTP = AverageMeter()
    sumFP = AverageMeter()
    sumFN = AverageMeter()
    sumgtFN = AverageMeter()

    if dist.get_rank()==0:
        if diff_eval != 0:
            sumgtFN.update(diff_eval/(config.val_set_scale*config.val_sample_frame_stride))

    TP = {}
    FN = {}
    FP = {}
    gtFN = {}

    end = time.time()
    for idx, (images,pes,bbox_gt, target,expid,expma) in enumerate(data_loader):

        images = images.cuda(non_blocking=True)
        pes = pes.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        expid= expid.cuda(non_blocking=True)
        expma = expma.cuda(non_blocking=True)

        bbox_gt = bbox_gt.cuda(non_blocking=True)

        output,bbox_offset = model(images,pes,bbox_gt,expid,expma)


        if len(output.shape)==4:
            
            output= output[:,-1,:,:]

        f_,n_,c_ = output.shape

        loss = criterion(output.flatten(0,1), target.flatten(0,1))

        acc1, acc5 = accuracy(output.flatten(0,1), target.flatten(0,1), topk=(1, 5))
        # print(((output.argmax(-1)==0) * (target)).sum())
        sumgtFN.update(((output.argmax(-1)==0) * (target)).sum())

        sumTP.update(((output.argmax(-1)>0) * (target)).sum())
        sumFP.update(((output.argmax(-1)>0) * (target == 0)).sum())
        sumFN.update(((output.argmax(-1)==0) * (target)).sum())

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
        
            reducesumTP = reduce_tensor_sum(sumTP.sum)#*dist.get_world_size()
            reducesumFP = reduce_tensor_sum(sumFP.sum)#*dist.get_world_size()
            reducesumFN = reduce_tensor_sum(sumFN.sum)#*dist.get_world_size()
            reducesumgtFN = reduce_tensor_sum(sumgtFN.sum)#*dist.get_world_size()

            PRECISION = reducesumTP / (reducesumTP + reducesumFP) * 100
            RECALL =reducesumTP / (reducesumTP + reducesumFN) * 100
            gtRECALL = reducesumTP / (reducesumTP + reducesumgtFN) * 100
            F1 = (2*PRECISION*RECALL) / (PRECISION+RECALL)# * 100
            gtF1 = (2*PRECISION*gtRECALL) / (PRECISION+gtRECALL)

            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Pr {PRECISION:.3f}\t'
                f'Re {RECALL:.3f} \t'
                f'gtRe {gtRECALL:.3f} \t'

                f'gtF1 {gtF1:.3f}\t'
                f'F1 {F1:.3f}\t'



                f'Mem {memory_used:.0f}MB')
            if writer is not None:
                writer.add_scalar('val/loss', loss_meter.val, epoch * (data_loader.__len__()) + idx)

    reducesumTP = reduce_tensor_sum(sumTP.sum)#*dist.get_world_size()
    reducesumFP = reduce_tensor_sum(sumFP.sum)#*dist.get_world_size()
    reducesumFN = reduce_tensor_sum(sumFN.sum)#*dist.get_world_size()
    reducesumgtFN = reduce_tensor_sum(sumgtFN.sum)#*dist.get_world_size()

    PRECISION = reducesumTP / (reducesumTP + reducesumFP) * 100
    RECALL =reducesumTP / (reducesumTP + reducesumFN) * 100
    gtRECALL = reducesumTP / (reducesumTP + reducesumgtFN) * 100

    F1 = (2*PRECISION*RECALL) / (PRECISION+RECALL)# * 100
    gtF1 = (2*PRECISION*gtRECALL) / (PRECISION+gtRECALL)
    

    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f} Pr {PRECISION:.3f} Re {RECALL:.3f} gtRE {gtRECALL:.3f} gtF1 {gtF1:.3f} F1 {F1:.3f}')
    if writer is not None:
        writer.add_scalar('val/PR', PRECISION, epoch)
        writer.add_scalar('val/RE', RECALL, epoch)
        writer.add_scalar('val/gtRE', gtRECALL, epoch)

        writer.add_scalar('val/gtF1', gtF1, epoch)
        writer.add_scalar('val/F1', F1, epoch)
        writer.add_scalar('val/loss_avg', loss_meter.avg, epoch)



    return gtF1,F1,gtRECALL,RECALL,PRECISION
def multi_dim_dict(n, types):
    if n == 0:
        return types()
    else:
        return defaultdict(lambda: multi_dim_dict(n-1, types))
def inference(config, data_loader, model):

    model.eval()
    res_dict=multi_dim_dict(4, list)
    batch_time = AverageMeter()

    end = time.time()
    for idx, (images,pes,bbox_gt,exp,expid,expma,video,obj,frame_id,last) in enumerate(data_loader):

        images = images.cuda(non_blocking=True)
        pes = pes.cuda(non_blocking=True)

        bbox_gt = bbox_gt.cuda(non_blocking=True)


        output,bbox_offset = model(images,pes,bbox_gt,expid,expma)
        
        if len(output.shape)==4: #btn2
            
            out = output[:,-1,:,:].detach().cpu().numpy()
        else:
            out = output.detach().cpu().numpy()


        for b in range(out.shape[0]):
            for l in range(out.shape[1]):
                if l < (config.sample_expression_num-last[b]):
                    continue
                
                res_dict[video[b]][obj[b]][frame_id[b]][exp[l][b]]=out[b,l].tolist()


        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:

            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            # memory_used = 0
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'

                f'Mem {memory_used:.0f}MB')
    
    logger.info(f'finished')

    return res_dict


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()


    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    # torch.autograd.set_detect_anomaly(True)
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    #cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(True)
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR# * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR# * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR# * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.VISUAL}+{config.TEXT}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
