
import os
import torch
import torch.distributed as dist

try:
    from torch._six import inf
except:
    from torch import inf

from torchvision.ops.boxes import box_area

import torch.nn.functional as F

def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']

    # delete rope_t since we always re-init it
    rope_t_keys = [k for k in state_dict.keys() if "rope_t_" in k]
    for k in rope_t_keys:
        del state_dict[k]

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                print('bicubic bicubic bicubic bicubic bicubic bicubic ')

                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                print('bicubic bicubic bicubic bicubic bicubic bicubic ')

                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    if not config.EVAL_MODE:
        #print(state_dict.keys())
        # if 'head.0.fc2.weight' not in state_dict.keys():
        #     head_bias_pretrained = state_dict['head.0.weight']
        # else:
        #     head_bias_pretrained = state_dict['head.0.fc2.weight']
        Nc1 = 12#head_bias_pretrained.shape[0]
        Nc2 = 2#model.head.fc2.bias.shape[0]
        if (Nc1 != Nc2):
            if Nc1 == 21841 and Nc2 == 1000:
                logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
                map22kto1k_path = f'data/map22kto1k.txt'
                with open(map22kto1k_path) as f:
                    map22kto1k = f.readlines()
                map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
                state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
                state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
            else:
                # torch.nn.init.constant_(model.head.fc1.bias, 0.)
                # torch.nn.init.constant_(model.head.fc1.weight, 0.)
                # torch.nn.init.constant_(model.head.fc2.bias, 0.)
                # torch.nn.init.constant_(model.head.fc2.weight, 0.)
                # del state_dict['head.weight']
                # del state_dict['head.bias']
                logger.warning(f"Error in loading classifier head, re-init classifier head to 0")
        ################
        # if state_dict['patch_embed.proj.weight'].shape[1] is not model.patch_embed.proj.weight.detach().cpu().shape[1]:
        #     state_dict['patch_embed.proj.weight'] = torch.cat((state_dict['patch_embed.proj.weight'],model.patch_embed.proj.weight[:,3:,:,:].detach().cpu()),dim=1)
        #     logger.warning(f"concat loading patch_embed")
        
        #if state_dict['patch_embed.proj.weight'].shape[1] is not model.patch_embed.proj_car.weight.detach().cpu().shape[1]:
            # state_dict['patch_embed.proj_car.weight'] = torch.cat((state_dict['patch_embed.proj.weight'],model.patch_embed.proj_car.weight[:,3:,:,:].detach().cpu()),dim=1)
            # state_dict['patch_embed.proj_ped.weight'] = torch.cat((state_dict['patch_embed.proj.weight'],model.patch_embed.proj_ped.weight[:,3:,:,:].detach().cpu()),dim=1)
            
        # state_dict['patch_embed.proj_car.weight'] = model.patch_embed.proj_car.weight#state_dict['patch_embed.proj.weight']
        # state_dict['patch_embed.proj_ped.weight'] = model.patch_embed.proj_ped.weight#state_dict['patch_embed.proj.weight']
    
        # state_dict['patch_embed.proj_car.bias'] = model.patch_embed.proj_car.bias#state_dict['patch_embed.proj.bias']
        # state_dict['patch_embed.proj_ped.bias'] = model.patch_embed.proj_ped.bias#state_dict['patch_embed.proj.bias']
        
        # state_dict['patch_embed.norm_car.weight'] = model.patch_embed.norm_car.weight#state_dict['patch_embed.norm.weight']
        # state_dict['patch_embed.norm_car.bias'] = model.patch_embed.norm_car.bias#state_dict['patch_embed.norm.bias']
        
        # state_dict['patch_embed.norm_ped.weight'] = model.patch_embed.norm_ped.weight#state_dict['patch_embed.norm.weight']
        # state_dict['patch_embed.norm_ped.bias'] = model.patch_embed.norm_ped.bias#state_dict['patch_embed.norm.bias']
######################
        
        # state_dict['patch_embed.proj_car.weight'] = state_dict['patch_embed.proj.weight']
        # state_dict['patch_embed.proj_ped.weight'] = state_dict['patch_embed.proj.weight']
        # state_dict['obj_offset_proj.0.weight'] = model.obj_offset_proj[0].weight
        # state_dict['obj_offset_proj.0.bias'] = model.obj_offset_proj[0].bias
        # state_dict['obj_offset_proj.1.weight'] = model.obj_offset_proj[1].weight
        # state_dict['obj_offset_proj.1.bias'] = model.obj_offset_proj[1].bias
        # state_dict['obj_offset_proj.2.weight'] = model.obj_offset_proj[2].weight
        # state_dict['obj_offset_proj.2.bias'] = model.obj_offset_proj[2].bias
        # state_dict['patch_embed.proj_car.bias'] = state_dict['patch_embed.proj.bias']
        # state_dict['patch_embed.proj_ped.bias'] = state_dict['patch_embed.proj.bias']
        
        # state_dict['patch_embed.norm_car.weight'] = state_dict['patch_embed.norm.weight']
        # state_dict['patch_embed.norm_car.bias'] = state_dict['patch_embed.norm.bias']
        
        # state_dict['patch_embed.norm_ped.weight'] = state_dict['patch_embed.norm.weight']
        # state_dict['patch_embed.norm_ped.bias'] = state_dict['patch_embed.norm.bias']
        # print(state_dict['patch_embed.proj.weight'].shape)
        # print(state_dict['patch_embed.proj.bias'].shape)
        # print(state_dict['patch_embed.norm.weight'].shape)
        # print(state_dict['patch_embed.norm.bias'].shape)

        # assert 1==2
        # state_dict['patch_embed.proj_global.weight'] = torch.nn.functional.interpolate(state_dict['patch_embed.proj.weight'],(8,8),mode='bilinear')#model.patch_embed.proj_global.weight
        # state_dict['patch_embed.proj_global.bias'] = state_dict['patch_embed.proj.bias']#model.patch_embed.proj_global.bias
        # state_dict['patch_embed.norm_global.weight'] = state_dict['patch_embed.norm.weight']#model.patch_embed.norm_global.weight
        # state_dict['patch_embed.norm_global.bias'] =state_dict['patch_embed.norm.bias']# model.patch_embed.norm_global.bias

        # state_dict['patch_embed.proj_global.weight'] = model.patch_embed.proj_global.weight
        # state_dict['patch_embed.proj_global.bias'] = model.patch_embed.proj_global.bias
        # state_dict['patch_embed.norm_global.weight'] = model.patch_embed.norm_global.weight
        # state_dict['patch_embed.norm_global.bias'] = model.patch_embed.norm_global.bias
            # logger.warning(f"concat loading patch_embed")
        ################
        # state_dict['absolute_lang_pos_embed']=model.absolute_lang_pos_embed
        # state_dict['head.fc1.weight']=model.head.fc1.weight
        # state_dict['head.fc1.bias']=model.head.fc1.bias
        # state_dict['head.fc2.weight']=model.head.fc2.weight
        # state_dict['head.fc2.bias']=model.head.fc2.bias

    # state_dict['head.weight']=model.head.weight
    # state_dict['head.bias']=model.head.bias


            # print(k)
            # print(v.shape)
            # if k.endswith('weight'):
            #     # print(k)
            #     # print(v.shape)
            #     state_dict[k] = torch.cat([torch.cat([v,v],dim=0),torch.cat([v,v],dim=0)],dim=1)
            # elif k.endswith('bias'):
            #     # print(k)
            #     # print(v.shape)
            #     state_dict[k] = torch.cat([v,v],dim=0)
            # else:
            #     state_dict[k] = torch.cat([v,v],dim=1)

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()

def load_ckpt_my(config, model, logger):

    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for test......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu',weights_only=False)
    state_dict = checkpoint['model']
    
    # check classifier, if not match, then re-init classifier to zero
    

    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_lang_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        if k not in model.state_dict().keys():
            continue
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                print('bicubic bicubic bicubic bicubic bicubic bicubic ')

                #S1 = int(L1)
                #S2 = int(L2)
                # absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, L1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 2,1)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=L2, mode='linear',)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2,1)
                # absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized
    new_state_dict = None
    
    if hasattr(model,'backbone'):
        new_state_dict = {}
        for k,v in model.state_dict().items():
            if k.startswith('backbone.'):
                if k not in state_dict.keys():
                    if k.replace('backbone.','') in state_dict.keys():
                        new_state_dict[k] = state_dict[k.replace('backbone.','')]
                    else:
                        assert 1==2
        for k,v in state_dict.items():
            if 'backbone.'+k not in new_state_dict.keys():
                    new_state_dict[k]=state_dict[k]
        logger.info(f"=> done backbone transfer")
    if new_state_dict is not None:

        msg = model.load_state_dict(new_state_dict, strict=False)
    else:
        msg = model.load_state_dict(state_dict, strict=False)

    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()

def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger,F1,bestnumber=None):
    save_state = {'model': model.state_dict(),
                  #'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'F1':F1,
                  #'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}

    if bestnumber is not None:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_best_{bestnumber}.pth')

        for i in range(bestnumber,4)[::-1]:
            existed_path = os.path.join(config.OUTPUT, f'ckpt_epoch_best_{i}.pth')
            existed_path_new = os.path.join(config.OUTPUT, f'ckpt_epoch_best_{i+1}.pth')
            if os.path.exists(existed_path):
                logger.info(f'mv {existed_path} {existed_path_new}')
                os.system(f'mv {existed_path} {existed_path_new}')
                
    
    else:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= torch.tensor(dist.get_world_size())
    return rt

def reduce_tensor_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    #rt /= torch.tensor(dist.get_world_size())
    return rt

def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler('cuda')

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        # self._scaler.scale(loss).backward(create_graph=create_graph)
        # self._scaler.step(optimizer)
        # self._scaler.update()
        # if update_grad:
        #     if clip_grad is not None:
        #         assert parameters is not None
        #         self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
        #         norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        #     else:
        #         self._scaler.unscale_(optimizer)
        #         norm = ampscaler_get_grad_norm(parameters)
        #     self._scaler.step(optimizer)
        #     self._scaler.update()
        # else:
            
        #     norm = None
        # return norm
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, weight=None):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.weight = weight

#     def forward(self, inputs, targets):
#         ce_loss = F.cross_entropy(inputs, targets,weight=self.weight)  # 使用交叉熵损失函数计算基础损失
#         pt = torch.exp(-ce_loss)  # 计算预测的概率
#         focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
#         return focal_loss




def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
