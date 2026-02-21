import os
import json
import random
import numpy as np
from PIL import Image
from os.path import join
from numpy.random import choice
from collections import defaultdict
import CLIP_my as clip
import warnings
warnings.filterwarnings('ignore')

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaTokenizerFast, BertTokenizerFast
from .utils import *
# from opts import opt


def get_dataloader(mode, opt, dataset='RMOT_Dataset', show=False, **kwargs):
    dataset = eval(dataset)(mode, opt, **kwargs)
    if show:
        dataset.show_information()
    if mode == 'train':
        dataloader = DataLoader(
            dataset,
            batch_size=opt.train_bs,
            shuffle=True,
            drop_last=True,
            num_workers=opt.num_workers,
        )
    elif mode == 'test':
        dataloader = DataLoader(
            dataset,
            batch_size=opt.test_bs,
            shuffle=False,
            drop_last=False,
            num_workers=opt.num_workers,
        )
    return dataloader


def get_transform(mode, opt):

    if mode == 'train':
        return T.Compose([

            T.Resize(opt.img_hw),
            T.ToTensor(),
            T.Normalize(opt.norm_mean, opt.norm_std),
        ])
    elif mode == 'val':
        return T.Compose([

            T.Resize(opt.img_hw),
            T.ToTensor(),
            T.Normalize(opt.norm_mean, opt.norm_std),
        ])
    elif mode == 'test':
        return T.Compose([

            T.Resize(opt.img_hw),
            T.ToTensor(),
            T.Normalize(opt.norm_mean, opt.norm_std),
        ])
    elif mode == 'unnorm':
        mean = opt.norm_mean
        std = opt.norm_std
        return T.Normalize(
            [-mean[i]/std[i] for i in range(3)],
            [1/std[i] for i in range(3)],
        )
    
def filter_target_expressions(gt, target_expressions, exp_key, only_car):
    """
    给定“帧级标签”和“视频级exp"，得到帧级exps和对应labels
    """
    OUT_EXPS, OUT_LABELS = list(), list()
    GT_EXPRESSIONS = gt[exp_key]
    for tgt_exp in target_expressions:
        if only_car and ('car' not in tgt_exp):
            continue
        OUT_EXPS.append(tgt_exp)
        if tgt_exp in GT_EXPRESSIONS:
            OUT_LABELS.append(1)
        else:
            OUT_LABELS.append(0)
    return OUT_EXPS, OUT_LABELS



class RMOT_Dataset(Dataset):
    """
    For the `car` + `color+direction+location` settings
    For the `car` + 'status' settings
    """
    def __init__(self, mode, opt, only_car=False):
        super().__init__()
        self.opt = opt

        assert mode in ('train','val', 'test')
        self.mode = mode 

        self.transform = get_transform(mode, opt) # resize, toTensor and norm

        self.exp_key = 'expression_raw'  # refer to Refer-KITTI-2_labels.json
        
        self.convert = opt.convert
        if opt.TEXT == 'roberta':
            self.tokenizer = RobertaTokenizerFast.from_pretrained(opt.ROBERTA_PATH)
        elif opt.TEXT == 'bert':
            self.tokenizer = BertTokenizerFast.from_pretrained(opt.BERT_PATH)
        elif opt.TEXT == 'clip':
            self.tokenizer = clip.tokenize

        self.text_len = opt.text_len # 25 for roberta and bert, 77 for clip (For refer-kitti, the minimum is 12)

        self.noise2 = opt.noise2 # Adding Gaussian noise to the detection box to enhance robustness
        self.data_root = opt.data_root
        if self.opt.DATA.DATASET == 'dance':
            self.img_root = join(self.data_root, 'DanceTrack/training/image_02')
        else:
            self.img_root = join(self.data_root, 'KITTI/training/image_02')
        self.track_root = opt.track_root # for test

        '''para for sample frame'''
        self.sample_frame_num=opt.sample_frame_num
        self.sample_frame_len=opt.sample_frame_len
        self.sample_frame_stride = opt.sample_frame_stride
        '''para for sample expression'''
        self.sample_expression_num = opt.sample_expression_num
        self.sample_alpha=opt.sample_alpha

        '''Control the size of the validation set to enhance efficiency'''
        self.val_set_scale=opt.val_set_scale 
        self.val_sample_frame_stride = opt.val_sample_frame_stride
        if self.mode=='val': 
            # Sample frames without interval during verification to ensure consistency
            self.sample_frame_len = self.sample_frame_num 
            self.sample_frame_stride = self.val_sample_frame_stride
            self.sample_expression_num = opt.val_sample_expression_num


        if not opt.noise1 :
            assert self.sample_frame_len == self.sample_frame_num
        
        '''parse data'''
        self.data,self.sample_dict,self.last,self.sample_node = self._parse_data()

        '''
        data_key: [video_obj_num] if mode == 'train'
        data_key: [video_obj_num_exp1+exp2+exp3+...+expN] if mode == 'test'
        '''
        self.data_keys = list(self.sample_dict.keys())

    def set_epoch(self,epoch):
        
        if self.mode=='train' and epoch>0:
            '''
            Parse data again at each epoch to shuffle the sampling frames in each trajectory
            Note: ''shuffle'' in the dataloader can only shuffle the order between trajectories
            '''
            self.data,self.sample_dict,self.last,self.sample_node = self._parse_data()
            
            self.data_keys = list(self.sample_dict.keys())

    def _parse_data(self):

        labels = json.load(open(join(self.data_root, 'labels.json')))

        if self.opt.DATA.DATASET=='kitti-1':
            newsent = json.load(open(join(self.data_root,'newsent.json'),'r'))
        data = multi_dim_dict(2, list)
        self.target_expressions = defaultdict(list)
        expression_dir = join(self.data_root, 'expression')

        videolist = VIDEOS[self.opt.DATA.DATASET][self.mode]



        for video in videolist:
            
            if self.mode=='train' and self.opt.DATA.DATASET=='kitti-1':

                if video in newsent.keys():
                    explist = os.listdir(join(expression_dir, video))+newsent[video]
                else:
                    explist = os.listdir(join(expression_dir, video))

            else:
                explist = os.listdir(join(expression_dir, video))


            '''Load expressions of each video'''
            for exp_file in explist:

                if self.mode=='train' and self.opt.DATA.DATASET=='kitti-1':

                    if  exp_file.endswith('.json'):
                        if os.path.exists(join(expression_dir, video,exp_file)):
                            expression = json.load(open(join(expression_dir, video,exp_file)))['sentence']
                    else:
                        expression = exp_file

                else:
                    expression = json.load(open(join(expression_dir, video,exp_file)))['sentence']

                if expression not in self.target_expressions[video]:
                    self.target_expressions[video].append(expression)

            '''load bbox data'''
            H, W = RESOLUTION[video]

            '''During testing, traverse the tracker results and load them into bbox'''
            if self.mode=='test':
                if self.opt.DATA.DATASET=='dance':
                    track = np.loadtxt(os.path.join(self.track_root,video+'.txt'),delimiter=',')
                    obj_ids = np.unique(track[:,1])
                    
                    for obj in obj_ids:
                        #obj = int(obj)
                        obj_key = f'{video}_{int(obj)}'
                        cur_track = track[track[:,1]==obj]
                        curr_data = defaultdict(list)
                        for frame in cur_track:
                            #frame
    
                            frame_id = int(frame[0])
    
                            bbox = frame[2:6]
                            bbox[2:] += bbox[0:2]
                            curr_data['bbox'].append([frame_id, bbox[0],bbox[1],bbox[2],bbox[3]])
                            curr_data['bbox_norm'].append([frame_id, bbox[0]/W,bbox[1]/H,bbox[2]/W,bbox[3]/H])
                            curr_data['class'].append(0)
                        data[obj_key] = curr_data.copy()
                else:
                    track_car = np.loadtxt(os.path.join(self.track_root,video,'car','predict.txt'),delimiter=',')
                    track_ped = np.loadtxt(os.path.join(self.track_root,video,'pedestrian','predict.txt'),delimiter=',')
    
                    obj_ids_car = np.unique(track_car[:,1])
                    
                    for obj in obj_ids_car:
                    
                        obj_key = f'{video}_{int(obj)}'
                        cur_track = track_car[track_car[:,1]==obj]
                        curr_data = defaultdict(list)
                        for frame in cur_track:
                            
                            '''The output of TempRMOT will be advanced by one frame'''
                            frame_id = int(frame[0])-1 ##################
    
                            bbox = frame[2:6]
                            bbox[2:] += bbox[0:2]
                            curr_data['bbox'].append([frame_id, bbox[0],bbox[1],bbox[2],bbox[3]])
                            curr_data['bbox_norm'].append([frame_id, bbox[0]/W,bbox[1]/H,bbox[2]/W,bbox[3]/H])
                            curr_data['class'].append(0)
    
                        data[obj_key] = curr_data.copy()
    
                    if track_ped.shape[0] > 0:
                        '''Increasing obj_id for pedestrian tracking results'''
                        obj_ids_ped = np.unique(track_ped[:,1])
    
                        for obj in obj_ids_ped:
                            obj_key = f'{video}_{int(obj+np.max(obj_ids_car))}'
                            cur_track = track_ped[track_ped[:,1]==obj]
                            curr_data = defaultdict(list)
                            for frame in cur_track:
                                
                                '''The output of TempRMOT will be advanced by one frame'''
                                frame_id = int(frame[0])-1 ##################
    
                                bbox = frame[2:6]
                                bbox[2:] += bbox[0:2]
                                curr_data['bbox'].append([frame_id, bbox[0],bbox[1],bbox[2],bbox[3]])
                                curr_data['bbox_norm'].append([frame_id, bbox[0]/W,bbox[1]/H,bbox[2]/W,bbox[3]/H])
                                curr_data['class'].append(1)
                                
                            data[obj_key] = curr_data.copy() 

                

                '''During training, Read bbox data directly from Refer-KITTI-2_1abels.json'''
            else:    
                for obj_id, obj_label in labels[video].items():

                    num = len(obj_label.keys())

                    '''Abandoning targets with too short survival time'''
                    if num < self.sample_frame_num:
                        continue

                    obj_key = f'{video}_{obj_id}'
                    pre_frame_id = -1
                    curr_data = defaultdict(list)
                    for frame_id, frame_label in obj_label.items():
                        # check that the `frame_id` is in order
                        frame_id = int(frame_id)
                        assert frame_id > pre_frame_id
                        pre_frame_id = frame_id

                        '''Abandoning targets lacking expressions'''
                        if self.opt.DATA.DATASET=='kitti-1':
                            if self.exp_key not in frame_label.keys():
                                exps = []
                            else:
                                exps = frame_label[self.exp_key]
                        else:
                            if self.exp_key not in frame_label.keys():
                                continue
                            #continue

                            '''load exps'''
                            exps = frame_label[self.exp_key]

                            if len(exps) == 0:
                                continue

                        '''load box'''
                        x, y, w, h = frame_label['bbox']
                        
                        '''save'''
                        curr_data['expression'].append(exps)

                        curr_data['bbox'].append([frame_id, x * W, y * H, (x + w) * W, (y + h) * H])
                        curr_data['bbox_norm'].append([frame_id, x, y, (x + w), (y + h)])

                    if len(curr_data['bbox']) >= self.sample_frame_num:
                        data[obj_key] = curr_data.copy()
        
        '''sample frames and load expressions'''
        sample_dict={}
        last_dict={}
        sample_node={}

        for vid_oid in data.keys():

            '''sample frames'''
            obj_last = len(data[vid_oid]['bbox']) #Target survival time

            num=0 # Distinguish different sampling frames of the same target
            iter=-1 # Reduce the size of the validation set by sampling at intervals during validation
                
            '''Total sampling range: Target survival time + self.sample_frame_num'''
            sample_start = -self.sample_frame_num+1
            sample_end = obj_last-self.sample_frame_num

            sample_node[vid_oid]={}

            '''Iterate the starting frame'''
            for start_idx in range(sample_start,sample_end,self.sample_frame_stride):

                iter+=1

                '''
                Randomly select ''self.sample_frame_num'' frames from ''self.sample_frame_len'' frames 
                to increase the discontinuity of training data and enhance the robustness of the model
                '''
                stop_idx = start_idx+self.sample_frame_len

                '''Fixed step size of 1 and no interval sampling during testing'''
                if self.mode=='test':
                    assert self.sample_frame_stride ==1
                    if start_idx < 0:
                        sample_indices= [ 0 if i < 0 else i for i in range(start_idx,stop_idx)]
                    else:
                        sample_indices = list(range(start_idx,stop_idx))
                
                else:
                    '''When sampling the tail frame of the target, it may already be less than 
                    'self.sample_frame_len' frames, but it must meet 'self.sample_frame_num' 
                    frames. At this time, reset the end frame'''
                    if stop_idx>obj_last:
                        stop_idx = obj_last
                    
                    '''When sampling the target head frame, repeat the first frame to obtain
                    'self.sample_frame_num' frames, so that the first few frames can also be
                    used as window end frames for training'''
                    if start_idx < 0:
                        sample_indices= [ 0 if i < 0 else i for i in range(start_idx,stop_idx)]
                    else:
                        sample_indices = list(range(start_idx,stop_idx))

                '''Reduce the size of the validation set by sampling at intervals during validation'''
                if self.mode=='val':
                    if iter%self.val_set_scale>0:
                        continue
                if self.mode=='train':
                    if torch.rand(1)<-1:
                        continue
                '''Randomly select ''self.sample_frame_num'' frames from ''self.sample_frame_len'' frames'''
                if self.sample_frame_num <self.sample_frame_len:

                    '''Control the beginning and end frames to increase the frame interval'''
                    sample_indices=sample_indices[0:1]+random.sample(sample_indices[1:-1],k=self.sample_frame_num-2)+sample_indices[-1:]

                    '''sort'''
                    sample_indices.sort()
                else:
                    sample_indices.sort()

                '''load all expressions for test and val'''
                if self.mode=='test' or self.mode=='val':
                    '''Load all expressions during testing or validation to ensure complete testing and consistency evaluation'''
                    
                    vid = vid_oid.split('_')[0]

                    '''
                    Directly indicate the current sequence in data_key
                    i.e. data_key: [video_obj_num_exp1+exp2+exp3+...+expN] if mode == 'test'
                    '''
                    for i in range(0,len(self.target_expressions[vid]),self.sample_expression_num):

                        if (len(self.target_expressions[vid])-i) < self.sample_expression_num:
                            tmp = '+'.join(self.target_expressions[vid][len(self.target_expressions[vid])-self.sample_expression_num:])
                            last=(len(self.target_expressions[vid])-i) #padding
                        else:
                            tmp = '+'.join(self.target_expressions[vid][i:i+self.sample_expression_num])
                            last = self.sample_expression_num
                        
                        sample_dict[vid_oid+f'_{num}'+f'_{tmp}']=sample_indices

                        last_dict[vid_oid+f'_{num}'+f'_{tmp}']=last

                else:
                    '''randomly sampling expressions in get_item() during training'''
                    sample_dict[vid_oid+f'_{num}']=sample_indices
                    last_dict = None

                num+=1

        return data,sample_dict,last_dict,sample_node

    def __getitem__(self, index):
        '''load video and obj id'''
        data_key = self.data_keys[index]

        video = data_key.split('_')[0]
        obj = data_key.split('_')[1]

        '''load obj apperence data'''
        data = self.data[f'{video}_{obj}'].copy()

        '''load frame indices and images'''
        sampled_indices = self.sample_dict[data_key]

        images = [
            Image.open(
                join(
                    self.img_root,
                    '{}/{:0>6d}.png'.format(video, int(data['bbox'][idx][0]))
                )
            ) for idx in sampled_indices
        ] 
        images = torch.stack([self.transform(i) for i in images],dim=0) # T H W 3

        '''load bbox and generate grid'''
        global_pe = [] # grid

        bbox_gt = [] # bbox sequences

        for idx,sample_ind in enumerate(sampled_indices):

            '''load bbox in each frame'''
            bbox = data['bbox_norm'][sample_ind][1:]

            cxcywh = torch.tensor(bbox,dtype=torch.get_default_dtype())
            cxcywh[2:] = cxcywh[2:]-cxcywh[:2]
            cxcywh[:2] = cxcywh[:2]+cxcywh[2:]/2
            
            '''Adding Gaussian perturbation on bbox to enhance robustness '''
            if self.noise2 and (self.mode=='train'):

                '''x1,y1,x2,y2 -> cx,cy,w,h'''
                bx, by, bx2, by2 = data['bbox_norm'][sample_ind][1:]
                bh = by2-by
                bw = bx2-bx
                cx = (bx+bx2)/2
                cy = (by+by2)/2

                '''Control the disturbance scale according to the size of bbox to simulate the output of a real tracker'''
                nw = bw+ bw*0.02*np.random.randn()
                nh = bh+ bh*0.02*np.random.randn()
                nx = cx+ bw*0.02*np.random.randn()
                ny = cy+ bh*0.02*np.random.randn()

                bbox = [nx-nw/2, ny-nh/2, nx+nw/2, ny+nh/2]

                cxcywh =torch.tensor([nx,ny,nw,nh],dtype=torch.get_default_dtype())
                
                
            '''generate grid'''
            bbox_gt.append(cxcywh)

            '''Only generate the largest grid size, and construct smaller grids through interpolation in model'''
            grid_w = (bbox[2]-bbox[0])/48
            grid_h = (bbox[3]-bbox[1])/16

            '''
            Refer to the 'align_corner' in 'grid_sample' to understand the translation of the diagonal points here

            '''
            x = torch.linspace(bbox[0]+grid_w/2,bbox[2]-grid_w/2,48)*2-1
            y = torch.linspace(bbox[1]+grid_h/2,bbox[3]-grid_h/2,16)*2-1

            x,y = torch.meshgrid((x,y),indexing='xy')

            '''grid_sample requires x to come first and y to come last'''
            global_pe.append(torch.stack([x,y]))


        bbox_gt = torch.stack(bbox_gt) #T 4
        global_pe = torch.stack(global_pe) # T 2 H W

        '''sample expressions (train)'''
        if self.mode == 'train':
            
            '''
            Randomly adjust the number of positive and negative expressions to avoid overfitting
            'sample_alpha' represents the number of positive examples
            '''
            if self.opt.gauss_sample:
                sample_alpha = self.sample_alpha+random.gauss(0,1)#*0.5
                sample_alpha = min(max(int(round(sample_alpha)),0),self.sample_expression_num)
            else:
                sample_alpha = self.sample_alpha

            '''Use sample when population is sufficient, otherwise use choice'''
            if sample_alpha == 0:
                sampled_target_exp = random.sample(self.target_expressions[video],self.sample_expression_num)

            elif sample_alpha>=self.sample_expression_num:
                if self.opt.DATA.DATASET == 'kitti-1':
                    if len(data['expression'][sampled_indices[-1]])==0:
                        if len(self.target_expressions[video])<(self.sample_expression_num): 
                            sampled_target_exp = random.choices(self.target_expressions[video],k=self.sample_expression_num)
                        else:
                            sampled_target_exp = random.sample(self.target_expressions[video],self.sample_expression_num)
                    else:
                        if len(data['expression'][sampled_indices[-1]])<self.sample_expression_num:
                            #print(11111111)
                            sampled_target_exp = random.choices(data['expression'][sampled_indices[-1]],k=self.sample_expression_num)
                        else:
                            sampled_target_exp = random.sample(data['expression'][sampled_indices[-1]],self.sample_expression_num)
                else:
                    if len(data['expression'][sampled_indices[-1]])<self.sample_expression_num:
                        sampled_target_exp = random.choices(data['expression'][sampled_indices[-1]],k=self.sample_expression_num)
                    else:
                        sampled_target_exp = random.sample(data['expression'][sampled_indices[-1]],self.sample_expression_num)

            else:
                neg_exps = [ i for i in self.target_expressions[video] if i not in data['expression'][sampled_indices[-1]]]

                if len(neg_exps)<=0:
                    neg_exps = self.target_expressions[video]
                if len(neg_exps)<(self.sample_expression_num-sample_alpha): 
                    sampled_target_exp = random.choices(neg_exps,k=self.sample_expression_num-sample_alpha)
                else:
                    sampled_target_exp = random.sample(neg_exps,self.sample_expression_num-sample_alpha)

                if self.opt.DATA.DATASET=='kitti-1':
                    if len(data['expression'][sampled_indices[-1]])==0:
                        if len(neg_exps)<(sample_alpha): 
                            sampled_target_exp = sampled_target_exp+random.choices(neg_exps,k=sample_alpha)
                        else:
                            sampled_target_exp = sampled_target_exp+random.sample(neg_exps,sample_alpha)
                    elif len(data['expression'][sampled_indices[-1]])<sample_alpha:
                        sampled_target_exp = sampled_target_exp+random.choices(data['expression'][sampled_indices[-1]],k=sample_alpha)
                    else:
                        sampled_target_exp = sampled_target_exp+random.sample(data['expression'][sampled_indices[-1]],sample_alpha)
                else:
                    if len(data['expression'][sampled_indices[-1]])<sample_alpha:
                        sampled_target_exp = sampled_target_exp+random.choices(data['expression'][sampled_indices[-1]],k=sample_alpha)
                    else:
                        sampled_target_exp = sampled_target_exp+random.sample(data['expression'][sampled_indices[-1]],sample_alpha)
                random.shuffle(sampled_target_exp)

        else:
            '''directly load expressions (val and test)'''

            sampled_target_exp = data_key.split('_')[-1].split('+')

        '''Complete tokenization directly in the dataset to improve processing efficiency'''
        if self.opt.TEXT == 'clip':
            # print()
            out,expma = self.tokenizer(sampled_target_exp)
            sampled_target_ids = out
            sampled_target_mask = expma
        else:
            out = self.tokenizer.batch_encode_plus(sampled_target_exp, padding="max_length", return_tensors='pt',max_length=self.text_len)#.to(x.device)

            sampled_target_ids = out['input_ids']
            sampled_target_mask = out['attention_mask']

        '''load gt label for each expression during val and train'''
        if not self.mode == 'test':
            label = torch.tensor([1 if i in data['expression'][sampled_indices[-1]] else 0 for i in sampled_target_exp] )

        else:
            '''Record the frame ID during test for generation of final results'''
            frame_id = data['bbox'][sampled_indices[-1]][0]

        if self.mode == 'val':
            return images,global_pe,bbox_gt,label.long(),sampled_target_ids,sampled_target_mask
        
        elif self.mode == 'train':
            return images,global_pe,bbox_gt,label.long(),sampled_target_ids,sampled_target_mask,data_key,torch.tensor(sampled_indices),sampled_target_exp,index
        
        # if not self.mode == 'test':
        #     return images,global_pe,bbox_gt,label.long(),sampled_target_ids,sampled_target_mask
        else:
            '''Return the obj information during test for generation of final results'''
            return images,global_pe,bbox_gt,sampled_target_exp,sampled_target_ids,sampled_target_mask,video,str(obj),str(frame_id),self.last[data_key]#,raw_sentence
            

    def __len__(self):

        print(f'{dist.get_rank()} is {len(self.data_keys)}')

        return len(self.data_keys)

    def show_information(self):
        print(
            f'===> Refer-KITTI ({self.mode}) <===\n'
            f"Number of identities: {len(self.data)}"
        )




if __name__ == '__main__':
    dataset = RMOT_Dataset('train')
    print(len(dataset))