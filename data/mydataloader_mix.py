import os
import json
import random
import numpy as np
from PIL import Image
from os.path import join
from numpy.random import choice
from collections import defaultdict
import wordninja
import warnings
warnings.filterwarnings('ignore')

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaTokenizerFast, BertTokenizerFast
from .utils import *
# from opts import opt
import CLIP_my as clip

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


def filter_gt_expressions(gt_expressions, KEY=None):
    OUT_EXPS = list()
    for gt_exp in gt_expressions:
        if KEY is None:
            OUT_EXPS.append(gt_exp)
        else:
            for key in WORDS[KEY]:
                if key in gt_exp:
                    OUT_EXPS.append(gt_exp)
                    break
    return OUT_EXPS


class RMOT_Dataset_mix(Dataset):
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

        if self.mode=='val' or self.mode=='test':
            self.data_root = opt.val_data_root
        else:
            self.data_root = opt.data_root

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
        #self.val_sample_frame_stride = opt.val_sample_frame_stride

        if self.mode=='val': 
            # Sample frames without interval during verification to ensure consistency
            self.sample_frame_len = self.sample_frame_num 
            self.sample_frame_stride = opt.val_sample_frame_stride
            self.sample_expression_num = opt.val_sample_expression_num

        if not opt.noise1 :
            assert self.sample_frame_len == self.sample_frame_num
        if self.mode != 'test':
            if self.mode == 'train':
                datasets = self.opt.DATA.TRAIN
            else:
                datasets = self.opt.DATA.TEST

            if self.opt.DATA.DATASET =='mix':
                #assert self.mode == 'train'
                assert isinstance(datasets,list)
                assert isinstance(self.sample_frame_stride,list)
                assert isinstance(self.val_set_scale,list)

                self.data = {}
                self.sample_dict={}
                self.last={}
                self.data_keys = []
                self.video_resolution={}
                self.dslen = {}
                self.target_expressions={}

                for i,ds in enumerate(datasets):
                    self.video_resolution[ds] = json.load(open(join(self.data_root,'sequences',self.mode, ds+'_RESOLUTION.json')))
                    self.data[ds],self.sample_dict[ds],self.last[ds], self.target_expressions[ds]=self._parse_data(ds,self.video_resolution[ds],self.sample_frame_stride[i],self.val_set_scale[i])
                    self.data_keys+=list(self.sample_dict[ds].keys())
                    self.dslen[ds] = len(self.data_keys)
                
            else:
                assert isinstance(datasets,str)
                self.video_resolution = json.load(open(join(self.data_root,'sequences',self.mode, datasets+'_RESOLUTION.json')))
                self.data,self.sample_dict,self.last,self.target_expressions = self._parse_data(datasets,self.video_resolution,self.opt.sample_frame_stride,self.val_set_scale)
                self.data_keys = list(self.sample_dict.keys())
                self.dslen = None

        #elif self.mode=='val':

        else:
            assert isinstance(self.opt.DATA.TEST,str)
            # if self.opt.DATA.TEST.startswith('TAO'):
            #     self.video_resolution = json.load(open(join(self.data_root,'sequences','val', 'TAO_RESOLUTION.json')))
            # else:
            self.video_resolution = json.load(open(join(self.data_root,'sequences','val', self.opt.DATA.TEST+'_RESOLUTION.json')))

            '''parse data'''
            self.data,self.sample_dict,self.last, self.target_expressions = self._parse_data(self.opt.DATA.TEST,self.video_resolution,1,1)

            '''
            data_key: [video_obj_num] if mode == 'train'
            data_key: [video_obj_num_exp1+exp2+exp3+...+expN] if mode == 'test'
            '''
            self.data_keys = list(self.sample_dict.keys())
            self.dslen = None

        if self.mode == 'train':
            self.tao_frame2file = json.load(open(join(self.data_root,'sequences/train/TAO_frame2file.json'),'r'))
        else:
            self.tao_frame2file = json.load(open(join(self.data_root,'sequences/val/TAO_frame2file.json'),'r'))


        # print(self.target_expressions)
    def set_epoch(self,epoch):
        
        if self.mode=='train' and epoch>0:
            '''
            Parse data again at each epoch to shuffle the sampling frames in each trajectory
            Note: ''shuffle'' in the dataloader can only shuffle the order between trajectories
            '''
            if self.mode == 'train':
                datasets = self.opt.DATA.TRAIN
            else:
                datasets = self.opt.DATA.TEST

            if self.opt.DATA.DATASET =='mix':
                #assert self.mode == 'train'
                assert isinstance(datasets,list)
                assert isinstance(self.sample_frame_stride,list)
                assert isinstance(self.val_set_scale,list)

                self.data = {}
                self.sample_dict={}
                self.last={}
                self.data_keys = []
                # self.video_resolution={}
                self.dslen = {}
                self.target_expressions={}

                for i,ds in enumerate(datasets):
                    
                    self.data[ds],self.sample_dict[ds],self.last[ds], self.target_expressions[ds]=self._parse_data(ds,self.video_resolution[ds],self.sample_frame_stride[i],self.val_set_scale[i])
                    self.data_keys+=list(self.sample_dict[ds].keys())
                    self.dslen[ds] = len(self.data_keys)
                
            else:
                assert isinstance(datasets,str)
                
                self.data,self.sample_dict,self.last,self.target_expressions = self._parse_data(datasets,self.video_resolution,self.opt.sample_frame_stride,self.val_set_cale)
                self.data_keys = list(self.sample_dict.keys())
                self.dslen = None
               


    def _parse_data(self, cur_dataset, video_resolution, sample_frame_stride,val_set_scale):
        '''Annotate this paragraph to change the dataset used for validation'''
        
        if self.mode == 'train':
            labels = json.load(open(join(self.data_root, 'sequences', 'train',cur_dataset+'_labels.json')))
        else:
            # if cur_dataset.startswith('TAO'):
            #     labels = json.load(open(join(self.data_root,'sequences','val','TAO_labels.json')))

            labels = json.load(open(join(self.data_root,'sequences','val', cur_dataset+'_labels.json')))

        data = multi_dim_dict(2, list)
        target_expressions = defaultdict(list)
        if self.mode == 'train':
            expression_dir = join(self.data_root,'annotations_v1','train', cur_dataset)
        else:
            expression_dir = join(self.data_root,'annotations_v1','val', cur_dataset)
        exp_last = {}
        

        # if self.opt.split_num == -1:
        #     videolist = labels.keys()
        # else:
        #     videolist = labels.keys()[5*(self.opt.split_num-1):5*self.opt.split_num]
        if cur_dataset=='TAO':
            explist = [i+'/'+j for i in os.listdir(expression_dir) for j in os.listdir(os.path.join(expression_dir,i))]
        else:
            explist = os.listdir(expression_dir)
            
        for video in labels.keys():
            # if cur_dataset.startswith("TAO"):

            '''Load expressions of each video'''
            for exp_file in explist:

                if not exp_file.replace('_'+exp_file.split('_')[-1],'')==video:
                    # print(exp_file)
                    continue
                expression = json.load(open(join(expression_dir,exp_file)))['language']

                if expression not in target_expressions[video]:
                    target_expressions[video].append(expression)
            if len(target_expressions[video])==0:
                print(video)
                continue
            '''load bbox data'''
            W,H = video_resolution[video]

            '''During testing, traverse the tracker results and load them into bbox'''
            if self.mode=='test':

                track = np.loadtxt(os.path.join(self.track_root,f'{video}.txt'),delimiter=',')

                obj_ids = np.unique(track[:,1])
                
                for obj in obj_ids:

                    obj_key = f'{video}#{int(obj)}'
                    cur_track = track[track[:,1]==obj]
                    curr_data = defaultdict(list)
                    for frame in cur_track:
                        
                        
                        frame_id = int(frame[0]) ##################

                        bbox = frame[2:6]
                        
                        bbox[2] += bbox[0]  
                        bbox[3] += bbox[1]  
                        
                        # bbox[0] = max(0, bbox[0])
                        # bbox[1] = max(0, bbox[1])
                        # bbox[2] = max(0, bbox[2])
                        # bbox[3] = max(0, bbox[3])
                        # bbox[0] = min(1, bbox[0])
                        # bbox[1] = min(1, bbox[1])
                        # bbox[2] = min(1, bbox[2])
                        # bbox[3] = min(1, bbox[3])
                        curr_data['bbox'].append([frame_id, bbox[0],bbox[1],bbox[2],bbox[3]])
                        curr_data['bbox_norm'].append([frame_id, bbox[0]/W,bbox[1]/H,bbox[2]/W,bbox[3]/H])
                        curr_data['class'].append(0)

                    data[obj_key] = curr_data.copy()


                '''During training, Read bbox data directly from Refer-KITTI-2_1abels.json'''
            else:    
                for obj_id, obj_label in labels[video].items():

                    num = len(obj_label.keys())

                    '''Abandoning targets with too short survival time'''
                    if num < self.sample_frame_num:
                        continue

                    obj_key = f'{video}#{obj_id}'
                    pre_frame_id = -1
                    curr_data = defaultdict(list)
                    for frame_id, frame_label in obj_label.items():
                        # check that the `frame_id` is in order
                        frame_id = int(frame_id)
                        assert frame_id > pre_frame_id
                        pre_frame_id = frame_id

                        '''Abandoning targets lacking expressions'''
                        if self.exp_key not in frame_label.keys():
                            exps = []
                        else:
                            '''load exps'''
                            exps = frame_label[self.exp_key]

                        # if len(exps) == 0:
                        #     continue

                        '''load box'''
                        x, y, w, h = frame_label['bbox']
                        bbox = frame_label['bbox']

                        #bbox[2:] += bbox[0:2]  #  x1,y1,w,h -> x1,y1,x2,y2 ->
                        bbox[2] += bbox[0]  
                        bbox[3] += bbox[1]  
                        #print(bbox)
                        #assert 1==2
                        bbox[0] = max(0, bbox[0])
                        bbox[1] = max(0, bbox[1])
                        bbox[2] = max(0, bbox[2])
                        bbox[3] = max(0, bbox[3])
                        bbox[0] = min(1, bbox[0])
                        bbox[1] = min(1, bbox[1])
                        bbox[2] = min(1, bbox[2])
                        bbox[3] = min(1, bbox[3])
                        
                        '''save'''
                        curr_data['expression'].append(exps)

                        curr_data['bbox'].append([frame_id, bbox[0]*W, bbox[1]*H, bbox[2]*W, bbox[3]*H])
                        curr_data['bbox_norm'].append([frame_id, bbox[0], bbox[1], bbox[2], bbox[3]])
                    # print(curr_data)
                    if len(curr_data['bbox']) >= self.sample_frame_num:
                        data[obj_key] = curr_data.copy()
        #print(data)
        '''sample frames and load expressions'''
        sample_dict={}
        last_dict={}
        sample_node={}

        for vid_oid in data.keys():

            '''sample frames'''
            obj_last = len(data[vid_oid]['bbox']) #Target survival time

            num=0 # Distinguish different sampling frames of the same target
            iters=-1 # Reduce the size of the validation set by sampling at intervals during validation
                
            '''Total sampling range: Target survival time + self.sample_frame_num'''
            sample_start = -self.sample_frame_num+1
            sample_end = obj_last-self.sample_frame_num

            

            '''Iterate the starting frame'''
            for start_idx in range(sample_start,sample_end,sample_frame_stride):

                iters+=1

                '''
                Randomly select ''self.sample_frame_num'' frames from ''self.sample_frame_len'' frames 
                to increase the discontinuity of training data and enhance the robustness of the model
                '''
                stop_idx = start_idx+self.sample_frame_len

                '''Fixed step size of 1 and no interval sampling during testing'''
                if self.mode=='test':
                    assert sample_frame_stride ==1
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
                    # if cur_dataset=='MOT17':
                    #     if iters%1>0:
                    #         continue
                    # elif cur_dataset=='SportsMOT':
                    #     if iters%30>0:
                    #         continue
                    # elif cur_dataset == 'VisDrone':
                    #     if iters%1>0:
                    #         continue
                    if iters%val_set_scale>0:
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
                    
                    vid = vid_oid.split('#')[0]

                    '''
                    Directly indicate the current sequence in data_key
                    i.e. data_key: [video_obj_num_exp1+exp2+exp3+...+expN] if mode == 'test'
                    '''
                    for i in range(0,len(target_expressions[vid]),self.sample_expression_num):
                        if len(target_expressions[vid]) < self.sample_expression_num:
                            overlap = [target_expressions[vid][0] for ee in range(self.sample_expression_num-len(target_expressions[vid]))]
                            tmp = '+'.join(overlap+target_expressions[vid])
                            last = (len(target_expressions[vid]))
                        else:
                            if (len(target_expressions[vid])-i) < self.sample_expression_num:
                                tmp = '+'.join(target_expressions[vid][len(target_expressions[vid])-self.sample_expression_num:])
                                last=(len(target_expressions[vid])-i)
                            else:
                                tmp = '+'.join(target_expressions[vid][i:i+self.sample_expression_num])
                                last = self.sample_expression_num
                        
                        sample_dict[vid_oid+f'#{num}'+f'#{tmp}']=sample_indices

                        last_dict[vid_oid+f'#{num}'+f'#{tmp}']=last

                else:
                    '''randomly sampling expressions in get_item() during training'''
                    sample_dict[vid_oid+f'#{num}']=sample_indices
                    last_dict = None

                num+=1

        return data,sample_dict,last_dict,target_expressions

    def __getitem__(self, index):
        '''load video and obj id'''
        data_key = self.data_keys[index]
        video = data_key.split('#')[0]
        obj = data_key.split('#')[1]
        # print(self.dslen)
        if self.dslen:
            dsnumlist = [0]+list(self.dslen.values())
            for i,dsnum in enumerate(dsnumlist):
                if index >= dsnum and index < dsnumlist[i+1]:
                    cur_ds = list(self.dslen.keys())[i]
                    break
            '''load obj apperence data'''
            data = self.data[cur_ds][f'{video}#{obj}'].copy()

            '''load frame indices and images'''
            sampled_indices = self.sample_dict[cur_ds][data_key]

            target_expressions = self.target_expressions[cur_ds]

        else:
            if self.mode =='train':
                cur_ds = self.opt.DATA.TRAIN
            else:
                cur_ds = self.opt.DATA.TEST
            '''load obj apperence data'''
            data = self.data[f'{video}#{obj}'].copy()

            '''load frame indices and images'''
            sampled_indices = self.sample_dict[data_key]
            
            target_expressions = self.target_expressions

        '''111'''
        splitfix = self.mode if self.mode == 'train' else 'val'

        if cur_ds == 'VisDrone':
            img_dir = join(self.data_root,'sequences',splitfix,f'{cur_ds}-sequences',video)
            images = [Image.open(join(img_dir,'{:0>7d}.jpg'.format(int(data['bbox'][idx][0])))) for idx in sampled_indices]
        elif cur_ds == 'TAO':
            img_dir = join(self.data_root,'sequences',splitfix,f'{cur_ds}-sequences')
            images = [Image.open(join(img_dir,self.tao_frame2file[str(data['bbox'][idx][0])])) for idx in sampled_indices]
        else:
            img_dir = join(self.data_root,'sequences',splitfix,f'{cur_ds}-sequences',video,'img1')
            images = [Image.open(join(img_dir,'{:0>6d}.jpg'.format(int(data['bbox'][idx][0])))) for idx in sampled_indices]
        
         
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

                nx = max(1e-5, min(1-1e-5, nx))
                ny = max(1e-5, min(1-1e-5, ny))

                nw = max(1e-5, min(2*min(abs(1-nx),nx), nw))
                nh = max(1e-5, min(2*min(abs(1-ny),ny), nh))

                bbox = [nx-nw/2, ny-nh/2, nx+nw/2, ny+nh/2]

                # bbox[0] = max(0, bbox[0])
                # bbox[1] = max(0, bbox[1])
                # bbox[2] = max(0, bbox[2])
                # bbox[3] = max(0, bbox[3])
                # bbox[0] = min(1, bbox[0])
                # bbox[1] = min(1, bbox[1])
                # bbox[2] = min(1, bbox[2])
                # bbox[3] = min(1, bbox[3])

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
                sampled_target_exp = random.sample(target_expressions[video],self.sample_expression_num)

            elif sample_alpha>=self.sample_expression_num:
                if len(data['expression'][sampled_indices[-1]])<self.sample_expression_num:
                    sampled_target_exp = random.choices(data['expression'][sampled_indices[-1]],k=self.sample_expression_num)
                else:
                    sampled_target_exp = random.sample(data['expression'][sampled_indices[-1]],self.sample_expression_num)

            else:
                neg_exps = [ i for i in target_expressions[video] if i not in data['expression'][sampled_indices[-1]]]

                if len(neg_exps)<=0:
                    neg_exps = target_expressions[video]
                # if len(neg_exps)==0:
                #     print(target_expressions[video])
                #     print(video)
                if len(neg_exps)<(self.sample_expression_num-sample_alpha): 
                    sampled_target_exp = random.choices(neg_exps,k=self.sample_expression_num-sample_alpha)
                else:
                    sampled_target_exp = random.sample(neg_exps,self.sample_expression_num-sample_alpha)

                #print(len(data['expression'][sampled_indices[-1]]))
                if len(data['expression'][sampled_indices[-1]])==0:
                    sampled_target_exp = sampled_target_exp+random.choices(target_expressions[video],k=sample_alpha)
                elif len(data['expression'][sampled_indices[-1]])<sample_alpha:
                    sampled_target_exp = sampled_target_exp+random.choices(data['expression'][sampled_indices[-1]],k=sample_alpha)
                else:
                    sampled_target_exp = sampled_target_exp+random.sample(data['expression'][sampled_indices[-1]],sample_alpha)

                random.shuffle(sampled_target_exp)

        else:
            '''directly load expressions (val and test)'''

            sampled_target_exp = data_key.split('#')[-1].split('+')

        '''Complete tokenization directly in the dataset to improve processing efficiency1111'''
        if cur_ds == 'TAO':
            converted =[]
            for i in sampled_target_exp:
                newexp = ' '.join(wordninja.split(i))
                # newexp = i.replace('_',' ').replace('(','or ').replace(')','')
                # if 'baby' in newexp and ' baby ' not in newexp and ' baby' not in newexp and 'baby ' not in newexp:
                #     newexp = newexp.replace('baby',' baby ')
                if 'baby' in newexp:
                    newexp = newexp.replace('baby','baby (may be women)')
                    

                converted.append(newexp)
            out = self.tokenizer.batch_encode_plus(converted, padding="max_length", return_tensors='pt',max_length=self.text_len)#.to(x.device)
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
            # print(images.shape)
            # print(global_pe.shape)
            # print(label.shape)
            # print(sampled_target_ids.shape)
            # print(sampled_target_mask.shape)
            return images,global_pe,bbox_gt,label.long(),sampled_target_ids,sampled_target_mask
        
        elif self.mode == 'train':
            return images,global_pe,bbox_gt,label.long(),sampled_target_ids,sampled_target_mask,data_key,torch.tensor(sampled_indices),sampled_target_exp,index
        
        # if not self.mode == 'test':
        #     return images,global_pe,bbox_gt,label.long(),sampled_target_ids,sampled_target_mask
        else:
            '''Return the obj information during test for generation of final results'''
            return images,global_pe,bbox_gt,sampled_target_exp,sampled_target_ids,sampled_target_mask,video,str(obj),str(frame_id),self.last[data_key]#,raw_sentence
            # return sampled_target_exp,video,str(obj),str(frame_id),self.last[data_key]#,raw_sentence
            

    def __len__(self):

        print(f'{dist.get_rank()} is {len(self.data_keys)}')
        print(self.dslen)
        # if self.mode=='train':
        #     return 500
        return len(self.data_keys)

    def show_information(self):
        print(
            f'===> {self.opt.DATA.DATASET} ({self.mode}) <===\n'
            f"Number of identities: {len(self.data_keys)}"
        )




if __name__ == '__main__':
    dataset = RMOT_Dataset('train')
    print(len(dataset))