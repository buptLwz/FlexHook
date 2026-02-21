import os
import json
import shutil
import numpy as np
from tqdm import tqdm
from os.path import join, exists
#from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
#from torchvision.utils import save_image

import warnings
warnings.filterwarnings('ignore')



def generate_final_results_mix(config, cls_dict, data_dir, track_dir, save_dir, thr_score=0.4):
    """
    给定`test_tracking`输出的结果，生成最终跟踪结果
    - cls_dict: video->id->frame->exp->
    """

    tao_imgid2frame = json.load(open('./datasets/LaMOT-main/sequences/val/TAO_imgid2frame.json','r'))

    template_dir = join(data_dir, 'gt_templates', config.DATA.TEST)
    expdir = os.path.join(config.data_root,'annotations_v1','val',config.DATA.TEST)
    if config.DATA.TEST=='TAO':
        template_dir_list = [i+'/'+j for i in os.listdir(join(data_dir, 'gt_templates', config.DATA.TEST)) for j in os.listdir(os.path.join(data_dir, 'gt_templates', config.DATA.TEST,i))]
        explist = [i+'/'+j for i in os.listdir(expdir) for j in os.listdir(os.path.join(expdir,i))]
    else:
        template_dir_list = os.listdir(template_dir)
        explist = os.listdir(expdir)

    if exists(save_dir):
        shutil.rmtree(save_dir)

    for video in template_dir_list:
        
        if video not in cls_dict:

            continue
        video_dir_in = join(template_dir, video)
        video_dir_out = join(save_dir, video)
        
        if config.DATA.TEST == 'VisDrone':
            framelist = os.listdir(join(data_dir,'sequences','val',config.DATA.TEST+'-sequences',video))
        elif config.DATA.TEST == 'TAO':
            framelist = [i+1 for i in range(len(os.listdir(join(data_dir,'sequences','val',config.DATA.TEST+'-sequences',video))))]
        else:
            framelist = os.listdir(join(data_dir,'sequences','val',config.DATA.TEST+'-sequences',video,'img1'))

        if not isinstance(framelist[0],int):
            framelist = sorted(framelist, key=lambda x: int(x.split('.')[0]))
            MIN_FRAME = int(framelist[0].split('.')[0])
            MAX_FRAME = int(framelist[-1].split('.')[0])
        else:
            framelist = sorted(framelist)
            MIN_FRAME = int(framelist[0])
            MAX_FRAME = int(framelist[-1])

        # symbolic link for `gt.txt`
        for exp in os.listdir(video_dir_in):
            exp_dir_in = join(video_dir_in, exp)
            exp_dir_out = join(video_dir_out, exp)
            os.makedirs(exp_dir_out, exist_ok=True)
            gt_path_in = join(exp_dir_in, 'gt.txt')
            gt_path_out = join(exp_dir_out, 'gt.txt' )
            if not exists(gt_path_out):
                os.symlink(gt_path_in, gt_path_out)
        # load tracks
        # noinspection PyBroadException
        try:
            tracks = np.loadtxt(join(track_dir, video+'.txt'), delimiter=',')[:,:10]
            tracks[:,-1]=1
            tracks[:,-2]=1
            tracks[:,-3]=1

        except:
            tracks_1 = np.loadtxt(join(track_dir, video, 'car', 'predict.txt'), delimiter=',')
            if len(tracks_1.shape) == 2:
                tracks = tracks_1
                max_obj_id = max(tracks_1[:, 1])
            else:
                tracks = np.empty((0, 10))
                max_obj_id = 0
            # tracks[:,0]=tracks[:,0]-1
            tracks_2 = np.loadtxt(join(track_dir, video, 'pedestrian', 'predict.txt'), delimiter=',')
            if len(tracks_2.shape) == 2:
                tracks_2[:, 1] += max_obj_id
                tracks = np.concatenate((tracks, tracks_2), axis=0)

        video_dict = cls_dict[video]
        for obj_id, obj_dict in video_dict.items():
            for frame_id, frame_dict in obj_dict.items():
                
                for exp_json in explist:
                    if not exp_json.replace('_'+exp_json.split('_')[-1],'')==video:
                        continue
                
                    

                    exp_input = json.load(open(os.path.join(expdir,exp_json)))['language']
                    
                    exp_dir_out = join(video_dir_out, exp_input).replace(' ', '-')

                    score = frame_dict[exp_input]
                    with open(join(exp_dir_out, 'predict.txt'), 'a') as f:

                        if np.argmax(np.array(score))>0:

                            bbox = tracks[
                                (tracks[:, 0] == int(frame_id)) *
                                (tracks[:, 1] == int(obj_id))
                            ][0]
                            
                            assert bbox.shape in ((9, ), (10, ))
                            if config.DATA.TEST == 'TAO':
                                bbox[0] = tao_imgid2frame[str(int(bbox[0]))]
                            if MIN_FRAME <= bbox[0] <= MAX_FRAME:  # TODO

                                bbox[0] = int(bbox[0])
                                bbox[1] = int(bbox[1])
                                bbox = list(bbox[:6])+[1,1,1]
                                f.write(','.join(list(map(str, bbox))) + '\n')

