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

FRAMES = {
    '0005': (0, 296),
    '0011': (0, 372),
    '0013': (0, 339),
    '0019': (0, 1058),
'dancetrack0081': [1, 984], 'dancetrack0087': [1, 1003], 'dancetrack0063': [1, 1000], 
     'dancetrack0098': [1, 1203], 'dancetrack0006': [1, 1202], 'dancetrack0075': [1, 803], 
     'dancetrack0057': [1, 622], 'dancetrack0047': [1, 1203], 'dancetrack0074': [1, 1203], 
     'dancetrack0035': [1, 703], 'dancetrack0097': [1, 1203], 'dancetrack0016': [1, 2163], 
     'dancetrack0037': [1, 1203], 'dancetrack0024': [1, 763], 'dancetrack0099': [1, 603], 
     'dancetrack0029': [1, 1263], 'dancetrack0072': [1, 1203], 'dancetrack0062': [1, 1203], 
     'dancetrack0043': [1, 183], 'dancetrack0015': [1, 1203], 'dancetrack0010': [1, 1203], 
     'dancetrack0002': [1, 1203], 'dancetrack0051': [1, 1203], 'dancetrack0007': [1, 1203], 
     'dancetrack0061': [1, 1203], 'dancetrack0027': [1, 403], 'dancetrack0008': [1, 883], 
     'dancetrack0049': [1, 1203], 'dancetrack0090': [1, 1004], 'dancetrack0032': [1, 604], 
     'dancetrack0058': [1, 1601], 'dancetrack0065': [1, 702], 'dancetrack0020': [1, 583], 
     'dancetrack0096': [1, 603], 'dancetrack0041': [1, 1003], 'dancetrack0039': [1, 1242], 
     'dancetrack0023': [1, 1483], 'dancetrack0012': [1, 1203], 'dancetrack0066': [1, 1202], 
     'dancetrack0079': [1, 1202], 'dancetrack0052': [1, 1203], 'dancetrack0068': [1, 1203], 
     'dancetrack0044': [1, 1203], 'dancetrack0018': [1, 503], 'dancetrack0004': [1, 1203], 
     'dancetrack0033': [1, 803], 'dancetrack0094': [1, 603], 'dancetrack0030': [1, 1263], 
     'dancetrack0055': [1, 1203], 'dancetrack0077': [1, 1203], 'dancetrack0083': [1, 603], 
     'dancetrack0019': [1, 2402], 'dancetrack0005': [1, 1203], 'dancetrack0045': [1, 1203], 
     'dancetrack0034': [1, 923], 'dancetrack0069': [1, 1403], 'dancetrack0086': [1, 603], 
     'dancetrack0026': [1, 302], 'dancetrack0014': [1, 1203], 'dancetrack0053': [1, 1204], 
     'dancetrack0080': [1, 1201], 'dancetrack0025': [1, 803], 'dancetrack0001': [1, 703], 
     'dancetrack0073': [1, 703], 'dancetrack0082': [1, 603]
}  # 视频起止帧



def generate_final_results(config, cls_dict, data_dir, track_dir, save_dir, thr_score=0.4):
    """
    给定`test_tracking`输出的结果，生成最终跟踪结果
    - cls_dict: video->id->frame->exp->
    """
    template_dir = join(data_dir, 'gt_template')
    if exists(save_dir):
        shutil.rmtree(save_dir)
    for video in os.listdir(template_dir):
        
        if video not in cls_dict:
            continue
        video_dir_in = join(template_dir, video)
        video_dir_out = join(save_dir, video)
        
        MIN_FRAME, MAX_FRAME = FRAMES[video]
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
            tracks = np.loadtxt(join(track_dir, video, 'all', 'gt.txt'), delimiter=',')
        except:
            tracks_1 = np.loadtxt(join(track_dir, video, 'car', 'predict.txt'), delimiter=',')
            if len(tracks_1.shape) == 2:
                tracks = tracks_1
                max_obj_id = max(tracks_1[:, 1])
            else:
                tracks = np.empty((0, 10))
                max_obj_id = 0

            tracks_2 = np.loadtxt(join(track_dir, video, 'pedestrian', 'predict.txt'), delimiter=',')
            if len(tracks_2.shape) == 2:
                tracks_2[:, 1] += max_obj_id
                tracks = np.concatenate((tracks, tracks_2), axis=0)

        if config.track_root.endswith('NeuralSORT_MIX'):
            tracks[:,0]=tracks[:,0]
        else:
            tracks[:,0]=tracks[:,0]-1

        video_dict = cls_dict[video]
        for obj_id, obj_dict in video_dict.items():
            for frame_id, frame_dict in obj_dict.items():
                
                for exp in os.listdir(os.path.join(config.data_root,'expression',video)):
                

                    exp_input = json.load(open(os.path.join(config.data_root,'expression',video,exp)))['sentence']
                    
                    exp_dir_out = join(video_dir_out, exp.replace('.json',''))

                    score = frame_dict[exp_input]
                    with open(join(exp_dir_out, 'predict.txt'), 'a') as f:

                        if np.argmax(np.array(score))>0:

                            
                            bbox = tracks[
                                (tracks[:, 0] == int(frame_id)) *
                                (tracks[:, 1] == int(obj_id))
                            ][0]

                            assert bbox.shape in ((9, ), (10, ))
                            if MIN_FRAME <= bbox[0] <= MAX_FRAME:  # TODO
                                if not config.track_root.endswith('NeuralSORT_MIX'):
                                    bbox[0]+=1
                                f.write(','.join(list(map(str, bbox))) + '\n')


def generate_final_results_dance(config, cls_dict, data_dir, track_dir, save_dir, thr_score=0.4):
    """
    给定`test_tracking`输出的结果，生成最终跟踪结果
    - cls_dict: video->id->frame->exp->
    """

    template_dir = join(data_dir, 'gt_template')
    if exists(save_dir):
        shutil.rmtree(save_dir)
    for video in os.listdir(template_dir):
        
        if video not in cls_dict:
            continue
        video_dir_in = join(template_dir, video)
        video_dir_out = join(save_dir, video)
        MIN_FRAME, MAX_FRAME = FRAMES[video]
        # symbolic link for `gt.txt`
        for exp in os.listdir(video_dir_in):
            exp_dir_in = join(video_dir_in, exp)
            exp_dir_out = join(video_dir_out, exp)
            os.makedirs(exp_dir_out, exist_ok=True)
            gt_path_in = join(exp_dir_in, 'gt.txt')
            gt_path_out = join(exp_dir_out, 'gt.txt' )
            if not exists(gt_path_out):
                os.symlink(gt_path_in, gt_path_out)

        tracks = np.loadtxt(join(track_dir, video+'.txt'), delimiter=',')
        if len(tracks.shape) == 1:
            tracks = np.empty((0, 10))

        video_dict = cls_dict[video]
        for obj_id, obj_dict in video_dict.items():
            for frame_id, frame_dict in obj_dict.items():

                
                for exp in os.listdir(os.path.join(config.data_root,'expression',video)):

                    exp_input = json.load(open(os.path.join(config.data_root,'expression',video,exp)))['sentence']

                    exp_dir_out = join(video_dir_out, exp.replace('.json',''))

                    score = frame_dict[exp_input]
                    with open(join(exp_dir_out, 'predict.txt'), 'a') as f:

                        if np.argmax(np.array(score))>0:

                            bbox = tracks[
                                (tracks[:, 0] == int(frame_id)) *
                                (tracks[:, 1] == int(obj_id))
                            ][0]
                            assert bbox.shape in ((9, ), (10, ))
                            if MIN_FRAME <= bbox[0] <= MAX_FRAME:  # TODO
                                # the min/max frame is not included in `gt.txt`
                                f.write(','.join(list(map(str, bbox))) + '\n')

