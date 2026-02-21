We reorganized most datasets for clarity and consistency. Please prepare them as follows:

1. Download the `datasets` folder from 
<a href="https://pan.baidu.com/s/1L-43y9SFDKmgl3dJNRlvNA?pwd=d3qj" title="model">FlexHook_best</a> (This link points to the anonymous cloud storage of FlexHook's weights and preprocessed public dataset for reproducing the model mentioned in the paper, and does not include any contributions beyond those mentioned in the paper) and place it in the FlexHook root directory.
2. Download <a href='https://github.com/Nathan-Li123/LaMOT'>LaMOT</a>, <a href='https://github.com/dyhBUPT/iKUN'>Refer-dance</a> and <a href='https://github.com/zyn213/TempRMOT'>Refer-KITTI/v2</a> datasets (all of them are public datasets and not a contribution of this paper) as per their official instructions.
3. As we only excluded image files from the `datasets` folder, simply remove placeholder folders and create soft links from your downloaded video data to the corresponding dataset folders.

Example:

```bash
rm -rf XXX/FlexHook/datasets/LaMOT-main/sequences/train/MOT17-sequences
ln -s XXX/data/MOT17/train XXX/FlexHook/datasets/LaMOT-main/sequences/train/MOT17-sequences

rm -rf XXX/FlexHook/datasets/refer-kitti/KITTI/training
ln -s XXX/data/kitti_tracking/data_tracking_image_2/training XXX/FlexHook/datasets/refer-kitti/KITTI/training
```

The code for processing datasets to generate JSON files under `datasets` will be provided later once organized.

