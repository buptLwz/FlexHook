##### TRACKERS_FOLDER and TRACKERS_TO_EVAL should be absolute path

python3 ./TrackEval/scripts/run_mot_challenge.py \
            --METRICS HOTA \
            --SEQMAP_FILE './seqmaps/mixseqmap.txt' \
            --SKIP_SPLIT_FOL True \
            --GT_FOLDER ./datasets/refer-kitti/KITTI/training/image_02 \
            --TRACKERS_FOLDER /XXX/FlexHook/retest-mix/LaMOT-best/mixeval \
            --GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
            --TRACKERS_TO_EVAL /XXX/FlexHook/retest-mix/LaMOT-best/mixeval \
            --USE_PARALLEL True \
            --NUM_PARALLEL_CORES 2 \
            --SKIP_SPLIT_FOL True \
            --PLOT_CURVES False
