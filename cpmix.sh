SRC_ROOT=./retest-mix/LaMOT-best

mkdir -p ${SRC_ROOT}/mixeval

cp -r ${SRC_ROOT}/aed-tao/results/* ${SRC_ROOT}/mixeval 

cp -r ${SRC_ROOT}/bot-mot17/results/* ${SRC_ROOT}/mixeval 

cp -r ${SRC_ROOT}/mcbyte-sports/results/* ${SRC_ROOT}/mixeval 

cp -r ${SRC_ROOT}/u2-drone/results/* ${SRC_ROOT}/mixeval 

########################################
# Feel free to delete the "rm" line
rm -rf ${SRC_ROOT}/mixeval/pedestrian_detailed.csv
rm -rf ${SRC_ROOT}/mixeval/pedestrian_summary.txt
