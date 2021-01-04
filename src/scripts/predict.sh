#!/bin/bash

# Read input params
for i in "$@"
do
case $i in
    --video=*)
    VIDEO="${i#*=}"
    shift # past argument=value
    ;;
    -w=*|--network_weights=*)
    NETWORKWEIGHTS="${i#*=}"
    shift # past argument=value
    ;;
    -p=*|--openposeroot=*)
    OPENPOSEROOTDIR="${i#*=}"
    shift # past argument=value
    ;;
    -h=*|--help=*)
    echo "Usage: predict.sh --video=<video path> -p=<openpose root dir> -w=<path to network wieghts>"
    EXIT=1
    ;;
    *)
        # unknown option
    ;;
esac
done

# Get absolute path of input files
if [[ "$VIDEO" != /* ]];
then
    VIDEO=$(readlink -f  "$VIDEO")
fi
OPENPOSEROOTDIR=$(readlink -f  "$OPENPOSEROOTDIR")
WORKINGDIR=$(pwd)
TMPKEYPOINTDIR="$WORKINGDIR/pose_est_tmp"

#mkdir $TMPKEYPOINTDIR

OPENPOSEBIN="$OPENPOSEROOTDIR/build/examples/openpose/openpose.bin"
cd $OPENPOSEROOTDIR # openpose requires to be in project root dir

# Do pose estimation
$OPENPOSEBIN \
        --keypoint_scale 4 \
        --display 0 --render_pose 0 \
        --video "$VIDEO" \
        --number_people_max 1 \
        --write_json "$TMPKEYPOINTDIR"

cd $WORKINGDIR

python predict.py -w "$NETWORKWEIGHTS" -j "$TMPKEYPOINTDIR"

rm -r $TMPKEYPOINTDIR