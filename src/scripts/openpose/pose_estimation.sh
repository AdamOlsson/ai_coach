#!/bin/bash

# Read input params
for i in "$@"
do
case $i in
    -a=*|--annotations=*)
    INPUT_ANNOTATIONSFILE="${i#*=}"
    shift # past argument=value
    ;;
    -o=*|--outputdir=*)
    OUTDIR="${i#*=}"
    shift # past argument=value
    ;;
    -p=*|--openposeroot=*)
    OPENPOSEROOTDIR="${i#*=}"
    shift # past argument=value
    ;;
    *)
        # unknown option
    ;;
esac
done

# Get absolute path of input files
INPUT_ANNOTATIONSFILE=$(readlink -f  "$INPUT_ANNOTATIONSFILE")
OUTDIR=$(readlink -f  "$OUTDIR")
OPENPOSEROOTDIR=$(readlink -f  "$OPENPOSEROOTDIR")


WRITEDIR="$OUTDIR/pose_predictions"
echo "Results are written to $WRITEDIR"

# Ask user to remove existing dir or not
if [[ -d "$WRITEDIR" ]]
then
    while [ "$ANSWER" != "yes" ] && [ "$ANSWER" != "no" ]
    do
        echo "$WRITEDIR exists on your filesystem."
        read -p "Would you like to remove it? (yes/no):" ANSWER
    done

    if [ "$ANSWER" == "yes" ]
    then
        echo "Removing $WRITEDIR"
        rm -r $WRITEDIR
    else
        echo "Not removing $WRITEDIR"
    fi
fi

# Create the new write dir and data subdir
echo "Creating directory $WRITEDIR"
DATADIR="$WRITEDIR/data"
mkdir -p "$DATADIR"

# Add alias for openpose binary
OPENPOSEBIN="$OPENPOSEROOTDIR/build/examples/openpose/openpose.bin"
cd $OPENPOSEROOTDIR # openpose requires to be in project root dir

# Annotations root is used to get absolute path for videos in annotations file
ANNOTATIONSROOT="$(dirname "${INPUT_ANNOTATIONSFILE}")"

# Start processing
IFS=','
while read -r FILEPATH LABEL
do
    # Skip header in annotations file
    if [[ "$FILEPATH" =~ "#".* ]]
    then
        continue
    fi

    # Create new directory for label
    LABELDIR="$DATADIR/$LABEL"
    if [[ ! -d "$LABELDIR" ]]
    then
        mkdir $LABELDIR
    fi

    # Make directory for keypoints of video
    FILENAME="$(basename $FILEPATH)"
    BASENAME=${FILENAME%.*}
    KEYPOINTDIR="$LABELDIR/$BASENAME"

    if [[ -d "$KEYPOINTDIR" ]]
    then
	echo "Found directory $KEYPOINTDIR. Checking if last run succeeded..."
	COUNT=$(ls -l $KEYPOINTDIR | wc -l)
	if [ $COUNT -gt 1 ]
	then
	    echo "Last run was successful, found $COUNT files."
	    continue
	else
	    echo "Last run failed, rerunning..."
	fi
    else
        mkdir $KEYPOINTDIR
    fi

   # Absolute path
    FILEPATHABS="$ANNOTATIONSROOT/$FILEPATH"

    # Do the pose estimations
    # --model_pose "COCO" for 18 node bodies
    $OPENPOSEBIN \
            --keypoint_scale 4 \
            --display 0 --render_pose 0 \
            --video "$FILEPATHABS" \
            --number_people_max 1 \
            --write_json "$KEYPOINTDIR"

    # Check exit code
    if [ $? -eq 0 ]
    then
        echo "Successfully processed $FILEPATHABS"
    else
        # Error
        echo "Error on $FILEPATHABS"

        # Create error dir
        if [[ ! -d "$WRITEDIR/errors" ]]
        then
            mkdir "$WRITEDIR/errors"
        fi

        # Create class error log
        if [[ ! -f "$WRITEDIR/errors/$LABEL.log" ]]
        then
            touch "$WRITEDIR/errors/$LABEL.log"
        fi

        # Write filename to error log
        echo "$FILEPATHABS" >> "$WRITEDIR/errors/$LABEL.log"
    fi
done < "$INPUT_ANNOTATIONSFILE"

echo "Done. Results can be found at"
echo $WRITEDIR
