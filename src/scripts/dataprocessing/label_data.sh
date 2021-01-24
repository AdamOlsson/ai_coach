#!/bin/bash

function isAnswerNo() {
    if [[ "$1" == "no" ]] || [[ "$1" == "n" ]] || [[ "$1" == "" ]] 
    then
        return 0
    else
        return 1
    fi
}

# Read input params
for i in "$@"
do
case $i in
    -d=*|--data=*)
    DATA_DIR="${i#*=}"
    shift # past argument=value
    ;;
    *)
        # unknown option
    ;;
esac
done

if [[ ! -v DATA_DIR ]];
then
    echo "Please provide the data directory with the -d flag."
    exit 1
fi

DATA_DIR=$(readlink -f  "$DATA_DIR") # get abs path of data dir

# ask for wanted labels
read -p "Please provide the labels in a space separated list:" -a LABEL_LIST

echo "Please confirm these labels:"
for i in ${!LABEL_LIST[@]};
do
    echo $( expr $i + 1 )")" ${LABEL_LIST[$i]}
done

read -p "(yes/no):" ANSWER

if isAnswerNo $ANSWER ;
then
    echo "Please rerun the script and provide the correct labels."
    exit 0
fi

# Create directories
for i in ${!LABEL_LIST[@]};
do
    LABEL_DIR="$DATA_DIR/${LABEL_LIST[$i]}"
    if [[ -d "$LABEL_DIR" ]]
    then
	    read -p "Found directory $LABEL_DIR. Would you like to continue using this? (yes/no):" ANSWER
        if isAnswerNo $ANSWER ;
        then
            echo "Please correct the data directory and rerun the script."
            exit 0
        fi
    else
        mkdir "$LABEL_DIR"
    fi
done

function whiteSpace(){
    for (( i=0; i<$1; i++ ))
    do
        echo ""
    done
    return 0
}

function listLabels(){
    echo "Here are the labels:"
    for i in ${!LABEL_LIST[@]};
    do
        echo $( expr $i + 1 )")" ${LABEL_LIST[$i]} # we print the first index as 1 because it is on the far left if the keyboard
    done
    return 0
}

function focusTerminalWindow(){
    # don't touch this IFS
    IFS='
'
    RAW=$(wmctrl -l)
    for line in $RAW;
    do
        if [[ "$line" == *"/ai_coach"* ]]; # we assume filter for bash window id using "/ai_coach"
        then
            WINDOW_ID=$(echo $line | cut -d ' ' -f 1)
            break
        fi
    done

    wmctrl -i -a $WINDOW_ID
}


shopt -s nullglob # Fixes so that "$DATA_DIR/*.mp4" is not expanded but instead lists files in dir
for FILE in $DATA_DIR/*.mp4;
do
    xdg-open "$FILE"

    focusTerminalWindow

    whiteSpace 3
    listLabels
    whiteSpace 1
    
    # user input
    read -p "What label should this file have?: " LABEL_ID
    while [[ ${#LABEL_LIST[@]} < $LABEL_ID ]] || [[ $LABEL_ID < 1 ]]
    do
        echo "Error: Please enter a valid label."
        read -p "What label should this file have?: " LABEL_ID
    done

    LABEL_ID=$( expr $LABEL_ID - 1)
    LABEL_DIR="$DATA_DIR/${LABEL_LIST[$LABEL_ID]}"

    echo "Labeling file as ${LABEL_LIST[$LABEL_ID]}"
    mv $FILE $LABEL_DIR
    
done