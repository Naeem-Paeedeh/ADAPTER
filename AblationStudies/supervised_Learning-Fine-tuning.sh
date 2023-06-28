#!/bin/sh

# TODO: Set the root directory. The program will create subdirectories for log files and model weights snapshots for each pair of datasets (base and target) in this root directory.
DIR_SAVES="/run/media/user/Data/Saves_Adapter"
mkdir "${DIR_SAVES}"

## The shared part
DIR_SHARED="$DIR_SAVES/Saves-shared"
mkdir ${DIR_SHARED}

ARGS_SHARED="--model_type CCT-14/7x2 -n_estimators_classifier 2 -n_layers_classifier 1"

P3=fsl
P4=supervised_one_domain
P5=supervised_base_and_support_set

LN="-----------------------------------------------"
# TODO: Divide these datasets between the GPUs.
DATASETS="ChestX ISIC EuroSAT CropDisease"  # ChestX ISIC EuroSAT CropDisease
DATASET_BASE="miniImageNet" # tieredImageNet miniImageNet
###---------------------------------------------------------------------------------
for ds_base in $DATASET_BASE
do
  for ds in $DATASETS
  do
    echo $LN
    echo "Base dataset: ${ds_base}"
    echo "Target dataset: ${ds}"
    DIR_DS="$DIR_SAVES/Saves-${ds_base}-${ds}"
    mkdir "${DIR_DS}"

  ###---------------------------------------------------------------------------------
    # TODO: Set the device and the num_workers.
    ARGS_TEMP="$ARGS_SHARED -p ${P5} \
    -device cuda -num_workers 4 -aug_train 0 -use_LGC 1 \
    -n_episodes 600 \
    -num_classes_base 64 \
    -b_ds ${ds_base} -t_ds ${ds} -display_freq 5 -display_interval 0.5 \
    -i ${DIR_SHARED}/MiniImageNet-P4e-Epoch=400 \
    -target_subset_split ../datasets/split_seed_1/${ds}_unlabeled_20.csv \
    -subset_split ../datasets/split_seed_1/${ds}_labeled_80.csv"

    echo $LN
    echo "Experiment: 1-shot:"

    python ../main.py $ARGS_TEMP \
    -lr 5e-3 -n_shots 1 -n_iterations 500 -batch_size 5 \
    -l ${DIR_DS}/Log-P4P5--1-shot.txt || exit 1

    echo $LN
    echo "Experiment: 5-shot:"

    python ../main.py $ARGS_TEMP \
    -lr 5e-3 -n_shots 5 -n_iterations 250 -batch_size 5 \
    -l ${DIR_DS}/Log-P4P5--5-shot.txt || exit 1

#
#  echo "------------------------------------------"
#  echo "| All experiments are done successfully! |"
#  echo "------------------------------------------"

  done
done