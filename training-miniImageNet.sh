#!/bin/sh

# TODO: Set the root directory. The program will create subdirectories for log files and model weights snapshots for each pair of datasets (base and target) in this root directory.
DIR_SAVES="Saves_Adapter"
mkdir "${DIR_SAVES}"

# TODO: Set the device and the num_workers.
ARGS_SHARED="--model_type CCT-14/7x2 -device cuda -num_workers 4 -use_LGC 1 -display_interval 0.5 -aug_train 1"

P1=train_DINO
P2=train_two_domains_DINO
P3=fsl

# TODO: Divide these datasets between the GPUs.
DATASETS="ChestX ISIC EuroSAT CropDisease"
DATASET_BASE="miniImageNet" # miniImageNet tieredImageNet

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
    ARGS_TEMP="$ARGS_SHARED \
    -b_ds ${ds_base} -t_ds ${ds} -p ${P2} -domain_dino target \
    -target_subset_split datasets/split_seed_1/${ds}_unlabeled_20.csv \
    -subset_split datasets/split_seed_1/${ds}_labeled_80.csv"

    ## Training
    # TODO: Set the batch size
    python main.py $ARGS_TEMP -n_epochs 500 \
    -save_freq_epoch 5 -time_interval_to_save 60 -batch_size 16 \
    -lr 0.00025 -min_lr 5e-7 \
    -o ${DIR_DS}/P2e \
    -l ${DIR_DS}/Log-P2.txt || exit 1

  echo "------------------------------------------"
  echo "| All experiments are done successfully! |"
  echo "------------------------------------------"

  done
done