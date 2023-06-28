#!/bin/sh

# TODO: Set the root directory. The program will create subdirectories for log files and model weights snapshots for each pair of datasets (base and target) in this root directory.
DIR_SAVES="/run/media/user/Data/Saves_Adapter"
mkdir "${DIR_SAVES}"

## The shared part
DIR_SHARED="$DIR_SAVES/Saves-shared"
mkdir ${DIR_SHARED}

ARGS_SHARED="--model_type CCT-14/7x2"

P3=fsl
P4=supervised_one_domain
P5=supervised_base_and_support_set

LN="-----------------------------------------------"
#### P4
ARGS_TEMP="$ARGS_SHARED $ARGS_FINAL \
 -p ${P4} -aug_train 1 -num_classes_base 64 -batch_size 200 \
 -l ${DIR_SHARED}/Log-P4.txt"

python ../main.py $ARGS_TEMP -n_epochs 400 -i ${DIR_SHARED}/MiniImageNet-P4e-Epoch=1 \
 -save_freq_epoch 50 -time_interval_to_save 60 -display_freq 100  \
 -lr 0.025 -o ${DIR_SHARED}/MiniImageNet-P4e || exit 1

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
    ARGS_EVAL="$ARGS_SHARED \
    -device cuda -num_workers 4 \
    -b_ds ${ds_base} -t_ds ${ds} -domain_dino target -aug_train 1 -use_LGC 1 \
    -i ${DIR_SHARED}/MiniImageNet-P4e-Epoch=400 \
    -target_subset_split ../datasets/split_seed_1/${ds}_unlabeled_20.csv \
    -subset_split ../datasets/split_seed_1/${ds}_labeled_80.csv \
    -aug_train 0 -n_epochs 100 -n_iterations 100 -freeze_backbone 1 \
    -n_estimators_classifier 4 -n_layers_classifier 4 \
    -inference_use_pretrained_head 0 -display_interval 10 -display_freq 10"

    echo $LN
    echo "Experiment: Let's evaluate it 1-shot:"

    python ../main.py $ARGS_EVAL -p ${P3} -n_shots 1 \
    -l ${DIR_DS}/Log-P4P3-Epoch=200-1_shot-LGC=1-Supervised_base.txt || exit 1

    echo $LN
    echo "Experiment: Let's evaluate it 5-shot:"

    python ../main.py $ARGS_EVAL -p ${P3} -n_shots 5 \
    -l ${DIR_DS}/Log-P4P3-Epoch=200-5_shot-LGC=1-Supervised_base.txt || exit 1

  echo "------------------------------------------"
  echo "| All experiments are done successfully! |"
  echo "------------------------------------------"

  done
done