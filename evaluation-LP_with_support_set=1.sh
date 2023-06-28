#!/bin/sh

# TODO: Set the root directory. The program will create subdirectories for log files and model weights snapshots for each pair of datasets (base and target) in this root directory.
DIR_SAVES="/run/media/user/Data/Saves_Adapter"
mkdir "${DIR_SAVES}"

ARGS_SHARED="--model_type CCT-14/7x2"

P1=train_DINO
P2=train_two_domains_DINO
P3=fsl

LN="-----------------------------------------------"
ARGS_EVAL="-aug_train 0 -n_epochs 100 -n_iterations 100 -batch_size_training_classifier_head 4 -freeze_backbone 1 -n_estimators_classifier 4 -n_layers_classifier 4 \
 -inference_use_pretrained_head 0 -display_interval 10 -display_freq 50 -LP_with_support_set 1"

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
    # TODO: Set the device and the num_workers.
    ARGS_TEMP="$ARGS_SHARED \
    -device cuda -num_workers 4 -lr_training_classifier_head 0.01 \
    -b_ds ${ds_base} -t_ds ${ds} -p ${P2} -domain_dino target -aug_train 1 -use_LGC 1 -display_interval 0.5 \
    -target_subset_split datasets/split_seed_1/${ds}_unlabeled_20.csv \
    -subset_split datasets/split_seed_1/${ds}_labeled_80.csv"

    echo $LN
    echo "Experiment: Let's evaluate it for 1-shot:"
    python main.py $ARGS_TEMP $ARGS_EVAL -p ${P3} -n_shots 1 \
    -i ${DIR_DS}/P2e-Epoch=200 -l ${DIR_DS}/Log-P2P3-Epoch=200-1_shot-LGC=1-LP_with_support_set=1-one_hot_logits_for_support_set=0 || exit 1

    echo $LN
    echo "Experiment: Let's evaluate it for 5-shot:"
    python main.py $ARGS_TEMP $ARGS_EVAL -p ${P3} -n_shots 5 \
    -i ${DIR_DS}/P2e-Epoch=200 -l ${DIR_DS}/Log-P2P3-Epoch=200-5_shot-LGC=1-LP_with_support_set=1-one_hot_logits_for_support_set=0 || exit 1

#    python main.py $ARGS_TEMP $ARGS_EVAL -p ${P3} -n_shots 5 \
#    -i ${DIR_DS}/P2e-Epoch=200 -one_hot_logits_for_support_set 1 \
#    -l ${DIR_DS}/Log-P2P3-Epoch=200-5_shot-LGC=1-LP_with_support_set=1-one_hot_logits_for_support_set=1.txt || exit 1
#
#    python main.py $ARGS_TEMP $ARGS_EVAL -p ${P3} -n_shots 1 \
#    -i ${DIR_DS}/P2e-Epoch=200 -one_hot_logits_for_support_set 1 \
#    -l ${DIR_DS}/Log-P2P3-Epoch=200-1_shot-LGC=1-LP_with_support_set=1-one_hot_logits_for_support_set=1.txt || exit 1

  echo "------------------------------------------"
  echo "| All experiments are done successfully! |"
  echo "------------------------------------------"

  done
done