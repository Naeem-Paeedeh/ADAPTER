#!/bin/sh

# TODO: Set the save directory. The program will create the subdirectories for each pair of base and target datasets.
DIR_SAVES="/home/user_name/ADAPTER-Saves/all"
mkdir "${DIR_SAVES}"
# TODO: Set the device, number of workers, and save file
DEVICE="cuda:0"
NUM_WORKERS=4
EPOCH=80

ARGS_SHARED="--model_type CCT-14/7x2"

P1=train_DINO
P2=train_two_domains_DINO
P3=fsl

LN="-----------------------------------------------"
ARGS_EVAL="-p ${P3}
-lr_training_classifier_head 1e-2
-weight_decay_training_classifier_head 1e-3
-use_LGC 1
-aug_train 1
-n_epochs 100
-n_iterations 100
-batch_size_training_classifier_head 4
-dropout_rate_classifier_head 0.0
-freeze_backbone 1
-n_estimators_classifier 4
-n_layers_classifier 4
-inference_use_pretrained_head 0
-display_interval 1
-display_freq 10
-LP_with_support_set 0
-device ${DEVICE}
-num_workers $NUM_WORKERS"

# TODO: We can divide them between the GPUs.
DATASETS="ChestX ISIC EuroSAT CropDisease"  # ChestX ISIC EuroSAT CropDisease
DATASET_BASE="tieredImageNet" # miniImageNet tieredImageNet

###---------------------------------------------------------------------------------
for ds_base in $DATASET_BASE
do
  for ds in $DATASETS
  do
    echo $LN
    echo "Base dataset: ${ds_base}"
    echo "Target dataset: ${ds}"
    DIR_DS="$DIR_SAVES/Saves-${ds_base}-${ds}"
    DIR_LOGS="${DIR_DS}/Logs"
    mkdir "${DIR_DS}"
    mkdir "${DIR_LOGS}"

    ###---------------------------------------------------------------------------------
    # TODO: Set the device and the num_workers.
    ARGS_TEMP="$ARGS_SHARED \
    -b_ds ${ds_base} -t_ds ${ds} -domain_dino target \
    -target_subset_split datasets/split_seed_1/${ds}_unlabeled_20.csv \
    -subset_split datasets/split_seed_1/${ds}_labeled_80.csv"

    echo $LN
    echo "Experiment: Let's evaluate it 1-shot:"

    python main.py $ARGS_TEMP $ARGS_EVAL \
    -n_shots 1 \
    -n_epochs_training_classifier_head 150 \
    -i ${DIR_DS}/P2e-Epoch=$EPOCH -l ${DIR_LOGS}/Log-P2P3-Epoch=$EPOCH-1_shot-LGC=1 || exit 1

    echo $LN
    echo "Experiment: Let's evaluate it 5-shot:"

    python main.py $ARGS_TEMP $ARGS_EVAL \
    -n_shots 5 \
    -n_epochs_training_classifier_head 150 \
    -i ${DIR_DS}/P2e-Epoch=$EPOCH -l ${DIR_LOGS}/Log-P2P3-Epoch=$EPOCH-5_shot-LGC=1 || exit 1

  echo "------------------------------------------"
  echo "All experiments are done successfully!"
  echo "------------------------------------------"

  done
done
