# ADAPTER

This repository contains the implementation of the ADAPTER algorithm from the "Cross-Domain Few-Shot Learning via Adaptive Transformer Networks" paper.

In the following, it is explained how to prepare the datasets and run the code.

## Requirements

For the miniImageNet experiments, you can use the Python version 3.10.9. We installed the torch v. 1.13.1-3 , numpy v. 1.24.1-1 , torchvision v. 0.14.1-1 , pandas v. 1.5.3-1 , Pillow v. 9.4.0-2, and tqdm==4.64.1-2 packages in Manjaro linux. tqdm can be installed with:

```bash
pip install tqdm
```

For tieredImageNet experiments, we used the Python version 3.9.16. The conda environment can be created with the following bash command:

```bash
conda create -n adapter python=3.9.16
```

Next, activate the new environment:

```bash
conda activate adapter
```

The requirements for the tieredImageNet experiments can be installed in the conda environment with the following command:

```bash
pip install -r requirements-tieredImageNet.txt
```

## Preparing the datasets

Follow the instructions of https://github.com/IBM/cdfsl-benchmark to download the Mini-ImageNet, ChestX, ISIC2018, EuroSAT, and Plant Disease. For Tiered-ImageNet, please follow the instructions in https://github.com/yaoyao-liu/tiered-imagenet-tools repository.

## How to run the code

Do the following to perform the training and evaluation:

1- Install the requirements. Different versions of the same packages should also work.

2- Set the directories in configs_dataset.py and run_experiments.sh

3- Set the batch size in the training script.

4- Run the training script.

5- Run the evaluation script.
