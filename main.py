#!/usr/bin/env python3

# We used some codes from the implementation of the following references and tried to conform to the conditions in their experiments:
# @misc{rw2019timm,
#   author = {Ross Wightman},
#   title = {PyTorch Image Models},
#   year = {2019},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   doi = {10.5281/zenodo.4414861},
#   howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
# }

# @article{Xu2021CDTransCT,
#   title={CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation},
#   author={Tongkun Xu and Weihua Chen and Pichao Wang and Fan Wang and Hao Li and Rong Jin},
#   journal={ArXiv},
#   year={2021},
#   volume={abs/2109.06165}
# }

# @article{Yang2021TVTTV,
#   title={TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation},
#   author={Jinyu Yang and Jingjing Liu and Ning Xu and Junzhou Huang},
#   journal={2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
#   year={2021},
#   pages={520-530}
# }

# @article{Caron2021EmergingPI,
#   title={Emerging Properties in Self-Supervised Vision Transformers},
#   author={Mathilde Caron and Hugo Touvron and Ishan Misra and Herv'e J'egou and Julien Mairal and Piotr Bojanowski and Armand Joulin},
#   journal={2021 IEEE/CVF International Conference on Computer Vision (ICCV)},
#   year={2021},
#   pages={9630-9640}
# }

from __future__ import absolute_import, division, print_function
from ast import List

import math
import sys
from builtins import Exception

import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from utils.other_utilities import Stopwatch, pair, MovingAverageSet, parse_arguments, DatasetFromTensors, AverageMeterSet_STARTUP
import utils.dino_utils as dino_utils
import copy
from configs.configs_training import ConfigurationTraining
from models.ViT_CCT import Conditions
from PseudoLabeling.LGC import LGC
import torch.nn.functional as F
from torchvision import datasets
from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot, miniImageNet_few_shot, tiered_ImageNet_few_shot
from torch.utils.data import Subset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.MLP_V import MLP_V


# The codes for loading the datasets are taken from the STARTUP code
def load_base_dataset_teacher(configs: ConfigurationTraining, batch_size, aug_train, directory='train', drop_last=False):
    if configs.base_dataset == 'miniImageNet':
        data_manager = miniImageNet_few_shot.SimpleDataManager(configs.image_size,
                                                               batch_size=batch_size,
                                                               split=None,
                                                               directory=directory)

        dataloader_base = data_manager.get_data_loader(aug=aug_train,
                                                       num_workers=configs.num_workers,
                                                       drop_last=drop_last)
        # num_classes = dataloader_base.dataset.d.labels.max() - dataloader_base.dataset.d.labels.min() + 1
        return dataloader_base
    else:
        msg = 'Unknown dataset!'
        configs.logger.exception(msg)
        raise ValueError(msg)


def load_base_dataset_student(configs: ConfigurationTraining, batch_size, split_to_train_val=True):
    # Create the base dataset
    if configs.base_dataset == 'miniImageNet':
        base_transform = miniImageNet_few_shot.TransformLoader(configs.image_size).get_composed_transform(aug=True)
        base_transform_test = miniImageNet_few_shot.TransformLoader(
            configs.image_size).get_composed_transform(aug=False)
        base_dataset = datasets.ImageFolder(root=configs.base_path, transform=base_transform)
        if configs.base_split is not None:
            base_dataset = miniImageNet_few_shot.construct_subset(
                base_dataset, configs.base_split)
    else:
        msg = 'Unknown dataset!'
        configs.logger.exception(msg)
        raise ValueError(msg)

    if split_to_train_val:
        # Generate train_set and val_set for base dataset
        base_ind = torch.randperm(len(base_dataset))

        base_train_ind = base_ind[:int((1 - configs.base_val_ratio) * len(base_ind))]
        base_val_ind = base_ind[int((1 - configs.base_val_ratio) * len(base_ind)):]

        base_dataset_val = copy.deepcopy(base_dataset)
        base_dataset_val.transform = base_transform_test

        base_train_set = Subset(base_dataset, base_train_ind)
        base_val_set = Subset(base_dataset_val, base_val_ind)

        print("Size of base validation set", len(base_val_set))

        base_train_loader = DataLoader(base_train_set, batch_size=batch_size,
                                       num_workers=configs.num_workers,
                                       shuffle=True, drop_last=True)

        base_val_loader = DataLoader(base_val_set, batch_size=batch_size,
                                     num_workers=configs.num_workers,
                                     shuffle=False, drop_last=False)

        return base_train_loader, base_val_loader
    else:
        base_train_loader = DataLoader(base_dataset, batch_size=batch_size,
                                       num_workers=configs.num_workers,
                                       shuffle=True, drop_last=True)
        return base_train_loader


def load_target_dataset_student(configs: ConfigurationTraining,
                                batch_size: int,
                                aug: bool,
                                split_to_train_val: bool = True,
                                shuffle_for_train_loader: bool = True,
                                drop_last_for_train_loader: bool = True,
                                dataloader_or_dataset: bool = True,
                                custom_transform_train=None):
    """Loads the unlabeled subset of the target dataset.

    Args:
        configs (ConfigurationTraining): The configurations
        batch_size (int): Bach-size when we want a dataloader
        aug (bool): Augmentation if we want to use the default transforms.
        split_to_train_val (bool, optional): Split the dataset to train and validation or not. Defaults to True.
        shuffle_for_train_loader (bool, optional): Enable shuffling if we want a dataloader. Defaults to True.
        drop_last_for_train_loader (bool, optional): Ignores the last batch if it is not divisible by batch-size. Defaults to True.
        dataloader_or_dataset (bool, optional): Do you want a dataloader or dataset? Defaults to True.
        custom_transform_train (_type_, optional): We can apply our custom transform on dataset. Defaults to None.
    """
    
    # If we give a custom transform, this function use it. Otherwise, it uses the default transforms.
    transform = custom_transform_train
    
    if custom_transform_train is None:
        if configs.target_dataset == 'ISIC':
            transform = ISIC_few_shot.TransformLoader(configs.image_size).get_composed_transform(aug=aug)   # Default aug was True
            transform_test = ISIC_few_shot.TransformLoader(configs.image_size).get_composed_transform(aug=False)
        elif configs.target_dataset == 'EuroSAT':
            transform = EuroSAT_few_shot.TransformLoader(configs.image_size).get_composed_transform(aug=aug)
            transform_test = EuroSAT_few_shot.TransformLoader(configs.image_size).get_composed_transform(aug=False)
        elif configs.target_dataset == 'CropDisease':
            transform = CropDisease_few_shot.TransformLoader(configs.image_size).get_composed_transform(aug=aug)
            transform_test = CropDisease_few_shot.TransformLoader(configs.image_size).get_composed_transform(aug=False)
        elif configs.target_dataset == 'ChestX':
            transform = Chest_few_shot.TransformLoader(
                configs.image_size).get_composed_transform(aug=aug)
            transform_test = Chest_few_shot.TransformLoader(configs.image_size).get_composed_transform(aug=False)
        elif configs.target_dataset == 'miniImageNet_test':
            transform = miniImageNet_few_shot.TransformLoader(configs.image_size).get_composed_transform(aug=aug)
            transform_test = miniImageNet_few_shot.TransformLoader(configs.image_size).get_composed_transform(aug=False)
        else:
            msg = 'Unknown dataset!'
            configs.logger.exception(msg)
            raise ValueError(msg)
    
    # Create the target dataset
    if configs.target_dataset == 'ISIC':
        dataset = ISIC_few_shot.SimpleDataset(transform, split=configs.target_subset_split)
    elif configs.target_dataset == 'EuroSAT':
        dataset = EuroSAT_few_shot.SimpleDataset(transform, split=configs.target_subset_split)
    elif configs.target_dataset == 'CropDisease':
        dataset = CropDisease_few_shot.SimpleDataset(transform, split=configs.target_subset_split)
    elif configs.target_dataset == 'ChestX':
        dataset = Chest_few_shot.SimpleDataset(transform, split=configs.target_subset_split)
    elif configs.target_dataset == 'miniImageNet_test':
        dataset = miniImageNet_few_shot.SimpleDataset(transform, split=configs.target_subset_split)
    else:
        msg = 'Unknown dataset!'
        configs.logger.exception(msg)
        raise ValueError(msg)

    print("Size of target dataset", len(dataset))

    if split_to_train_val:
        dataset_test = copy.deepcopy(dataset)

        ind = torch.randperm(len(dataset))

        # split the target dataset into train and val
        # 10% of the unlabeled data is used for validation
        train_ind = ind[:int(0.9 * len(ind))]
        val_ind = ind[int(0.9 * len(ind)):]

        train_set = Subset(dataset, train_ind)
        val_set = Subset(dataset_test, val_ind)

        if dataloader_or_dataset:
            train_loader = DataLoader(train_set,
                                      batch_size=batch_size,
                                      num_workers=configs.num_workers,
                                      shuffle=shuffle_for_train_loader,
                                      drop_last=drop_last_for_train_loader)
            val_loader = DataLoader(val_set, batch_size=batch_size,
                                    num_workers=configs.num_workers,
                                    shuffle=False,
                                    drop_last=False)

            return train_loader, val_loader
        else:   # When we need the dataset
            return train_set, val_set, transform, transform_test
    else:
        if dataloader_or_dataset:
            train_loader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      num_workers=configs.num_workers,
                                      shuffle=shuffle_for_train_loader,
                                      drop_last=drop_last_for_train_loader)
            return train_loader
        else:  # When we need the dataset
            return dataset, transform, transform_test


def loader_episodic(configs: ConfigurationTraining, aug=False):
    if configs.target_dataset == 'ISIC':
        data_manager = ISIC_few_shot
    elif configs.target_dataset == 'EuroSAT':
        data_manager = EuroSAT_few_shot
    elif configs.target_dataset == 'CropDisease':
        data_manager = CropDisease_few_shot
    elif configs.target_dataset == 'ChestX':
        data_manager = Chest_few_shot
    elif configs.target_dataset == 'miniImageNet_test':
        data_manager = miniImageNet_few_shot
    else:
        raise ValueError("Invalid Dataset!")

    novel_loader = \
        data_manager.SetDataManager(configs.image_size,
                                    n_episode=configs.n_episodes,
                                    n_query=configs.n_queries,
                                    n_way=configs.n_ways,
                                    n_support=configs.n_shots,
                                    split=configs.subset_split).get_data_loader(aug=aug,
                                                                                num_workers=configs.num_workers)
    return novel_loader


def load_target_dataset_for_DINO(configs: ConfigurationTraining):
    transform = dino_utils.DataAugmentationDINO(
        configs.global_crops_scale,
        configs.local_crops_scale,
        configs.local_crops_number,
    )
    # create the target dataset
    if configs.target_dataset == 'ISIC':
        dataset = ISIC_few_shot.SimpleDataset(transform, split=configs.target_subset_split)
    elif configs.target_dataset == 'EuroSAT':
        dataset = EuroSAT_few_shot.SimpleDataset(transform, split=configs.target_subset_split)
    elif configs.target_dataset == 'CropDisease':
        dataset = CropDisease_few_shot.SimpleDataset(transform, split=configs.target_subset_split)
    elif configs.target_dataset == 'ChestX':
        dataset = Chest_few_shot.SimpleDataset(transform, split=configs.target_subset_split)
    elif configs.target_dataset == 'miniImageNet_test':
        dataset = miniImageNet_few_shot.SimpleDataset(transform, split=configs.target_subset_split)
    else:
        msg = 'Unknown dataset!'
        configs.logger.exception(msg)
        raise ValueError(msg)

    print("Size of target dataset", len(dataset))

    train_loader = DataLoader(dataset,
                              batch_size=configs.batch_size,
                              num_workers=configs.num_workers,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    return train_loader, transform


def load_base_dataset_DINO(configs: ConfigurationTraining):
    transform = dino_utils.DataAugmentationDINO(
        configs.global_crops_scale,
        configs.local_crops_scale,
        configs.local_crops_number,
    )

    if configs.base_dataset == 'miniImageNet':
        data_manager = miniImageNet_few_shot.SimpleDataManager(configs.image_size,
                                                               batch_size=configs.batch_size,
                                                               split=None,
                                                               directory='train')
    elif configs.base_dataset == 'tieredImageNet':
        data_manager = tiered_ImageNet_few_shot.SimpleDataManager(configs.image_size,
                                                                  batch_size=configs.batch_size,
                                                                  split=None)
    else:
        raise NotImplementedError

    dataloader_base = data_manager.get_data_loader(aug=False,
                                                   num_workers=configs.num_workers,
                                                   drop_last=True,
                                                   transform=transform)

    return dataloader_base, transform


def calculate_and_display_accuracy(configs: ConfigurationTraining, acc_all):
    acc_all_np = np.asarray(acc_all)
    acc_mean = np.mean(acc_all_np)
    acc_std = np.std(acc_all_np)
    configs.logger.info('\nTest Acc (%d episodes) = %4.2f%% ± %4.2f%%' % (len(acc_all),
                                                                          acc_mean,
                                                                          1.96 * acc_std / np.sqrt(len(acc_all))
                                                                          )
                        )


def evaluate_on_query_set_LP_wihtout_support_set(configs: ConfigurationTraining,
                                                 x_query,
                                                 acc_all
                                                 ) -> List:
    """This method evaluates the accuracy of the model on query set. It uses head_target_self_attention from the configurations as classifier head.

    Args:
        configs (ConfigurationTraining): Configurations and settings.
        x_query (_type_): Query set samples.
        acc_all (_type_): The list of accuracies for all runs.

    Returns:
        _type_: List
    """
    classifier = configs.head_target_self_attention
    classifier.eval()

    model = configs.get_the_model()

    with torch.no_grad():
        scores_list = []
        features_list = []
        ds_temp = TensorDataset(x_query)
        dataloader_temp = DataLoader(ds_temp,
                                     batch_size=configs.batch_size,
                                     shuffle=False,
                                     drop_last=False)
        # query_size = configs.n_ways * configs.n_queries
        for x, in dataloader_temp:
            features = model(Conditions.single_self_attention, {'x': x.to(configs.device)})
            features_list.append(features)
            if configs.phase == 'supervised_base_and_support_set':
                # We utilize self-attention features twice
                scores_list.append(classifier(concatenate_features(features, configs.n_estimators_classifier)))
            else:
                scores_list.append(classifier(concatenate_features(features, configs.n_estimators_classifier)))

    scores = torch.cat(scores_list, dim=0)
    features = torch.cat(features_list, dim=0)
    if configs.use_LGC:
        with torch.no_grad():
            lgc = LGC()
            scores = lgc.compute(features, scores, configs.logger)
    # y_query = torch.arange(configs.n_ways).repeat_interleave(configs.n_queries).to(configs.device)
    y_query = np.repeat(range(configs.n_ways), configs.n_queries)
    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()

    top1_correct = np.sum(topk_ind[:, 0] == y_query)
    correct_this, count_this = float(top1_correct), len(y_query)
    acc_all.append((correct_this / count_this * 100))
    return acc_all


def evaluate_on_query_set_LP_with_support_set(configs: ConfigurationTraining,
                                              x_query,
                                              x_support,
                                              y_support,
                                              acc_all) -> List:
    """This method evaluates the accuracy of the model on query set with label propagation. It utilizes the lebles of the support set for label propagation, but it would ignore them for calculating the accuracy. It uses head_target_self_attention from the configurations as classifier head.

    Args:
        configs (ConfigurationTraining): Configurations and settings.
        x_query (_type_): Query set samples.
        x_support (_type_): Support set samples for obtaining the embeddings.
        y_support (_type_): Support set labels to assign labels to the neighbors of the support set samples.
        acc_all (_type_): The list of accuracies for all runs.

    Returns:
        _type_: List
    """

    # We use the support set only for label propagation.

    model = configs.get_the_model()

    classifier = configs.head_target_self_attention
    classifier.eval()

    with torch.no_grad():
        scores_list = []
        features_list = []

        # We add the support set for more accurate label propagation.
        if configs.use_LGC:
            dataset_support = TensorDataset(x_support)

            dataloader_support = DataLoader(dataset_support,
                                            batch_size=configs.batch_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=0)

            for x, in dataloader_support:
                features = model(Conditions.single_self_attention, {'x': x.to(configs.device)})
                features_list.append(features)

            if configs.one_hot_logits_for_support_set:
                logits = F.one_hot(y_support.to(configs.device), num_classes=configs.n_ways).float()
                scores_list.append(logits)
            else:
                if classifier is not None:
                    scores_list.append(classifier(concatenate_features(features, configs.n_estimators_classifier)))

        ds_temp = TensorDataset(x_query)
        dataloader_temp = DataLoader(ds_temp,
                                     batch_size=configs.batch_size,
                                     shuffle=False,
                                     drop_last=False)
        # query_size = configs.n_ways * configs.n_queries
        for x, in dataloader_temp:
            features, logits = forward_self_attention(configs, model, classifier, x)
            features_list.append(features)
            if classifier is not None:
                scores_list.append(logits)

    # Without a classifier, we just rely on the label propagation algorithm (for ablation studies)
    if classifier is None:
        logits = torch.zeros((len(x_query), configs.n_ways), dtype=torch.float).to(configs.device)
        scores_list.append(logits)
    scores = torch.cat(scores_list, dim=0)
    features = torch.cat(features_list, dim=0)
    if configs.use_LGC:
        with torch.no_grad():
            lgc = LGC()
            scores = lgc.compute(features, scores, configs.logger)
        scores = scores[len(y_support):]    # We have to remove the support set after label propagation
    # y_query = torch.arange(configs.n_ways).repeat_interleave(configs.n_queries).to(configs.device)
    y_query = np.repeat(range(configs.n_ways), configs.n_queries)
    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()

    top1_correct = np.sum(topk_ind[:, 0] == y_query)
    correct_this, count_this = float(top1_correct), len(y_query)
    acc_all.append((correct_this / count_this * 100))
    return acc_all


def concatenate_features(features, n_times):
    lst = []
    for _ in range(n_times):
        lst.append(features)
    return torch.cat(lst, dim=-1)


# We follow the implementation of https://github.com/cpphoo/STARTUP
# FSL phase in the paper.
def fsl(configs: ConfigurationTraining):
    """We can train a classifier on top of the frozen model with the target samples. We can also fine-tune the
     model and a classifier on top of that with the target samples."""
    # We only have access to the target domain
    model = configs.get_the_model()
    model.zero_grad()

    configs.set_seed()
    dataloader_episodic = loader_episodic(configs)

    acc_all = []
    sw = Stopwatch(['epoch', 'episode', 'display'])

    # n_epochs = 500  # if configs.freeze_backbone else 50
    # batch_size = 4  # min(configs.batch_size, support_size)        # In STARTUP they set it to 4
    next_time = configs.display_interval * 60

    iter_episode = -1

    for x_episode, y_episode in tqdm(dataloader_episodic, position=0, leave=False, desc='Episode'):
        # y_episode.shape is (n_ways, n_shots + n_queries)
        iter_episode += 1
        sw.reset('episode')
        assert len(y_episode.unique()) == configs.n_ways, \
            f"Error: The algorithm might have a flaw because it does not cover exactly {configs.n_ways} classes!"

        y_support = torch.arange(configs.n_ways).repeat_interleave(configs.n_shots).cpu()
        # x_support.shape is (n_ways * n_shots, n_channels, image_size, image_size)
        x_support = x_episode[:, :configs.n_shots, :, :, :].contiguous().view(configs.n_ways * configs.n_shots,
                                                                              *x_episode.size()[2:]).cpu()
        # x_query.shape is (n_ways * n_queries, n_channels, image_size, image_size)
        x_query = x_episode[:, configs.n_shots:, :, :, :].contiguous().view(configs.n_ways * configs.n_queries,
                                                                            *x_episode.size()[2:]).cpu()
        
        # We reset the model for every episode!
        configs.reset_model()
        model = configs.get_the_model()

        train_classifier_head(configs, x_support, y_support)

        if configs.LP_with_support_set:
            acc_all = evaluate_on_query_set_LP_with_support_set(configs, x_query, x_support, y_support, acc_all)
        else:
            acc_all = evaluate_on_query_set_LP_wihtout_support_set(configs, x_query, acc_all)

        if sw.elapsed_time('display') >= next_time or (iter_episode + 1) % configs.display_freq == 0:
            next_time += configs.display_interval * 60
            calculate_and_display_accuracy(configs, acc_all)

    calculate_and_display_accuracy(configs, acc_all)
    configs.logger.info("\nFSL phase is done!")


def train_classifier_head(configs: ConfigurationTraining,
                          x_support: torch.Tensor,
                          y_support: torch.Tensor,
                          tqdm_position: int = 1):
    """Trains a new head_target_self_attention and the model (if not freezed) with the support set samples.

    Args:
        configs (ConfigurationTraining): _description_
        x_support (torch.Tensor): Samples from the support set
        y_support (torch.Tensor): Labels from the support set
    """
    
    dim_inp = configs.config_model.embed_dim * configs.n_estimators_classifier

    configs.head_target_self_attention = MLP_V(configs,
                                               dim_inp,
                                               4 * dim_inp,
                                               configs.n_ways)

    model = configs.get_the_model()
    classifier = configs.head_target_self_attention
    classifier.train()
    params = list(classifier.parameters())

    if configs.freeze_backbone:
        model.eval()
        features_list = []
        ds_temp = TensorDataset(x_support, y_support)
        dataloader_temp = DataLoader(ds_temp,
                                     batch_size=configs.batch_size_training_classifier_head,
                                     shuffle=False,
                                     drop_last=False)
        with torch.no_grad():
            for x, _ in dataloader_temp:
                features = model(Conditions.single_self_attention, {'x': x.to(configs.device)})
                features_list.append(concatenate_features(features, configs.n_estimators_classifier))
        f_support = torch.cat(features_list, dim=0)
        # Features as inputs
        dataset_support = TensorDataset(f_support.cpu(), y_support.cpu())
    else:
        model.train()
        model.zero_grad()
        params += list(model.parameters())
        # Inputs to be given to the model during the training!
        dataset_support = TensorDataset(x_support.cpu(), y_support.cpu())

    optimizer = torch.optim.SGD(params, lr=configs.lr_training_classifier_head, momentum=configs.momentum, dampening=configs.dampening, weight_decay=configs.weight_decay_training_classifier_head)

    dataloader_support = DataLoader(dataset_support, batch_size=configs.batch_size_training_classifier_head, shuffle=True, drop_last=False)

    ce_loss = nn.CrossEntropyLoss().to(configs.device)  # This is \mathcal{L}_{S} for the initialization phase of FSL in our paper.

    for _ in tqdm(range(configs.n_epochs_training_classifier_head), position=1, leave=False, desc='Epoch'):
        # sw.reset('epoch')
        for iteration, (x_batch, y_batch) in enumerate(dataloader_support):
            # y_batch.shape is (n_ways)
            # x_episode.shape is (n_ways, n_shots + n_queries, n_channels, image_size, image_size)

            x_batch = x_batch.to(configs.device)
            y_batch = y_batch.to(configs.device)

            if configs.freeze_backbone:
                logits = classifier(x_batch)    # We computed the outputs before
            else:
                features, logits = forward_self_attention(configs, model, classifier, x_batch)
            loss = ce_loss(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def forward_self_attention(configs: ConfigurationTraining, model, classifier, x: torch.Tensor):
    features = model(Conditions.single_self_attention, {'x': x.to(configs.device)})
    logits = None
    if classifier is not None:
        logits = classifier(concatenate_features(features, configs.n_estimators_classifier))
    return features, logits


# Derived from the FSL method
def evaluation_only_label_propagation(configs: ConfigurationTraining):
    """
    In this step, we only rely on label propagation to assign labels to unlabeled samples in the target domain.
    """

    model = configs.get_the_model()
    model.zero_grad()

    configs.set_seed()
    dataloader_episodic = loader_episodic(configs)

    configs.use_LGC = True
    configs.logger.info(f"\nuse_LGC = {configs.use_LGC}")

    acc_all = []
    sw = Stopwatch(['epoch', 'episode', 'display'])

    next_time = configs.display_interval * 60

    iter_episode = -1

    for x_episode, y_episode in tqdm(dataloader_episodic, position=0, leave=False, desc='Episode'):
        # y_episode.shape is (n_ways, n_shots + n_queries)
        iter_episode += 1
        sw.reset('episode')
        assert len(y_episode.unique()) == configs.n_ways, \
            f"Error: The algorithm might have a flaw because it does not cover exactly {configs.n_ways} classes!"

        y_support = torch.arange(configs.n_ways).repeat_interleave(configs.n_shots).to(configs.device)
        # x_support.shape is (n_ways * n_shots, n_channels, image_size, image_size)
        x_support = x_episode[:, :configs.n_shots, :, :, :].contiguous().view(configs.n_ways * configs.n_shots,
                                                                              *x_episode.size()[2:]).to(configs.device)
        # x_query.shape is (n_ways * n_queries, n_channels, image_size, image_size)
        x_query = x_episode[:, configs.n_shots:, :, :, :].contiguous().view(configs.n_ways * configs.n_queries,
                                                                            *x_episode.size()[2:]).to(configs.device)

        # We reset the model for every episode!
        configs.reset_model()
        model.eval()

        configs.head_target_self_attention = None
        acc_all = evaluate_on_query_set_LP_with_support_set(configs, x_query, x_support, y_support, acc_all)

        if sw.elapsed_time('display') >= next_time or (iter_episode + 1) % configs.display_freq == 0:
            next_time += configs.display_interval * 60
            calculate_and_display_accuracy(configs, acc_all)

    calculate_and_display_accuracy(configs, acc_all)
    configs.logger.info("\nEvaluation is done!")


# We modified the code from https://github.com/facebookresearch/dino
def train_DINO(configs: ConfigurationTraining):
    configs.set_seed()

    if configs.domain_dino == 'base':
        data_loader, _ = load_base_dataset_DINO(configs)
    elif configs.domain_dino == 'target':
        data_loader, _ = load_target_dataset_for_DINO(configs)
    else:
        raise Exception(f"\"{configs.domain_dino}\" domain is not specified!")

    student = configs.get_the_model()
    teacher = configs.get_the_teacher()

    # teacher and student start with the same weights
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    configs.logger.info("Student and Teacher are built.")

    # ============ preparing loss ... ============
    dino_loss = dino_utils.DINOLoss(
        configs.out_dim,
        configs.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        configs.warmup_teacher_temp,
        configs.teacher_temp,
        configs.warmup_teacher_temp_epochs,
        configs.n_epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = dino_utils.get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)
    configs.load_optimizer(optimizer, None)

    # ============ init schedulers ... ============
    lr_scheduler = dino_utils.cosine_scheduler(
        configs.learning_rate * configs.batch_size / 256.,  # linear scaling rule
        configs.min_lr,
        configs.n_epochs, len(data_loader),
        warmup_epochs=configs.warmup_epochs,
    )

    configs.load_optimizer(optimizer)

    wd_schedule = dino_utils.cosine_scheduler(
        configs.weight_decay,
        configs.weight_decay_end,
        configs.n_epochs, len(data_loader),
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = dino_utils.cosine_scheduler(configs.momentum_teacher, 1,
                                                    configs.n_epochs, len(data_loader))
    configs.logger.info("Loss, optimizer and schedulers ready.")

    if configs.reproducible:
        start_epoch = 0
    else:
        start_epoch = configs.epoch

    sw = Stopwatch(['saved', 'epoch', 'total'])
    meters = AverageMeterSet_STARTUP()

    configs.logger.info("Starting DINO training !")

    def display_losses(log_stats: dict):
        for k, v in log_stats.items():
            configs.logger.info(f"{k} = {v}")

    for epoch in range(start_epoch, configs.n_epochs):
        sw.reset('epoch')
        # data_loader.sampler.set_epoch(epoch)
        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch_DINO(student, teacher, dino_loss,
                                           data_loader, optimizer, lr_scheduler, wd_schedule, momentum_schedule,
                                           epoch, configs)

        if epoch + 1 == configs.n_epochs or sw.elapsed_time('saved') >= configs.time_interval_to_save * 60:
            configs.save(epoch + 1, 0, optimizer, lr_scheduler=None)
            sw.reset('saved')
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        displayed = False
        if epoch + 1 == configs.n_epochs or sw.elapsed_time('saved') >= configs.time_interval_to_save * 60:
            configs.save(epoch + 1, 0, optimizer, lr_scheduler=None)
            display_losses(log_stats)
            displayed = True
            sw.reset('saved')

        # Save a snapshot
        if (epoch + 1) % configs.save_freq_epoch == 0:
            configs.save(epoch + 1, 0, optimizer, lr_scheduler=None, snapshot=True)
            display_losses(log_stats)
            displayed = True
            sw.reset('saved')
        if not displayed:
            display_losses(log_stats)
        meters.reset()
        configs.logger.info('This epoch took %s.', sw.elapsed_time_in_hours_minutes('epoch'))
        configs.logger.info('The training took %s from the beginning.', sw.elapsed_time_in_hours_minutes('total'))
    tm = sw.elapsed_time('total')
    configs.logger.info('DINO training took %s', sw.convert_to_hours_minutes(tm))


# We used the code from https://github.com/facebookresearch/dino
def train_one_epoch_DINO(student, teacher, dino_loss, data_loader,
                         optimizer, lr_scheduler, wd_schedule, momentum_schedule, epoch, configs):
    metric_logger = dino_utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, configs.n_epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_scheduler[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
        student_output = student(images)
        loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            configs.logger.info(f"Loss is {loss.item():.4f}, stopping training")
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        # param_norms = None
        loss.backward()
        if args.clip_grad:
            dino_utils.clip_gradients(student, args.clip_grad)  # returns param_norms
        dino_utils.cancel_gradients_last_layer(epoch, student,
                                               args.freeze_last_layer)
        optimizer.step()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# DINO with cross-attentions
# We modified the code from https://github.com/facebookresearch/dino
def train_two_domains_DINO(configs: ConfigurationTraining):
    configs.set_seed()

    data_loader_base, transform_base = load_base_dataset_DINO(configs)
    data_loader_target, transform_target = load_target_dataset_for_DINO(configs)

    assert len(data_loader_base) >= len(data_loader_target)

    student = configs.get_the_model()
    teacher = configs.get_the_teacher()

    # teacher and student start with the same weights
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    configs.logger.info("Student and Teacher are built.")

    # ============ preparing loss ... ============
    dino_loss_base = dino_utils.DINOLoss(
        configs.out_dim,
        configs.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        configs.warmup_teacher_temp,
        configs.teacher_temp,
        configs.warmup_teacher_temp_epochs,
        configs.n_epochs,
    ).cuda()

    dino_loss_target = dino_utils.DINOLoss(
        configs.out_dim,
        configs.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        configs.warmup_teacher_temp,
        configs.teacher_temp,
        configs.warmup_teacher_temp_epochs,
        configs.n_epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = dino_utils.get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)
    configs.load_optimizer(optimizer, None)

    # ============ init schedulers ... ============
    # Since the base dataset has more samples, we calculate the learning rate in the scheduler based on the epoch in
    #  the base dataset.
    niter_per_ep = max(len(data_loader_base), len(data_loader_target))

    lr_scheduler = dino_utils.cosine_scheduler(
        configs.learning_rate * configs.batch_size / 256.,  # linear scaling rule
        configs.min_lr,
        configs.n_epochs, niter_per_ep,
        warmup_epochs=configs.warmup_epochs,
    )

    wd_schedule = dino_utils.cosine_scheduler(
        configs.weight_decay,
        configs.weight_decay_end,
        configs.n_epochs, niter_per_ep,
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = dino_utils.cosine_scheduler(configs.momentum_teacher, 1,
                                                    configs.n_epochs, niter_per_ep)
    configs.logger.info("Loss, optimizer and schedulers ready.")

    if configs.reproducible:
        start_epoch = 0
    else:
        start_epoch = configs.epoch

    sw = Stopwatch(['saved', 'epoch', 'total'])
    meters = MovingAverageSet(configs.display_freq)

    configs.logger.info("Starting DINO training !")

    def display(meters, epoch):
        configs.logger.info('-' * 80)
        configs.logger.info(f"Epoch={epoch}")
        meters.display(configs.logger)

    for epoch in range(start_epoch, configs.n_epochs):
        sw.reset('epoch')
        # data_loader.sampler.set_epoch(epoch)
        # ============ training one epoch of DINO ... ============
        # meters, sw = \

        # For reproducibility, we skipped the required number of epochs. Now we revert the transform functions.
        if epoch == configs.epoch:
            configs.logger.info("Dino transforms are enabled!")
            data_loader_base.dataset.d.transform = transform_base
            data_loader_target.dataset.d.transform = transform_target

        train_two_domains_one_epoch_DINO(student, teacher, dino_loss_base, dino_loss_target,
                                         data_loader_base, data_loader_target,
                                         optimizer, lr_scheduler,
                                         wd_schedule, momentum_schedule,
                                         epoch, configs, meters, sw)

        displayed = False
        if epoch + 1 == configs.n_epochs or sw.elapsed_time('saved') >= configs.time_interval_to_save * 60:
            configs.save(epoch + 1, 0, optimizer, lr_scheduler=None)
            display(meters, epoch + 1)
            displayed = True
            sw.reset('saved')

        # Save a snapshot
        if (epoch + 1) % configs.save_freq_epoch == 0:
            configs.save(epoch + 1, 0, optimizer, lr_scheduler=None, snapshot=True)
            display(meters, epoch + 1)
            displayed = True
            sw.reset('saved')
        if not displayed:
            display(meters, epoch + 1)
        configs.logger.info('This epoch took %s.', sw.elapsed_time_in_hours_minutes('epoch'))
        configs.logger.info('The training took %s from the beginning.', sw.elapsed_time_in_hours_minutes('total'))
    configs.logger.info('DINO training took %s.', sw.elapsed_time_in_hours_minutes('total'))


# We used the code from https://github.com/facebookresearch/dino
def train_two_domains_one_epoch_DINO(student, teacher, dino_loss_base, dino_loss_target, data_loader_base,
                                     data_loader_target,
                                     optimizer, lr_scheduler, wd_schedule, momentum_schedule, epoch, configs, meters,
                                     sw):
    iter_base = iter(data_loader_base)
    iter_target = iter(data_loader_target)
    if len(data_loader_base) < len(data_loader_target):
        raise NotImplementedError

    n_iters = max(len(data_loader_base), len(data_loader_target))

    for it in range(n_iters):
        try:
            images_base, _ = next(iter_base)
        except StopIteration:
            configs.logger.warning("We should not have reached here! 389794165")
            iter_base = iter(data_loader_base)
            images_base, _ = next(iter_base)
        try:
            images_target, _ = next(iter_target)
        except StopIteration:
            iter_target = iter(data_loader_target)
            images_target, _ = next(iter_target)

        # update weight decay and learning rate according to their schedule
        it_total = len(data_loader_base) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_scheduler[it_total]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it_total]

        # move images to gpu
        images_base = [im.cuda(non_blocking=True) for im in images_base]
        images_target = [im.cuda(non_blocking=True) for im in images_target]
        # teacher and student forward passes + compute dino loss
        teacher_output_base, teacher_output_target = teacher(images_base[:2], images_target[:2])  # only the 2 global views pass through the teacher
        student_output_base, student_output_target = student(images_base, images_target)
        loss_base = dino_loss_base(student_output_base, teacher_output_base, epoch)
        loss_target = dino_loss_target(student_output_target, teacher_output_target, epoch)
        loss = loss_base + loss_target

        if not math.isfinite(loss.item()):
            configs.logger.info(f"Loss is {loss.item():.4f}, stopping training")
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        # param_norms = None
        loss.backward()
        if args.clip_grad:
            dino_utils.clip_gradients(student, args.clip_grad)    # returns param_norms
        dino_utils.cancel_gradients_last_layer(epoch, student,
                                               args.freeze_last_layer)
        optimizer.step()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it_total]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        meters.update(loss=loss.item())
        meters.update(loss_base=loss_base.item())
        meters.update(loss_target=loss_target.item())
        meters.update(lr=optimizer.param_groups[0]["lr"])
        meters.update(wd=optimizer.param_groups[0]["weight_decay"])

        if sw.elapsed_time('display') >= configs.display_interval * 60:
            sw.reset('display')
            print('-' * 80)
            print(f"Iteration: {it}/{n_iters}")
            meters.display()


def evaluate_accuracy_base_dataset(configs: ConfigurationTraining):
    model = configs.get_the_model()

    model.eval()

    dataloader_base = load_base_dataset_teacher(configs, configs.batch_size, False)
    configs.logger.info(f"num_classes_base = {configs.num_classes_base}")

    assert configs.num_classes_base > 0, 'Error: Wrong number of classes for the base classifier head'

    configs.head_base_self_attention = nn.Linear(configs.config_model.embed_dim, configs.num_classes_base).to(configs.device)

    configs.reset_head_base_self_attention(configs.device)
    configs.head_base_self_attention.eval()

    configs.logger.info('-' * 80)
    configs.logger.info("Evaluating the accuracy of the network on the base domain ...")

    num_correct = 0
    count = 0

    iteration = -1
    for x_base, labels in tqdm(dataloader_base, position=1, leave=False, desc='Iteration'):
        iteration += 1
        inp = {'x': x_base.to(configs.device)}
        labels = labels.to(configs.device)
        with torch.no_grad():
            features, logits = forward_self_attention(configs, model, configs.head_base_self_attention, inp)
            output_labels = logits.argmax(dim=1)
        assert len(output_labels) == len(labels)
        num_correct += torch.sum(1 * (labels == output_labels)).item()
        count += len(labels)
        if (iteration + 1) % configs.display_freq == 0:
            print(f"\nAccuracy: {100.0 * (num_correct / count):.4f}")

    print()
    configs.logger.info(f"Accuracy: {100.0 * (num_correct / count):.4f}")


def supervised_one_domain(configs: ConfigurationTraining, dataset=None):
    """
    We train the network on a single dataset with the self-attention.
    :param configs:
    :param dataset: The default dataset is the base dataset (Mini-ImageNet) when it is None.
    :return:
    """
    configs.set_seed()

    model = configs.get_the_model()

    model.train()
    model.zero_grad()

    ce_loss = nn.CrossEntropyLoss().to(configs.device)

    if dataset is None:
        dataloader_base = load_base_dataset_teacher(configs, configs.batch_size, configs.aug_train)
        configs.logger.info(f"num_classes_base = {configs.num_classes_base}")
        assert configs.num_classes_base > 0, 'Error: Wrong number of classes for the base classifier head'
    else:
        dataloader_base = DataLoader(dataset,
                                     batch_size=configs.batch_size,
                                     num_workers=configs.num_workers,
                                     shuffle=True,
                                     drop_last=True)
    configs.set_seed()
    configs.head_base_self_attention = nn.Linear(configs.config_model.embed_dim, configs.num_classes_base).to(configs.device)

    configs.reset_head_base_self_attention(configs.device)

    configs.head_base_self_attention.train()
    configs.head_base_self_attention.zero_grad()

    optimizer = torch.optim.SGD(list(model.parameters()) + list(configs.head_base_self_attention.parameters()),
                                lr=configs.learning_rate,
                                momentum=configs.momentum,
                                weight_decay=configs.weight_decay)

    lr_scheduler = ReduceLROnPlateau(optimizer,
                                     mode='min', factor=0.5,
                                     patience=10, verbose=False,
                                     cooldown=10,
                                     threshold_mode='rel',
                                     threshold=1e-4, min_lr=1e-5)

    configs.load_optimizer(optimizer, lr_scheduler)

    configs.logger.info('-' * 80)
    configs.logger.info("Training the network on the base domain with softmax and cross-entropy ...")

    def display_losses(epoch, optimizer, meters):
        configs.logger.info('\nEpoch: %d', epoch)
        configs.logger.info("lr: %f", optimizer.param_groups[0]['lr'])
        meters.display(configs.logger)
        configs.logger.info('This epoch took %s.', sw.elapsed_time_in_hours_minutes('epoch'))
        configs.logger.info('%s alapsed from the beginning', sw.elapsed_time_in_hours_minutes('total'))

    meters = MovingAverageSet(configs.display_freq)
    sw = Stopwatch(['saved', 'epoch', 'total'])

    start_epoch = configs.epoch

    if configs.epoch == 0:
        configs.save(0, 0, optimizer, lr_scheduler, snapshot=True)

    next_time = configs.time_interval_to_save * 60

    for epoch in tqdm(range(start_epoch, configs.n_epochs), position=1, leave=False, desc='Epoch'):
        sw.reset('epoch')
        iteration = -1
        for x_base, labels in tqdm(dataloader_base, position=2, leave=False, desc='Iteration'):
            iteration += 1
            _, logits = forward_self_attention(configs, model, configs.head_base_self_attention, x_base)
            labels = labels.to(configs.device)
            loss = ce_loss(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            meters.update(loss=loss.item())

        loss_was_displayed = False

        # We save the model at the end or after an iteration after making sure that at least a certain amount of
        # time passed from the last time which we saved the model!
        if epoch + 1 == configs.n_epochs or sw.elapsed_time('saved') >= next_time:
            next_time += configs.time_interval_to_save * 60
            configs.save(epoch + 1, 0, optimizer, lr_scheduler)
            display_losses(epoch, optimizer, meters)
            loss_was_displayed = True

        # Save a snapshot
        if (epoch + 1) % configs.save_freq_epoch == 0 or sw.elapsed_time('saved') >= next_time:
            next_time += configs.time_interval_to_save * 60
            configs.save(epoch + 1, 0, optimizer, lr_scheduler, snapshot=True)
            if not loss_was_displayed:
                display_losses(epoch, optimizer, meters)
                loss_was_displayed = True

        if not loss_was_displayed:
            # tm = sw.elapsed_time('epoch')
            display_losses(epoch, optimizer, meters)
            # print(f'This epoch took ' + sw.convert_to_hours_minutes(tm) + '!')
        if dataset is None and ((epoch + 1) % 10 == 0 or (epoch + 1) == configs.n_epochs):
            evaluate_accuracy_base_dataset(configs)
        lr_scheduler.step(meters['loss'].calculate())

    configs.save(configs.n_epochs + 1, 0, optimizer, lr_scheduler)
    configs.logger.info("\nThe process is finished!")
    configs.logger.info(f"The whole process took {sw.convert_to_hours_minutes(sw.elapsed_time('total'))}.")
    configs.head_base_self_attention = configs.head_base_self_attention.to('cpu')


# Fine-tuning phase in the ablation studies
def supervised_base_and_support_set(configs: ConfigurationTraining):
    """
    It fine-tunes a previously trained network in a supervised manner with the base dataset and support set
    """
    model = configs.get_the_model()
    model.zero_grad()

    configs.set_seed()
    dataloader_episodic = loader_episodic(configs)

    if configs.epoch > 0:
        configs.logger.info(f"The network was previously trained for {configs.epoch} epochs.")
    elif configs.iteration > 0:
        configs.logger.info(f"The network was previously trained for {configs.iteration} iterations.")

    configs.logger.info(f"\nuse_LGC = {configs.use_LGC}")

    acc_all = []
    sw = Stopwatch(['epoch', 'episode', 'display'])

    # Default was n_epochs = 100
    if configs.n_shots == 1:
        lr_classifier = 0.01
        # n_epochs = 500
        weight_decay = 0.001
    else:
        lr_classifier = 0.01
        # n_epochs = 100
        weight_decay = 0.001

    lr = configs.learning_rate
    configs.logger.info(f'n_iterations is set to {configs.n_iterations}')
    batch_size = configs.batch_size
    configs.logger.info(f'Batch size = {batch_size}')
    configs.logger.info(f"n_iterations for the supervised step = {configs.n_iterations}")
    configs.logger.info(f"lr = {lr}")
    configs.logger.info(f"lr_classifier rate = {lr_classifier}")
    configs.logger.info(f"Weight decay = {weight_decay}")
    configs.logger.info(f"n_estimators_classifier = {configs.n_estimators_classifier}")
    configs.logger.info(f"n_layers_classifier = {configs.n_layers_classifier}")
    next_time = configs.display_interval * 60

    dataloader_base = load_base_dataset_teacher(configs, batch_size, configs.aug_train)
    iter_base = iter(dataloader_base)
    configs.logger.info(f"num_classes_base = {configs.num_classes_base}")
    assert configs.num_classes_base > 0, 'Error: Wrong number of classes for the base classifier head'

    iter_episode = -1

    if configs.aug_train:
        transform = miniImageNet_few_shot.TransformLoader(pair(configs.image_size)).get_composed_transform(configs.aug_train)
    else:
        transform = None

    for x_episode, y_episode in tqdm(dataloader_episodic, position=0, leave=False, desc='Episode'):
        # y_episode.shape is (n_ways, n_shots + n_queries)
        iter_episode += 1
        sw.reset('episode')
        assert len(y_episode.unique()) == configs.n_ways, \
            f"Error: The algorithm might have a flaw because it does not cover exactly {configs.n_ways} classes!"

        y_support = torch.arange(configs.n_ways).repeat_interleave(configs.n_shots).to(configs.device)
        # x_support.shape is (n_ways * n_shots, n_channels, image_size, image_size)
        x_support = x_episode[:, :configs.n_shots, :, :, :].contiguous().view(configs.n_ways * configs.n_shots,
                                                                              *x_episode.size()[2:]).to(configs.device)
        # x_query.shape is (n_ways * n_queries, n_channels, image_size, image_size)
        x_query = x_episode[:, configs.n_shots:, :, :, :].contiguous().view(configs.n_ways * configs.n_queries,
                                                                            *x_episode.size()[2:]).to(configs.device)

        # We reset the model for every episode!
        configs.reset_model()

        dim_inp = 2 * configs.config_model.embed_dim * configs.n_estimators_classifier

        if configs.n_shots == 1:
            classifier_base = MLP_V(configs,
                                    dim_inp,
                                    3 * dim_inp,
                                    configs.num_classes_base)
            classifier_target = MLP_V(configs,
                                      dim_inp,
                                      3 * dim_inp,
                                      configs.n_ways)
        else:
            classifier_base = MLP_V(configs,
                                    dim_inp,
                                    dim_inp,
                                    configs.num_classes_base)
            classifier_target = MLP_V(configs,
                                      dim_inp,
                                      dim_inp,
                                      configs.n_ways)

        classifier_base.train()
        classifier_base.zero_grad()
        classifier_target.train()
        classifier_target.zero_grad()

        model = configs.get_the_model()
        model.train()
        model.zero_grad()
        params = list(classifier_target.parameters()) + list(classifier_base.parameters()) + list(model.parameters())
        # Inputs to be given to the model during the training!

        dataset_support = DatasetFromTensors(x_support.cpu(), y_support.cpu(), transform=transform, ToPILImage=True)
        dataloader_support = DataLoader(dataset_support,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        drop_last=True)
        iter_target = iter(dataloader_support)

        ce_loss = nn.CrossEntropyLoss().to(configs.device)

        optimizer = torch.optim.SGD(params, lr=lr, momentum=configs.momentum, dampening=configs.dampening, weight_decay=weight_decay)

        # 1- we train the model on both domains with linear layers
        for _ in tqdm(range(configs.n_iterations), position=1, leave=False, desc='Iter.'):
            try:
                x_base, y_base = next(iter_base)
            except StopIteration:
                iter_base = iter(dataloader_base)
                x_base, y_base = next(iter_base)

            try:
                x_support, y_support = next(iter_target)
            except StopIteration:
                iter_target = iter(dataloader_support)
                x_support, y_support = next(iter_target)

            x_base = x_base.to(configs.device)
            y_base = y_base.to(configs.device)
            x_support = x_support.to(configs.device)
            y_support = y_support.to(configs.device)

            base_features, target_base_features, target_features, base_target_features = model(Conditions.all_attention_blocks,
                                                                                               {'base': x_base,
                                                                                                'target': x_support})
            features_base_dominant = torch.cat([base_features, target_base_features], dim=-1)
            logits_base = classifier_base(concatenate_features(features_base_dominant, configs.n_estimators_classifier))
            features_target_dominant = torch.cat([target_features, base_target_features], dim=-1)
            logits_target = classifier_target(concatenate_features(features_target_dominant, configs.n_estimators_classifier))
            # logits_base = classifier_base(target_base_features)
            # logits_target = classifier_target(base_target_features)

            loss_base = ce_loss(logits_base, y_base)
            loss_target = ce_loss(logits_target, y_support)
            loss = loss_base + loss_target
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        if configs.LP_with_support_set:
            acc_all = evaluate_on_query_set_LP_with_support_set(configs, classifier_target, x_query, x_support, y_support, acc_all)
        else:
            acc_all = evaluate_on_query_set_LP_wihtout_support_set(configs, model, classifier_target, x_query, acc_all)

        if sw.elapsed_time('display') >= next_time or (iter_episode + 1) % configs.display_freq == 0:
            next_time += configs.display_interval * 60
            calculate_and_display_accuracy(configs, acc_all)

    calculate_and_display_accuracy(configs, acc_all)
    configs.logger.info("\nThe process is finished!")


phase_str_to_func = {'fsl': fsl,
                     'train_DINO': train_DINO,
                     'train_two_domains_DINO': train_two_domains_DINO,
                     'evaluate_accuracy_base_dataset': evaluate_accuracy_base_dataset,
                     'supervised_one_domain': supervised_one_domain,
                     'evaluation_only_label_propagation': evaluation_only_label_propagation,
                     'supervised_base_and_support_set': supervised_base_and_support_set
                     }


def main(args):
    configs = ConfigurationTraining(args)
    configs.set_seed()
    # cudnn.benchmark = True
    phase_str_to_func[configs.phase](configs)
    return 0


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
