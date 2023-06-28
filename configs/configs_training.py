# You don't need to set anything here. We set the variables according to the arguments

import utils.dino_utils as dino_utils
import torch
import numpy as np
from configs.configs_model import string_to_model_config
from models.ViT_CCT import ViT_CCT, PositionalEmbeddingType
import random
import os
from copy import deepcopy
import gc
from utils.other_utilities import parse_str_bool, get_logger, get_time_str


class ConfigurationTraining:
    def __init__(self, args):
        self.args = args

        # To help the autocomplete capability of the editor
        self.seed = None
        self.reproducible = None
        self.aug_train = None
        self.phase = None
        self.batch_size = None
        self.force_cpu = None
        self.n_epochs = None
        self.image_size = None
        self.n_iterations = None
        self.n_ways = None
        self.n_shots = None
        self.n_episodes = None
        self.n_queries = None
        self.use_LGC = None
        self.LP_with_support_set = None
        self.one_hot_logits_for_support_set = None
        self.num_workers = None

        # Settings for training the classifier head
        self.batch_size_training_classifier_head = None
        self.dropout_rate_classifier_head = None
        self.n_epochs_training_classifier_head = None
        self.n_layers_classifier = None
        self.n_estimators_classifier = None
        self.inference_use_pretrained_head = None
        self.lr_training_classifier_head = None
        self.weight_decay_training_classifier_head = None

        self.display_freq = None

        self.learning_rate = None

        self.freeze_backbone = None

        self.log_file = None

        # Split files
        self.base_path = None
        self.base_split = None
        self.target_path = None
        self.subset_split = None
        self.target_subset_split = None
        self.base_val_ratio = None

        self.momentum = None
        self.dampening = None
        self.weight_decay = None

        self.num_classes_base = None
        self.num_classes_target = None

        self.base_dataset = None
        self.target_dataset = None

        # Configuration of the model
        self.model_type = None

        self.time_interval_to_save = None
        self.display_interval = None
        self.save_freq_epoch = None
        self.save_freq_iter = None

        self.in_channels = 0

        # DINO
        self.domain_dino = None
        self.out_dim = None
        self.norm_last_layer = None
        self.local_crops_number = None
        self.local_crops_scale = None
        self.global_crops_scale = None
        self.warmup_teacher_temp = None
        self.teacher_temp = None
        self.warmup_teacher_temp_epochs = None
        self.weight_decay_end = None
        self.min_lr = None
        self.warmup_epochs = None
        self.momentum_teacher = None
        self.clip_grad = None
        self.freeze_last_layer = None
        self.use_bn_in_head = None

        # Set the values from the args
        for k, v in args._get_kwargs():
            setattr(self, k, v)

        # If we don't require reproducibility, instead of skipping the same samples, we just change the seed.
        if not self.reproducible:
            self.seed += 1

        self.set_seed()

        self._n_epochs_real = self.n_epochs           # When we process a single dataset
        self._n_iterations_real = self.n_iterations   # When we process two datasets

        # For resuming the experiment
        self.epoch = 0
        self.iteration = 0

        self._output_file = args.output_file
        delattr(self, 'output_file')
        self._input_file = args.input_file
        delattr(self, 'input_file')
        # We calculate the relative path of the files in the specified save directory.
        self._experiment_name = args.experiment_name
        delattr(self, 'experiment_name')

        self.logger = get_logger(f"{self.log_file}_{get_time_str()}.txt")

        # The settings of the learning rate scheduler
        self.lr_scheduler = None

        # self.lr_decay_type = args.lr_decay_type
        # self.warmup_steps = args.warmup_steps

        # Configuration of the model
        self.config_model = string_to_model_config[self.model_type]

        self.config_model.log_file = self.log_file

        self.config_dataset_target_episodic = None

        self._loader_episodic = None

        self.is_cuda_available()

        # self.n_gpu = torch.cuda.device_count()

        self._logger_level_saved = None

        self._state_optimizer = None
        self._state_lr_scheduler = None
        self._state_head_base_self_attention = None
        self._state_head_target_self_attention = None
        self._state_head_base_with_cross_attention = None
        self._state_head_target_with_cross_attention = None
        # We will assign this linear head during the first step (on the base domain)

        self.head_base_self_attention = None
        self.head_target_self_attention = None
        # We will set it during the last step (fine-tuning)
        self.head_base_with_cross_attention = None
        self.head_target_with_cross_attention = None

        self.model = None
        self.teacher = None
        self._state_teacher = None
        self._state_model = None

        self.load()
        if self.force_cpu:
            if self.batch_size == 0:
                msg = 'Error: You should set the batch size.'
                self.logger.exception(msg)
                raise Exception(msg)
            self.device = 'cpu'
        gc.collect()
        # self.logger.info(f"Batch size = {self.batch_size}")
        self.print_arguments()

    def print_arguments(self):
        attributes = self.__dict__
        ignore_set = {'args', 'log_file'}
        useful_types = {int, float, str, list, bool, tuple}

        message = "The given arguments"
        temp = 80 - 2 - len(message) // 2   # :)
 
        self.logger.info('-' * temp + ' ' + message + ' ' + '-' * temp)

        for name in attributes.keys():
            value = getattr(self, name)
            type_attr = type(value)
            if name.startswith("_") or name in ignore_set or type_attr not in useful_types or value is None:
                continue

            if name == 'epoch':
                self.logger.info("The network was previously trained for %d epochs." % self.epoch)
            elif name == 'iteration':
                self.logger.info("The network was previously trained for %d iterations." % self.iteration)
            elif type_attr != str:
                self.logger.info("%s = %s" % (name, str(getattr(self, name))))
            else:
                self.logger.info('%s = "%s"' % (name, getattr(self, name)))
        
        self.logger.info('-' * 80)
        print('Log file\'s path = "%s"' % self.log_file)

    def is_cuda_available(self):
        """
        It checks if the cuda is available, then devices for different configurations to the result.
        :return:
        """
        self.device = torch.device(self.device) if torch.cuda.is_available() else 'cpu'

        return torch.cuda.is_available()

    def initialize_model(self, use_wrapper_for_DINO: bool, norm_last_layer=None, pretrained_model=None):
        if pretrained_model is None:
            model = ViT_CCT(config=self.config_model,
                            image_size=self.image_size,
                            in_channels=self.in_channels,
                            pos_embedding_type=PositionalEmbeddingType.SinCos
                            )
        else:   # For DINO, we should wrap the pretrained model here!
            model = pretrained_model

        if use_wrapper_for_DINO:
            assert norm_last_layer is not None
            if self.phase == 'train_DINO':
                model = dino_utils.MultiCropWrapper_self_attn(model,
                                                              dino_utils.DINOHead(
                                                                  self.config_model.embed_dim,
                                                                  self.out_dim,
                                                                  use_bn=False,
                                                                  norm_last_layer=norm_last_layer,
                                                              )
                                                              )
            elif self.phase == 'train_two_domains_DINO':
                model = dino_utils.MultiCropWrapper_quadruple(model,
                                                              dino_utils.DINOHead(
                                                                  2 * self.config_model.embed_dim,
                                                                  self.out_dim,
                                                                  use_bn=self.use_bn_in_head,
                                                                  norm_last_layer=norm_last_layer,
                                                              ),
                                                              dino_utils.DINOHead(
                                                                  2 * self.config_model.embed_dim,
                                                                  self.out_dim,
                                                                  use_bn=self.use_bn_in_head,
                                                                  norm_last_layer=norm_last_layer,
                                                              )
                                                              )
            else:
                raise "Not defined!"
        return model

    def get_the_model(self):
        self.model.to(self.device)
        return self.model

    def get_the_teacher(self):
        self.teacher.to(self.device)
        return self.teacher

    # Set seed for reproducibility
    def set_seed(self, another_seed=None):
        """It set the seed to predefined seed if another seed is not provided. Otherwise, it considers the new seed.

        Args:
            another_seed (_type_, optional): A new seed instead of previously define seed. Defaults to None.
        """
        if another_seed is None:
            another_seed = self.seed
        # torch.use_deterministic_algorithms(True)
        np.random.seed(another_seed)
        random.seed(another_seed)
        torch.random.manual_seed(another_seed)
        random.seed(another_seed)
        torch.manual_seed(another_seed)
        torch.cuda.manual_seed(another_seed)
        torch.cuda.manual_seed_all(another_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Saving and loading is inspired from https://github.com/SHI-Labs/Compact-Transformers
    def save(self, epoch: int,
             iteration: int,
             optimizer=None,
             lr_scheduler=None,
             snapshot=False):
        self.epoch = epoch
        self.iteration = iteration

        state = {'model': deepcopy(self.model.state_dict()),
                 'model_type': self.model_type}

        if self.teacher is not None:
            state['teacher'] = deepcopy(self.teacher.state_dict())
        # if metric is not None:
        #     state['metric'] = metric
        state['epoch'] = epoch
        state['iteration'] = iteration
        if optimizer is not None:
            state['optimizer'] = deepcopy(optimizer.state_dict())

        if lr_scheduler is not None:
            state['lr_scheduler'] = deepcopy(lr_scheduler.state_dict())

        if self.head_base_self_attention is not None:
            self._state_head_base_self_attention = deepcopy(self.head_base_self_attention.state_dict())
        if self.head_target_self_attention is not None:
            self._state_head_target_self_attention = deepcopy(self.head_target_self_attention.state_dict())
        if self.head_base_with_cross_attention is not None:
            self._state_head_base_with_cross_attention = deepcopy(self.head_base_with_cross_attention.state_dict())

        if self.head_target_with_cross_attention is not None:
            self._state_head_target_with_cross_attention = deepcopy(self.head_target_with_cross_attention.state_dict())

        if self._state_head_base_self_attention is not None:
            state['head_base_self_attention'] = self._state_head_base_self_attention
        if self._state_head_target_self_attention is not None:
            state['head_target_self_attention'] = self._state_head_target_self_attention
        if self._state_head_base_with_cross_attention is not None:
            state['head_base_with_cross_attention'] = self._state_head_base_with_cross_attention
        if self._state_head_target_with_cross_attention is not None:
            state['head_target_with_cross_attention'] = self._state_head_target_with_cross_attention

        # state['num_classes_base'] = self.num_classes_base
        # state['num_classes_target'] = self.num_classes_target

        state['batch_size'] = self.batch_size

        state['learning_rate'] = self.learning_rate

        state['phase'] = self.phase

        torch.save(state, self._output_file)

        if snapshot:
            path = self.args.output_file
            if epoch > 0:
                path += f"-Epoch={self.epoch}"
            elif iteration > 0:
                path += f"-Iter={self.iteration}"

            path += ".zip"

            if path != self._output_file:
                torch.save(state, path)
                self.logger.info('\nA snapshot is saved! <--------')
        else:
            print()
            self.logger.info('\nThe model is saved! <--------')

    def load(self, key: str = None):
        """
        Loads the last state from the disk. If the process finished before, it will return true. Some the states of
        some PyTorch modules are saved in the variables to be called later
        :param key: If we need just the value for a specific key. We skip from loading the other parts.
        :return:
        """
        # We try to resume the process from the last save file. Otherwise, we resume from the saved input file.
        self._input_file += '.zip'
        self._output_file += '.zip'

        if os.path.exists(self._output_file) and os.path.isfile(self._output_file):
            file_path = self._output_file
        else:
            file_path = self._input_file

        if os.path.exists(file_path):
            state = torch.load(file_path, map_location="cpu")

            # If the phase has not changed, we resume the process, otherwise we ignore some settings.
            same_phase = (self.phase == state['phase'])

            if 'epoch' in state and (same_phase or
                                     self.phase == 'fsl' or
                                     self.phase == 'evaluation_only_label_propagation' or
                                     self.phase == 'supervised_base_and_support_set'):
                self.epoch = state['epoch']
            if 'iteration' in state and (same_phase or
                                         self.phase == 'fsl' or
                                         self.phase == 'evaluation_only_label_propagation' or
                                         self.phase == 'supervised_base_and_support_set'):
                self.iteration = state['iteration']

            # When we load from the output, we resume if we haven't reached the n_epochs or n_iterations
            if file_path == self._output_file and (self.epoch >= self.n_epochs or self.iteration > self.n_iterations) \
                    and self.phase != "fsl" and self.phase != 'evaluation_only_label_propagation':
                self.logger.warning(f"This process was done before. We skip the {self.phase} phase!")
                exit(0)

            # # When we load from the input file, we ignore the epochs or iterations from the previous step
            # if file_path == self._input_file:
            #     self.epoch = 0
            #     self.iteration = 0

            if key is not None:
                if key in state:
                    return state[key]
                else:
                    msg = f'The key {key} does not exist in the save file \"{file_path}\"!'
                    self.logger.exception(msg)
                    raise Exception(msg)

            dino = (self.phase == 'train_DINO' or self.phase == 'train_two_domains_DINO')   # or 'teacher' in state

            if 'model' in state:
                self.model_type = state['model_type']
                if dino:
                    if state['phase'] == 'train_DINO' or state['phase'] == 'train_two_domains_DINO':  # We load a model which is trained with DINO
                        self.model = self.initialize_model(True, self.norm_last_layer)
                        self._state_model = state['model']
                        self.model.load_state_dict(self._state_model)

                        # We also have the teacher model's weights in the saved file
                        self._state_teacher = state['teacher']
                    else:   # We should wrap the model and add a head for DINO
                        pretrained_model = self.initialize_model(False)
                        pretrained_model.load_state_dict(state['model'])
                        self.model = self.initialize_model(True,
                                                           self.norm_last_layer,
                                                           pretrained_model=pretrained_model)
                        self._state_model = deepcopy(self.model.state_dict())
                        self._state_teacher = deepcopy(self.model.state_dict())

                    self.teacher = self.initialize_model(True, False)
                    self.teacher.load_state_dict(self._state_teacher)

                else:
                    if 'teacher' in state:  # If we trained the model with DINO in the previous step
                        # When we do not continue the training of DINO, we load the pretrained weights of the teacher.
                        state_dict = state['teacher']
                        # remove `module.` prefix
                        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                        # remove `backbone.` prefix induced by multicrop wrapper
                        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                        self.model = self.initialize_model(dino, self.norm_last_layer)
                        msg = self.model.load_state_dict(state_dict, strict=False)
                        self._state_model = deepcopy(self.model.state_dict())
                        print('-' * 80)
                        print(f'--> The weights of the teacher model is loaded with msg: {msg}')
                    else:   # We are resuming from a saved model, which is not trained with DINO.
                        self._state_model = state['model']
                        self.model = self.initialize_model(False)
                        self.model.load_state_dict(self._state_model)
                        # We load the teacher again when we resume the training with DINO

            # self.num_classes_base = state['num_classes_base']
            # self.num_classes_target = state['num_classes_target']

            # If the user suggested a learning rate, the app uses that lr.
            if self.learning_rate == 0 and state['learning_rate'] != 0:
                self.learning_rate = state['learning_rate']
            if not same_phase:    # We started from the save file of another phase
                self.epoch = 0
                self.iteration = 0

            if 'head_base_self_attention' in state:
                self._state_head_base_self_attention = state['head_base_self_attention']

            if 'head_target_self_attention' in state:
                self._state_head_target_self_attention = state['head_target_self_attention']

            if 'head_base_with_cross_attention' in state:
                self._state_head_base_with_cross_attention = state['head_base_with_cross_attention']

            if 'head_target_with_cross_attention' in state:
                self._state_head_target_with_cross_attention = state['head_target_with_cross_attention']

            if same_phase and 'optimizer' in state:
                self._state_optimizer = state['optimizer']
            if 'lr_scheduler' in state:
                self._state_lr_scheduler = state['lr_scheduler']

            if 'batch_size' in state and same_phase:
                if self.batch_size <= 0:
                    self.batch_size = state['batch_size']
                    self.logger.info(f"We use the same batch size = {state['batch_size']} that we had used before.")
                else:
                    if self.batch_size == state['batch_size']:
                        self.logger.info(f"We use the same batch size = {self.batch_size } that we had used before.")
                    else:
                        self.logger.info(f"The batch size is changed to {self.batch_size }.")

            self.logger.info(f"File \"{file_path}\" is loaded")
        else:
            self.logger.info('No save file is loaded!')
            ans = parse_str_bool(input("Do you want to train a model from scratch? "))
            if not ans:
                exit(0)

            dino = (self.phase == 'train_DINO' or self.phase == 'train_two_domains_DINO')  # or 'teacher' in state
            self.logger.warning("We are starting to train a model from scratch!")
            self.model = self.initialize_model(dino, self.norm_last_layer)
            self._state_model = deepcopy(self.model.state_dict())
            if dino:
                self.teacher = self.initialize_model(dino, False)
                self._state_teacher = deepcopy(self.model.state_dict())
                self.teacher.load_state_dict(self._state_teacher)

    def reset_model(self):
        if self._state_model is not None:
            self.model.load_state_dict(self._state_model, strict=True)
        else:
            msg = "Error: The model was not loaded!"
            self.logger.exception(msg)
            raise Exception(msg)

    def reset_teacher(self):
        if self._state_teacher is not None:
            self.teacher.load_state_dict(self._state_teacher)
        else:
            msg = "Error: The teacher model was not loaded!"
            self.logger.exception(msg)
            raise Exception(msg)

    def load_optimizer(self, optimizer, lr_scheduler=None):
        if self._state_optimizer is not None:
            optimizer.load_state_dict(self._state_optimizer)

        if self._state_lr_scheduler is not None and \
                lr_scheduler is not None and \
                self._state_lr_scheduler is not None:
            lr_scheduler.load_state_dict(self._state_lr_scheduler)

    def reset_head_base_self_attention(self, device=None):
        if self.head_base_self_attention is not None:
            if self._state_head_base_self_attention is not None:
                self.head_base_self_attention.load_state_dict(self._state_head_base_self_attention)
            if device is not None:
                self.head_base_self_attention = self.head_base_self_attention.to(device)

    def reset_head_target_self_attention(self, device=None):
        if self.head_target_self_attention is not None:
            if self._state_head_target_self_attention is not None:
                self.head_target_self_attention.load_state_dict(self._state_head_target_self_attention)
            if device is not None:
                self.head_target_self_attention = self.head_target_self_attention.to(device)

    def save_state_head_base_self_attention(self):
        self._state_head_base_self_attention = deepcopy(self.head_base_self_attention.state_dict())

    def save_state_head_target_self_attention(self):
        self._state_head_target_self_attention = deepcopy(self.head_target_self_attention.state_dict())

    def save_state_head_base_with_cross_attention(self):
        self._state_head_base_with_cross_attention = deepcopy(self.head_base_with_cross_attention.state_dict())

    def save_state_head_target_with_cross_attention(self):
        self._state_head_target_with_cross_attention = deepcopy(self.head_target_with_cross_attention.state_dict())

    def reset_head_base_with_cross_attention(self, device=None):
        if self.head_base_with_cross_attention is not None:
            if self._state_head_base_with_cross_attention is not None:
                self.head_base_with_cross_attention.load_state_dict(self._state_head_base_with_cross_attention)
            if device is not None:
                self.head_base_with_cross_attention = self.head_base_with_cross_attention.to(device)

    def reset_head_target_with_cross_attention(self, device=None):
        if self.head_target_with_cross_attention is not None:
            if self._state_head_target_with_cross_attention is not None:
                self.head_target_with_cross_attention.load_state_dict(self._state_head_target_with_cross_attention)
            if device is not None:
                self.head_target_with_cross_attention = self.head_target_with_cross_attention.to(device)

    def reset_all(self):
        # We should save the model and heads before fine-tuning!
        self.reset_model()
        self.reset_head_base_self_attention()
        self.reset_head_target_self_attention()
        self.reset_head_base_with_cross_attention()
        self.reset_head_target_with_cross_attention()
