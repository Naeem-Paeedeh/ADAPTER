import argparse
import torch
import time
import numpy as np
from torchvision import transforms as T
from torch.utils.data import Dataset
import logging


class DatasetFromTensors(Dataset):
    def __init__(self, inputs: torch.Tensor, labels: torch.Tensor, transform=None, ToPILImage: bool = False):
        """_summary_

        Args:
            inputs (torch.Tensor): All samples
            labels (torch.Tensor): All labels
            transform (_type_, optional): tranfroms and augmentations. Defaults to None.
            ToPILImage (bool, optional): If we need ToPILImage applied before other transforms. Defaults to True.
        """
        super().__init__()
        self.data = inputs
        # self.data = F.to_pil_image(inputs.numpy())
        self.labels = labels
        self.n_samples = len(self.labels)  # self.labels.shape[0]
        self.index = 0
        self.transform = transform
        self.ToPILImage = ToPILImage

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        if self.transform is not None:
            if self.ToPILImage:
                sample = T.ToPILImage()(sample)
            sample = self.transform(sample)

        return sample, label


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# This calculation is less likely to fail when we have an almost ill-conditioned matrix.
# @article{paeedeh2021improving,
#   title={Improving the backpropagation algorithm with consequentialism weight updates over mini-batches},
#   author={Paeedeh, Naeem and Ghiasi-Shirazi, Kamaledin},
#   journal={Neurocomputing},
#   volume={461},
#   pages={86--98},
#   year={2021},
#   publisher={Elsevier}
# }
def pinv_robust(x, coef, device):
    dim1 = x.size()[0]
    dim2 = x.size()[1]
    x = x.to(device)

    if dim1 >= dim2:
        cor = x.t() @ x
        idn = coef * torch.eye(dim2, dim2).to(device)
        nv = (cor + idn).inverse() @ x.t()
    else:
        cor = x @ x.t()
        idn = coef * torch.eye(dim1, dim1).to(device)
        nv = x.t() @ (cor + idn).inverse()
    return nv


class Stopwatch:
    """
    Stopwatch computes the time between start and stop.
    Then we can add time to the total_elapsed_time dictionary by watch name.
    """
    def __init__(self, keys=None):
        if keys is None:
            keys = []
        self._start_time = {k: time.time() for k in keys}

    def reset(self, key):
        self._start_time[key] = time.time()

    def elapsed_time(self, key):
        if key in self._start_time:
            return time.time() - self._start_time[key]

        self.reset(key)
        return 0.0

    @staticmethod
    def convert_to_hours_minutes(time_in_seconds: float) -> str:
        time_in_seconds = int(time_in_seconds)
        hours = time_in_seconds // 3600
        minutes = (time_in_seconds % 3600) // 60
        seconds = time_in_seconds % 60
        if hours > 0:
            return f"{hours} hours and {minutes} minutes and {seconds} seconds"
        elif minutes > 0:
            return f"{minutes} minutes and {seconds} seconds"
        else:
            return f"{seconds} seconds"

    def elapsed_time_in_hours_minutes(self, key):
        return self.convert_to_hours_minutes(self.elapsed_time(key))

    def __getitem__(self, name):
        return self.elapsed_time(name)

    def __getattr__(self, name: str):
        return self.elapsed_time(name)


class MovingAverageSet:
    def __init__(self, capacity):
        self.capacity = capacity
        self.meters: dict[str, _MovingAverage] = {}

    def __getitem__(self, key):
        if key in self.meters:
            return self.meters[key]
        raise Exception(f"Error: You didn't define or add a value to the {key} key!")

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))

            if k not in self.meters:
                self.meters[k] = _MovingAverage(self.capacity)
            self.meters[k].update(v)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def display(self, logger=None):
        for key, val in self.meters.items():
            output = f"Average {key} for the last {val.count} iterations: {val.calculate()}"
            if logger is None:
                print(output)
            else:
                logger.info(output)


class _MovingAverage:
    def __init__(self, capacity):
        self.capacity = capacity
        self.array = np.zeros(self.capacity)
        self.ind = 0
        self.count = 0
        self.sum = 0.0

    def update(self, x):
        self.sum += x - self.array[self.ind]
        self.array[self.ind] = x
        self.ind = (self.ind + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def calculate(self):
        return self.sum / self.count

    def reset(self):
        self.array = np.zeros(self.capacity)
        self.ind = 0
        self.count = 0
        self.sum = 0


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def parse_str_bool(x):
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x in ('yes', 'y', 'true', 't', '1', 'on'):
        return True
    
    if x in ('no', 'n', 'false', 'f', '0', 'off'):
        return False
    
    msg = 'Boolean value is expected!'
    # logger.exception(msg)
    raise argparse.ArgumentTypeError(msg)


# From the https://github.com/cpphoo/STARTUP
class AverageMeterSet_STARTUP:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))

            if k not in self.meters:
                self.meters[k] = AverageMeter_STARTUP()
            self.meters[k].update(v)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.value for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.average for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


# From the https://github.com/cpphoo/STARTUP
class AverageMeter_STARTUP:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        """
        value is the average value
        n : the number of items used to calculate the average
        """
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count

    # def __format__(self, format):
    #     return "{self.value:{format}} ({self.average:{format}})".format(self=self, format=format)


# From the https://github.com/cpphoo/STARTUP
def accuracy_STARTUP(logits, ground_truth, top_k=None):
    if top_k is None:
        top_k = [1, ]
    assert len(logits) == len(ground_truth)
    # this function will calculate per class acc
    # average per class acc and acc

    n, d = logits.shape

    label_unique = torch.unique(ground_truth)
    acc = {'average': torch.zeros(len(top_k)),
           'per_class_average': torch.zeros(len(top_k)),
           'per_class': [[] for _ in label_unique],
           'gt_unique': label_unique,
           'topk': top_k,
           'num_classes': d}

    max_k = max(top_k)
    argsort = torch.argsort(logits, dim=1, descending=True)[:, :min([max_k, d])]
    correct = (argsort == ground_truth.view(-1, 1)).float()

    for indi, i in enumerate(label_unique):
        ind = torch.nonzero(ground_truth == i, as_tuple=False).view(-1)
        correct_target = correct[ind]

        # calculate topk
        for indj, j in enumerate(top_k):
            num_correct_partial = torch.sum(correct_target[:, :j]).item()
            acc_partial = num_correct_partial / len(correct_target)
            acc['average'][indj] += num_correct_partial
            acc['per_class_average'][indj] += acc_partial
            acc['per_class'][indi].append(acc_partial * 100)

    acc['average'] = acc['average'] / n * 100
    acc['per_class_average'] = acc['per_class_average'] / len(label_unique) * 100

    return acc


# We mostly follow the name conventions of the STARTUP and TVT for argument names.
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', '-o', type=str, default='', required=False,
                        help='The absolute path of the file to save. ')
    parser.add_argument('--input_file', '-i', type=str, default='', required=False,
                        help='The absolute path to the file to resume from.')
    parser.add_argument('--experiment_name', type=str, default='',
                        help='The name of experiment which we add to the save files.')
    parser.add_argument('--log_file', '-l', type=str, default='Log.txt',
                        help='The absolute path to the file to save the reports of the experiments.')
    parser.add_argument("-reproducible", type=parse_str_bool, default=True,
                        help="Enable it for reproducibility in resuming. It will increase the computations.")

    parser.add_argument('-device', type=str, default='cuda',
                        help="""The CUDA device.""")

    parser.add_argument("-aug_train", type=parse_str_bool, default=True,
                        help="Perform data augmentation or not during the training")

    parser.add_argument('-force_cpu', type=parse_str_bool, default=False,
                        help='Force to use the CPU for all computations.')

    parser.add_argument('-time_interval_to_save', type=float, default=10,
                        help='Time threshold in minutes to save a checkpoint after finishing an epoch or an iteration')

    parser.add_argument('--load_path', type=str, default=None,
                        help='Path to the checkpoint to be loaded')
    parser.add_argument('--resume_latest', action='store_true',
                        help='resume from the latest model in args.dir')
    parser.add_argument('-seed', type=int, default=1,
                        help='Seed for randomness')
    parser.add_argument('-num_workers', type=int, default=4,
                        help='Number of workers for dataloader')

    parser.add_argument('-in_channels', type=int, default=3,
                        help='Number of channels in the input images')

    parser.add_argument('-n_ways', type=int, default=5,
                        help='Number of classes in few-shot learning')
    parser.add_argument('-n_shots', type=int, default=5,
                        help='Number of labeled samples in each class in few-shot learning')
    parser.add_argument('-n_episodes', type=int, default=600,
                        help='Number of episodes in few-shot learning')
    parser.add_argument('-n_queries', type=int, default=15,
                        help='Number of query examples in few-shot learning')

    parser.add_argument('-use_LGC', type=parse_str_bool, default=True,
                        help='Utilize label propagation algorithm')
    parser.add_argument('-LP_with_support_set', type=parse_str_bool, default=False,
                        help='Utilize the support set to improve the accuracy of the label propagation.')
    parser.add_argument('-one_hot_logits_for_support_set', type=parse_str_bool, default=False,
                        help='One-hot logits for support set for label propagation algorithm.')

    parser.add_argument('--base_dataset', '-b_ds', type=str,
                        choices=['miniImageNet', 'tieredImageNet', 'ChestX', 'ISIC', 'EuroSAT', 'CropDisease'], default="miniImageNet",
                        required=False, help='The dataset for the base domain.')
    parser.add_argument('-base_path', type=str, required=False,
                        help='Path to the base dataset')
    parser.add_argument('-base_split', type=str,
                        help='split for the base dataset')
    parser.add_argument('-base_val_ratio', type=float, default=0.05,
                        help='amount of base dataset set aside for validation')

    parser.add_argument('--target_dataset', '-t_ds', type=str,
                        choices=['miniImageNet', 'ChestX', 'ISIC', 'EuroSAT', 'CropDisease'],
                        default='ChestX', required=False,
                        help='The dataset for the target domain')
    parser.add_argument('-target_path', type=str, required=False,
                        help='Path to the target dataset')
    parser.add_argument('-target_subset_split', type=str,
                        help='path to the csv files that specifies the unlabeled split for the target dataset')
    parser.add_argument('-subset_split', type=str,
                        help='path to the csv files that contains the split of the data')
    parser.add_argument('-num_classes_base', default=-1, type=int,
                        help='Number of classes for the base dataset')
    parser.add_argument('-num_classes_target', default=-1, type=int,
                        help='Number of classes for the target dataset')

    parser.add_argument("-image_size", type=int, default=224,
                        help="Resolution size")

    parser.add_argument("--model_type", choices=["CCT-7/3x1", 'ViT-B/16', 'CCT-7/5x1-Modified', 'CCT-14/7x2'],
                        default='CCT-7/5x1-Modified', help="Which variant to use.")

    parser.add_argument("-batch_size", type=int, default=256,
                        help="The batch size for normal training.")

    parser.add_argument('-n_epochs', type=int, default=50,      # default=400  # 10 for debugging
                        help='Number of training epochs')
    # 400*500=200000 iterations also might be a good value!
    parser.add_argument('-n_iterations', type=int, default=200000,     # Or 50000
                        help='Number of training iterations for domain adaptation')
    parser.add_argument('-display_freq', type=int, default=200,
                        help='After how many iterations we want to see the statistics')
    parser.add_argument('-display_interval', type=float, default=1,
                        help='Time threshold in minutes to display the evaluation results')
    
    parser.add_argument("--learning_rate", '-lr', type=float, default=0,
                        help="The suggested learning rate.")
    
    # ----- The specific artuments for training the classifier head -------------------------------
    parser.add_argument("-lr_training_classifier_head", type=float, default=0.01,
                        help="The suggested learning rate.")
    
    parser.add_argument('-weight_decay_training_classifier_head', type=float, default=1e-5,
                        help='Weight decay for the training the classifier head')
    
    parser.add_argument("-batch_size_training_classifier_head", type=int, default=4,
                        help="The batch size for training the classifier head.")
    
    parser.add_argument("-n_epochs_training_classifier_head", type=int, default=100,
                        help="The number of epochs for training the classifier head.")
    
    parser.add_argument('-dropout_rate_classifier_head', type=float, default=0.0,
                        help='The dropout rate for training the classifier head.')
    
    parser.add_argument("-n_layers_classifier", type=int, default=2,
                        help="The number of layers of the classifier for inference.")

    parser.add_argument("-n_estimators_classifier", type=int, default=2,
                        help="With this argument, we can define the number of estimators in few-shot learning")
    # ---------------------------------------------------------------------------------------------

    parser.add_argument('-save_freq_epoch', type=int, default=10,
                        help='Frequency (in epoch) to save')
    parser.add_argument('-save_freq_iter', type=int, default=2000,
                        help='Frequency (in iteration) to save')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='Frequency (in epoch) to evaluate on the val set')

    parser.add_argument('--print_freq', type=int, default=5,
                        help='Frequency (in step per epoch) to print training stats')

    # parser.add_argument('-drop_path', type=float, default=-1,
    #                     help='Drop path rate (default: -1 -> default in the program)')

    parser.add_argument("-momentum", type=float, default=0.9,
                        help="The momentum.")
    parser.add_argument("-dampening", type=float, default=0.9,
                        help="The dampening for the SGD optimizer.")

    parser.add_argument("--phase", '-p', type=str, choices=['fsl',
                                                            'fine_tuning_with_target_dominant_head',
                                                            'evaluate_accuracy_base_dataset',
                                                            'train_DINO',
                                                            'train_two_domains_DINO',
                                                            'supervised_one_domain',
                                                            'evaluation_only_label_propagation',
                                                            'supervised_base_and_support_set'
                                                            ],
                        help="what do we want to do?")

    parser.add_argument("-freeze_backbone", type=parse_str_bool, default=True,
                        help="Freezing the backbone during the training")

    parser.add_argument("-inference_use_pretrained_head", type=parse_str_bool, default=False,
                        help="We can use the head which is trained before in other phases.")

    # parser.add_argument("--lr_decay_type", choices=["cosine", "linear"], default="cosine",
    #                     help="How to decay the learning rate.")
    # parser.add_argument("--warmup_steps", type=int, default=500,
    #                     help="Step of training_phase to perform learning rate warmup for.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float,
    #                     help="Max gradient norm.")

    # parser.add_argument("--local_rank", type=int, default=-1,
    #                     help="local_rank for distributed training_phase on gpus")
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
    #                     help="Number of updates steps to accumulate before performing a backward/update pass.")
    # -----------------------
    # DINO
    parser.add_argument('-domain_dino', type=str, choices=['base', 'target'],
                        help="""Specify the domain to train with DINO.""")
    parser.add_argument('-out_dim', default=65536, type=int, help="""Dimensionality of
                the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('-norm_last_layer', default=True, type=parse_str_bool,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
                Not normalizing leads to better performance but can make the training unstable.
                In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('-local_crops_number', type=int, default=8, help="""Number of small
                local views to generate. Set this parameter to 0 to disable multi-crop training.
                When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('-local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
                Used for small local view cropping of multi-crop.""")
    # Temperature teacher parameters
    parser.add_argument('-warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
            Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('-teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
            of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
            starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('-warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-5,
                        help='Weight decay for the model')
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
            weight decay. We use a cosine schedule for WD and using a larger decay by
            the end of training improves performance for ViTs.""")
    parser.add_argument('-min_lr', type=float, default=1e-6, help="""Target LR at the
            end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument("-warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    # Multi-crop parameters
    parser.add_argument('-global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
            Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
            recommend using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('-momentum_teacher', default=0.996, type=float, help="""Base EMA
            parameter for teacher update. The value is increased to 1 during training with cosine schedule.
            We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('-clip_grad', type=float, default=3.0, help="""Maximal parameter
            gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
            help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('-freeze_last_layer', default=1, type=int, help="""Number of epochs
            during which we keep the output layer fixed. Typically doing so during
            the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument('-use_bn_in_head', default=False, type=parse_str_bool,
                        help="Whether to use batch normalizations in projection head (Default: False)")
    # parser.add_argument("-checkpoint_key", default="teacher", type=str,
    #                     help='Key to use in the checkpoint (example: "teacher")')
    # ---------------------------------

    args = parser.parse_args()
    return args


# We modified these codes from F2M (https://github.com/moukamisama/f2m) {
def get_logger(log_file,
               logger_name='ADAPTER',
               log_level=logging.INFO):
   
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized before.
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s:%(levelname)s:%(name)s:%(lineno)d:%(message)s'

    logging.basicConfig(format=format_str, level=log_level, force=True)
    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(logging.Formatter(format_str))
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)

    return logger


def get_time_str():
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

# }
