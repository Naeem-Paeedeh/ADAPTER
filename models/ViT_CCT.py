# We used some codes from TVT, CDTrans, and STARTUP papers, and the
# timm (https://github.com/rwightman/pytorch-image-models), https://github.com/lucidrains/vit-pytorch repositories,
# and https://github.com/asrafulashiq/dynamic-cdfsl.

# The implementation of both ViT and CCT are merged.

import math
import torch
import torch.nn.functional as F
from torch import Tensor as T
import torch.nn as nn
from torch.nn import Dropout, Linear
import einops
from utils.pos_embed import get_2d_sincos_pos_embed
from configs.configs_model import ConfigurationModel
import logging
import utils.dino_utils as dino_utils
from enum import Enum
from utils.other_utilities import pair


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatterVar = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(lineno)d:%(message)s')
streamHandler = logging.StreamHandler()
# streamHandler.setFormatter(formatterVar)
logger.addHandler(streamHandler)


class Conditions(Enum):
    all_attention_blocks = 1
    single_self_attention = 2


class PositionalEmbeddingType(Enum):
    Learnable = 1
    SinCos = 2
    # Disabled = 0    # We might test if for Compact-Transformers


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)

    @staticmethod
    def drop_path(x, drop_prob: float = 0., training: bool = False):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class PatchEmbedding(nn.Module):
    def __init__(self, config: ConfigurationModel, img_size: int, in_channels: int, in_planes: int = 64):  # in_channels=3
        super().__init__()
        self.img_size: tuple = pair(img_size)
        self.patch_size = pair(config.patch_size)
        self.num_patches = None

        self.proj = None

        n_filter_list = [in_channels] + \
                        [in_planes for _ in range(config.n_conv_layers - 1)] + \
                        [config.embed_dim]

        if config.net_type.is_cct():
            self.proj = nn.Sequential(
                *[nn.Sequential(
                    nn.Conv2d(in_channels=n_filter_list[i],
                              out_channels=n_filter_list[i + 1],
                              kernel_size=self.patch_size,
                              stride=config.stride,
                              padding=config.padding,
                              bias=False),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=config.pooling_kernel_size,
                                 stride=config.pooling_stride,
                                 padding=config.pooling_padding)
                ) for i in range(config.n_conv_layers)]
            )
        else:
            self.proj = nn.Conv2d(in_channels=in_channels,
                                  out_channels=config.embed_dim,
                                  kernel_size=self.patch_size,
                                  stride=config.stride,
                                  padding=config.padding)

        # For both CCT and ViT
        self.num_patches = self.forward(torch.zeros((1, in_channels, self.img_size[0], self.img_size[1]))).shape[1]
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x: T) -> T:
        # x.shape is (n_samples, num_channels, image_size, image_size)
        _, _, height, width = x.shape

        x = self.proj(x)  # -> batch_size x embed_dim x n_patches_dim_1 x n_patches_dim_2
        x = x.flatten(2)  # -> batch_size x embed_dim x n_patches
        x = x.transpose(-1, -2)  # -> batch_size x n_patches x embed_dim
        return x  # batch_size x n_patches x embed_dim


class MLP(nn.Module):
    def __init__(self, config: ConfigurationModel):
        super().__init__()
        self.fc1 = Linear(config.embed_dim, config.transformer_mlp_dim)
        self.fc2 = Linear(config.transformer_mlp_dim, config.embed_dim)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, config: ConfigurationModel):
        super().__init__()
        self.config = config
        self.num_heads = config.transformer_num_heads
        self.embed_dim = config.embed_dim
        self.head_dim = self.embed_dim // self.num_heads

        if self.embed_dim % self.num_heads != 0:
            msg = "Error: Embedding size should be divisible by the number of heads!"
            logger.exception(msg)
            raise Exception(msg)

        self.scale = self.head_dim ** -0.5

        self.query = None
        self.key = None
        self.value = None
        self.query_cross = None
        self.key_cross = None
        self.value_cross = None

        self.query = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.key = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.value = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)

        self.attn_dropout = nn.Dropout(config.attention_dropout_rate)

        self.proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Dropout(config.dropout_rate)
        )

        self.proj_cross = None

    def transpose_for_scores(self, x: T) -> T:
        # x.shape is (batch_size, seq_length, embed_dim) for ViT
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        # new shape is (batch_size, seq_length, self.num_heads, self.head_dim) for ViT
        x = x.view(*new_x_shape)
        # The shape of returned tensor is (batch_size, self.num_heads, seq_length, self.head_dim) for ViT
        return x.permute(0, 2, 1, 3)

    def forward_quadruple(self, inp: dict):
        x_base = inp['base']
        x_target = inp['target']
        # x_base.shape is (batch_size, seq_length, embed_dim) for ViT
        num_samples, num_patches_and_cls, num_channels = x_target.shape

        # For ViT, the shapes after the linear layers will be (batch_size, seq_length, embed_dim)
        q_base =    self.transpose_for_scores(self.query(x_base))
        k_base =    self.transpose_for_scores(self.key(  x_base))
        v_base =    self.transpose_for_scores(self.value(x_base))
        q_target =  self.transpose_for_scores(self.query(x_target))
        k_target =  self.transpose_for_scores(self.key(  x_target))
        v_target =  self.transpose_for_scores(self.value(x_target))

        attn_base_prob = torch.softmax(q_base @ k_base.transpose(-2, -1) * self.scale, dim=-1)
        attn_target_prob = torch.softmax(q_target @ k_target.transpose(-2, -1) * self.scale, dim=-1)
        attn_base_target_prob = torch.softmax(q_base @ k_target.transpose(-2, -1) * self.scale, dim=-1)
        attn_target_base_prob = torch.softmax(q_target @ k_base.transpose(-2, -1) * self.scale, dim=-1)

        attn_base_prob = self.attn_dropout(attn_base_prob)
        attn_target_prob = self.attn_dropout(attn_target_prob)
        attn_base_target_prob = self.attn_dropout(attn_base_target_prob)
        attn_target_base_prob = self.attn_dropout(attn_target_base_prob)

        x_base = attn_base_prob @ v_base
        x_target = attn_target_prob @ v_target
        x_base_target = attn_base_target_prob @ v_target
        x_target_base = attn_target_base_prob @ v_base

        x_base = x_base.transpose(1, 2).reshape(num_samples, num_patches_and_cls, num_channels)
        x_target = x_target.transpose(1, 2).reshape(num_samples, num_patches_and_cls, num_channels)
        x_base_target = x_base_target.transpose(1, 2).reshape(num_samples, num_patches_and_cls, num_channels)
        x_target_base = x_target_base.transpose(1, 2).reshape(num_samples, num_patches_and_cls, num_channels)

        # Projection
        # outputs = {k: self.proj(x) for k, x in outputs.items()}
        x_base = self.proj(x_base)
        x_target = self.proj(x_target)
        x_base_target = self.proj(x_base_target)
        x_target_base = self.proj(x_target_base)

        outputs = {'base': x_base,
                   'target': x_target,
                   'base-target': x_base_target,
                   'target-base': x_target_base}

        return outputs

    def forward_without_cross_attention(self, inp: dict) -> T:
        x = inp['x']
        # x.shape is (batch_size, seq_length, embed_dim) for ViT
        num_samples, num_patches_and_cls, num_channels = x.shape

        q = self.query(x)   # q.shape after the linear layer is (batch_size, seq_length, embed_dim)
        k = self.key(x)     # k.shape after the linear layer is (batch_size, seq_length, embed_dim)
        v = self.value(x)   # v.shape after the linear layer is (batch_size, seq_length, embed_dim)

        q, k, v = list(map(self.transpose_for_scores, (q, k, v)))

        attn_prob = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)

        attn_prob = self.attn_dropout(attn_prob)

        x = (attn_prob @ v).transpose(1, 2).reshape(num_samples, num_patches_and_cls, num_channels)

        # Projection and dropout
        x = self.proj(x)

        return x

    def forward(self, cond: Conditions, inp: dict):
        # We compute the self-attention for the base and target inputs.
        # We also compute the cross-attention for base_to_target, and target_to_base.
        if cond == Conditions.single_self_attention:
            return {'x': self.forward_without_cross_attention(inp)}
        else:
            return self.forward_quadruple(inp)


class Block(nn.Module):
    def __init__(self, config: ConfigurationModel, drop_path_rate=0.1):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.norm_attn = nn.LayerNorm(self.embed_dim)
        self.attn = Attention(config)
        self.drop_path = \
            DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm_mlp = nn.LayerNorm(self.embed_dim)
        # mlp_hidden_dim = int(self.embed_dim * mlp_ratio)
        self.mlp = MLP(config)
        self.mlp_cross = None

    def forward(self, cond: Conditions, z: dict):
        z_norm = {}
        for k in z.keys():
            z_norm[k] = self.norm_attn(z[k])
        attn_dict = self.attn(cond, z_norm)
        # Residual connections, dropouts, norms, additions
        z_hat = {}
        for k in attn_dict.keys():
            z_hat[k] = z[k] + self.drop_path(attn_dict[k])
        z = {}
        for k in z_hat.keys():
            z[k] = z_hat[k] + self.drop_path(self.mlp(self.norm_mlp(z_hat[k])))
        return z


# ViT or CCT for Domain Adaptation with Quadruple Transformer Blocks
class ViT_CCT(nn.Module):
    """
    These are the specifications of the vanilla ViT, CCT and quadruple transformer blocks for Adapter.
    """
    def __init__(self, config: ConfigurationModel, image_size: int,
                 in_channels: int = 3, pos_embedding_type: PositionalEmbeddingType = PositionalEmbeddingType.SinCos):
        super().__init__()

        self.config = config
        self.in_channels = in_channels
        self.patch_embedding = PatchEmbedding(config, image_size, in_channels=in_channels)
        self.seq_length = self.patch_embedding.num_patches
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(formatterVar)
        logger.addHandler(file_handler)

        if config.net_type.is_vit():
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            self.seq_length += 1
        else:
            self.attention_pool = nn.Linear(config.embed_dim, 1)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.num_layers = config.num_layers

        # We follow the CCT implementation
        dpr = [x.item() for x in torch.linspace(0, config.stochastic_depth, config.num_layers)]

        self.blocks = nn.ModuleList([Block(config, drop_path_rate=dpr[i])
                                     for i in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.embed_dim)

        self.pos_embedding = None
        self.pos_embedding_type = pos_embedding_type

        self.initialize_positional_embeddings()

    def initialize_positional_embeddings(self):
        if self.pos_embedding_type is PositionalEmbeddingType.Learnable:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, self.seq_length, self.config.embed_dim),
                requires_grad=True)
        elif self.pos_embedding_type is PositionalEmbeddingType.SinCos:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, self.seq_length, self.config.embed_dim),
                requires_grad=False)

            # Positional embedding for the backbone
            pos_embed = get_2d_sincos_pos_embed(self.pos_embedding.shape[-1],
                                                int(self.patch_embedding.num_patches ** .5),
                                                cls_token=self.config.net_type.is_vit())
            self.pos_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            msg = "Error: Unknown positional embedding type!"
            logger.exception(msg)
            raise Exception(msg)

    def interpolate_pos_embedding(self, x, w, h):
        if x.shape[1] == self.pos_embedding.shape[1] and w == h:
            return self.pos_embedding

        dim = x.shape[-1]

        if self.config.net_type.is_cct():
            N = self.pos_embedding.shape[1]
            patch_pos_embed = self.pos_embedding
            scale = math.sqrt(x.shape[1] / self.pos_embedding.shape[1])
        else:   # For ViT
            N = self.pos_embedding.shape[1] - 1
            class_pos_embed = self.pos_embedding[:, 0]
            patch_pos_embed = self.pos_embedding[:, 1:]
            scale = math.sqrt((x.shape[1] - 1) / (self.pos_embedding.shape[1] - 1))

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(scale, scale),
            mode='bicubic',
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        if self.config.net_type.is_cct():
            return patch_pos_embed
        else:   # For ViT
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    # As a Vanilla ViT or CCT with single input
    def forward_single_domain_without_cross_attention(self, x: T):
        # x_base.shape is (n_samples, num_channels, image_size, image_size).

        n_samples, _, w, h = x.shape
        x = self.patch_embedding(x)

        if self.config.net_type.is_vit():
            # Replicating cls_token for all samples
            cls_tokens = einops.repeat(self.cls_token, '1 1 d -> n 1 d', n=n_samples)
            x = torch.cat((cls_tokens, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)

        # (n_samples, 1 + n_patches, embed_dim)
        # add positional encoding to each token
        x += self.interpolate_pos_embedding(x, w, h)

        x = self.dropout(x)

        x = {'x': x}
        for block in self.blocks:
            x = block(Conditions.single_self_attention, x)

        x = self.norm(x['x'])

        if self.config.net_type.is_cct():
            x = (F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2) @ x).squeeze(-2)
        else:
            x = x[:, 0]     # cls_token

        return x

    def forward_both_domains(self, x: dict):
        # x_base: T = self.bn1(x['base'])
        # x_target: T = self.bn1(x['target'])
        x_base: T = x['base']
        x_target: T = x['target']
        # x_base.shape is (n_samples, num_channels, image_size, image_size).

        # We assume that number of samples of x_base and x_target are equal.
        if x_base.shape[0] != x_target.shape[0]:
            msg = "Error: The base and target mini-batches should have the same number of samples"
            logger.exception(msg)
            raise Exception(msg)
        n_samples, _, w, h = x_base.shape

        x_base = self.patch_embedding(x_base)
        x_target = self.patch_embedding(x_target)

        if self.config.net_type.is_vit():
            # Replicating cls_token for all samples
            cls_tokens = einops.repeat(self.cls_token, '1 1 d -> n 1 d', n=n_samples)

            x_base = torch.cat((cls_tokens, x_base), dim=1)        # (n_samples, 1 + n_patches, embed_dim)
            x_target = torch.cat((cls_tokens, x_target), dim=1)  # (n_samples_target, 1 + n_patches, embed_dim)

        pos_embed_interpolated = self.interpolate_pos_embedding(x_base, w, h)
        x_base += pos_embed_interpolated       # (n_samples, 1 + n_patches, embed_dim)
        x_target += pos_embed_interpolated   # (n_samples_target, 1 + n_patches, embed_dim)

        x_base = self.dropout(x_base)
        x_target = self.dropout(x_target)

        x = {'base': x_base, 'target': x_target, 'base-target': x_target, 'target-base': x_base}

        for i, block in enumerate(self.blocks):
            x = block(Conditions.all_attention_blocks, x)

        for k in x.keys():
            x[k] = self.norm(x[k])
        # x = {k: self.norm(v) for k, v in x.items()}

        if self.config.net_type.is_cct():
            # base_features, target_features, base_target_features, target_base_features = \
            #     list(map(lambda inp: (F.softmax(self.attention_pool(inp), dim=1).transpose(-1, -2) @ inp).squeeze(-2),
            #              (x['base'], x['target'], x['base-target'], x['target-base'])
            #              ))
            base_features = (F.softmax(self.attention_pool(x['base']), dim=1).transpose(-1, -2) @ x['base']).squeeze(-2)
            target_features = (F.softmax(self.attention_pool(x['target']), dim=1).transpose(-1, -2) @ x['target']).squeeze(-2)
            base_target_features = (F.softmax(self.attention_pool(x['base-target']), dim=1).transpose(-1, -2) @ x['base-target']).squeeze(-2)
            target_base_features = (F.softmax(self.attention_pool(x['target-base']), dim=1).transpose(-1, -2) @ x['target-base']).squeeze(-2)

            # base_dominant_features = torch.cat((base_features, target_base_features), dim=-1)  # Z_hat_base
            # target_dominant_features = torch.cat((target_features, base_target_features), dim=-1)  # Z_hat_target
        else:   # cls_tokens are our features for ViT
            base_features = x['base'][:, 0]
            target_features = x['target'][:, 0]
            target_base_features = x['target-base'][:, 0]
            base_target_features = x['base-target'][:, 0]
            # base_dominant_features = torch.cat((base_features, target_base_features), dim=-1)
            # target_dominant_features = torch.cat((target_features, base_target_features), dim=-1)

        return base_features, target_base_features, target_features, base_target_features

    def forward(self, cond: Conditions, inp: dict):
        """

        :param cond:
        :param inp: The inputs as a dictionary.
        'x' is for the situation that we have single input.
        When we have two inputs, we call them 'base' and 'target'
        During the calculations, we will use base-target and target-base to return values
        :return:
        """
        # x_base.shape and x_target.shape are (n_samples, num_channels, image_size, image_size).

        if cond is Conditions.all_attention_blocks:
            return self.forward_both_domains(inp)
        elif cond is Conditions.single_self_attention:
            return self.forward_single_domain_without_cross_attention(inp['x'])
        else:
            logger.exception('Not implemented!')
            raise NotImplementedError


class DINOHead(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 use_bn=False,
                 norm_last_layer=True,
                 nlayers=3,
                 hidden_dim=2048,
                 bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            dino_utils.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
