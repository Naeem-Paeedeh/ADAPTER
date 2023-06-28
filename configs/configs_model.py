# You don't need to set anything here!

from utils.other_utilities import pair
from enum import Enum


class ViTType(Enum):
    VanillaViT = 1
    CCT = 2
    Quadruple_ViT = 3
    Quadruple_CCT = 4

    def is_vit(self):
        return self is self.VanillaViT or self is self.Quadruple_ViT

    def is_quadruple(self):
        return self is self.Quadruple_ViT or self is self.Quadruple_ViT

    def is_cct(self):
        return self is self.CCT or self is self.Quadruple_CCT   # return not is_vit()


class ConfigurationModel:
    def __init__(self, embed_dim, transformer_num_heads, num_layers, net_type: ViTType,
                 patch_size, log_file='', stride=None, qkv_bias: bool = True,
                 transformer_mlp_dim: int = 0, transformer_mlp_ratio: float = 0.0,
                 attention_dropout_rate: float = 0.1,
                 dropout_rate: float = 0.1, droppath_rate: float = 0.1, stochastic_depth=0.1,
                 padding=None, pooling_kernel_size=0, pooling_stride=0, pooling_padding=0,
                 n_conv_layers=-1
                 ):
        self.embed_dim = embed_dim

        # We define both transformer_mlp_dim and transformer_mlp_ratio to be more flexible with different architectures
        # and implementations!
        self.transformer_mlp_dim = transformer_mlp_dim
        self.transformer_mlp_ratio = transformer_mlp_ratio

        self.transformer_num_heads = transformer_num_heads
        self.num_layers = num_layers
        self.net_type = net_type        # True for ViT
        self.qkv_bias = qkv_bias
        self.attention_dropout_rate = attention_dropout_rate
        self.dropout_rate = dropout_rate
        self.droppath_rate = droppath_rate
        self.stochastic_depth = stochastic_depth
        if self.transformer_mlp_dim == 0 and self.transformer_mlp_ratio == 0.0:
            raise "Error: You should set transformer_mlp_dim or transformer_mlp_ratio!"

        if self.transformer_mlp_dim > 0 and self.transformer_mlp_ratio > 0.0:
            raise "Error: You can't set both transformer_mlp_dim and transformer_mlp_ratio at the same time!"

        if self.transformer_mlp_dim == 0:
            self.transformer_mlp_dim = int(self.embed_dim * self.transformer_mlp_ratio)

        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_stride = pooling_stride
        self.pooling_padding = pooling_padding

        self.n_conv_layers = n_conv_layers

        self.log_file = log_file

        if self.net_type.is_cct():
            if self.stride is None:
                self.stride = pair(max(1, (patch_size // 2) - 1))
            else:
                self.stride = pair(stride)
            if self.padding is None:
                self.padding = pair(max(1, (patch_size // 2)))
            else:
                self.padding = pair(padding)
        else:   # ViT
            if self.stride is None:
                self.stride = self.patch_size
            self.stride = pair(self.stride)
            assert self.stride == pair(self.patch_size), "Error: Wrong stride value for ViT!"
            self.padding = pair(0)


# CCT
#  kernel_size=7, stride=2, padding=3, pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
#  stochastic_depth=0.1,


config_ViT_b16 = ConfigurationModel(patch_size=16,
                                    stride=16,
                                    embed_dim=768,
                                    transformer_mlp_dim=3072,
                                    transformer_num_heads=12,
                                    num_layers=12,
                                    attention_dropout_rate=0.0,
                                    dropout_rate=0.1,
                                    net_type=ViTType.VanillaViT)

config_ViT_b16_16_Quadruple = ConfigurationModel(patch_size=16,
                                                 stride=16,
                                                 embed_dim=768,
                                                 transformer_mlp_dim=768*4,
                                                 transformer_num_heads=12,
                                                 num_layers=12,
                                                 attention_dropout_rate=0.0,
                                                 dropout_rate=0.1,
                                                 net_type=ViTType.Quadruple_ViT)

config_cct_7_3x1 = ConfigurationModel(patch_size=3,
                                      stride=None,
                                      padding=None,
                                      embed_dim=256,
                                      num_layers=7,
                                      transformer_num_heads=4,
                                      transformer_mlp_ratio=2.0,
                                      pooling_kernel_size=3,
                                      pooling_stride=2,
                                      pooling_padding=1,
                                      dropout_rate=0.0,
                                      attention_dropout_rate=0.1,
                                      stochastic_depth=0.1,
                                      net_type=ViTType.CCT,
                                      n_conv_layers=1)

config_cct_7_5x1_modified = ConfigurationModel(patch_size=5,
                                               stride=4,
                                               padding=None,
                                               embed_dim=256,
                                               num_layers=7,
                                               transformer_num_heads=4,
                                               transformer_mlp_ratio=2.0,
                                               pooling_kernel_size=3,
                                               pooling_stride=2,
                                               pooling_padding=1,
                                               dropout_rate=0.0,
                                               attention_dropout_rate=0.1,
                                               stochastic_depth=0.1,
                                               net_type=ViTType.CCT,
                                               n_conv_layers=1)

config_cct_14_7x2 = ConfigurationModel(patch_size=7,
                                       stride=None,
                                       padding=None,
                                       embed_dim=384,
                                       num_layers=7,
                                       transformer_num_heads=4,
                                       transformer_mlp_ratio=2.0,
                                       pooling_kernel_size=3,
                                       pooling_stride=2,
                                       pooling_padding=1,
                                       n_conv_layers=2,
                                       dropout_rate=0.0,
                                       attention_dropout_rate=0.1,
                                       stochastic_depth=0.1,
                                       net_type=ViTType.CCT)

# For arguments of the program as strings
string_to_model_config = {'CCT-7/3x1': config_cct_7_3x1,
                          'ViT-B/16': config_ViT_b16_16_Quadruple,
                          'CCT-7/5x1-Modified': config_cct_7_5x1_modified,
                          'CCT-14/7x2': config_cct_14_7x2}
