from einops.layers import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init, Linear, ModuleList, Dropout, LayerNorm, ReLU
from .utils.tokenizer import Tokenizer
from .utils.transformers import TransformerEncoderLayer
import torch

try:
    from timm.models.registry import register_model
except ImportError:
    from .registry import register_model

class HybridCCT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 dim_naive_features=5600,
                 *args, **kwargs):
        super(HybridCCT, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        self.classifier = HybridCCTClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding,
            dim_naive_features=dim_naive_features
        )

    def forward(self, x):
        x_spect = self.tokenizer(x[0])
        x_wav = x[1]
        return self.classifier(x_spect, x_wav)


class HybridCCTClassifier(nn.Module):
    def __init__(self,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='learnable',
                 sequence_length=None,
                 type='concatenate',
                 dim_naive_features=5600):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.num_tokens = 0

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        assert type == 'concatenate' or type == 'sum' or type == 'average', \
            f"Union output type is set to {type} that is not specified"
        self.union_type = type

        self.attention_pool_cct = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                requires_grad=True)
                init.normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm_cct = LayerNorm(embedding_dim)

        self.fc_wav = Linear(dim_naive_features, embedding_dim)

        fc_dim = embedding_dim
        if self.union_type == 'concatenate':
            fc_dim = embedding_dim * 2

        self.norm_union = LayerNorm(fc_dim)
        self.fc = Linear(fc_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x_spect, x_wav):
        if self.positional_emb is None and x_spect.size(1) < self.sequence_length:
            x_spect = F.pad(x_spect, (0, 0, 0, self.n_channels - x_spect.size(1)), mode='constant', value=0)

        if self.positional_emb is not None:
            x_spect += self.positional_emb

        x_spect = self.dropout(x_spect)

        for blk in self.blocks:
            x_spect = blk(x_spect)

        x_spect = self.norm_cct(x_spect)
        x_spect = torch.matmul(F.softmax(self.attention_pool_cct(x_spect), dim=1).transpose(-1, -2), x_spect).squeeze(
            -2)

        x_naive = torch.flatten(x_wav, start_dim=1)

        x_naive = self.fc_wav(x_naive)
        if self.union_type == 'concatenate':
            x = torch.cat((x_spect, x_naive), 1)
        elif self.union_type == 'sum':
            x = torch.sum(torch.stack([x_spect, x_naive]), dim=0)
        elif self.union_type == 'average':
            x = torch.mean(torch.stack([x_spect, x_naive]), dim=0)

        x = self.norm_union(x)
        x = self.fc(x)

        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


def _hybrid_cct(num_layers, num_heads, mlp_ratio, embedding_dim,
                         kernel_size=3, stride=None, padding=None,
                         type='concatenate',
                         *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = HybridCCT(num_layers=num_layers,
                              num_heads=num_heads,
                              mlp_ratio=mlp_ratio,
                              embedding_dim=embedding_dim,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              type=type,
                              dim_naive_features=5600,
                              *args, **kwargs)
    return model

def hybrid_cct_4(type, *args, **kwargs):
    return _hybrid_cct(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                                type=type, *args, **kwargs)

def hybrid_cct_7(type, *args, **kwargs):
    return _hybrid_cct(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                                type=type, *args, **kwargs)

@register_model
def hybrid_cct_7_3x1_32(img_size=32, positional_embedding='learnable', num_classes=10,
                      *args, **kwargs):
    return hybrid_cct_7(arch='hybrid_cct_7_3x1_32',
                                 kernel_size=3, n_conv_layers=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='concatenate',
                                 *args, **kwargs)

@register_model
def hybrid_cct_7_3x1_32_sum(img_size=32, positional_embedding='learnable', num_classes=10,
                      *args, **kwargs):
    return hybrid_cct_7(arch='hybrid_cct_7_3x1_32_sum',
                                 kernel_size=3, n_conv_layers=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='sum',
                                 *args, **kwargs)

@register_model
def hybrid_cct_7_3x1_32_average(img_size=32, positional_embedding='learnable', num_classes=10,
                      *args, **kwargs):
    return hybrid_cct_7(arch='hybrid_cct_7_3x1_32_average',
                                 kernel_size=3, n_conv_layers=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='average',
                                 *args, **kwargs)

@register_model
def hybrid_cct_4_3x1_32(img_size=32, positional_embedding='learnable', num_classes=10,
                      *args, **kwargs):
    return hybrid_cct_4(arch='hybrid_cct_7_3x1_32',
                                 kernel_size=3, n_conv_layers=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='concatenate',
                                 *args, **kwargs)

@register_model
def hybrid_cct_4_3x1_32_sum(img_size=32, positional_embedding='learnable', num_classes=10,
                      *args, **kwargs):
    return hybrid_cct_4(arch='hybrid_cct_7_3x1_32_sum',
                                 kernel_size=3, n_conv_layers=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='sum',
                                 *args, **kwargs)

@register_model
def hybrid_cct_4_3x1_32_average(img_size=32, positional_embedding='learnable', num_classes=10,
                      *args, **kwargs):
    return hybrid_cct_4(arch='hybrid_cct_7_3x1_32_average',
                                 kernel_size=3, n_conv_layers=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='average',
                                 *args, **kwargs)
