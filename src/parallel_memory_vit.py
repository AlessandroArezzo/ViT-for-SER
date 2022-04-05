import torch.nn as nn
from torch.hub import load_state_dict_from_url

from .utils.helpers import pe_check
from .utils.tokenizer import Tokenizer
import torch
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Parameter, init, LSTM
import torch.nn.functional as F
from .utils.transformers import TransformerEncoderLayer
from .utils.memory_tokenizer import VerticalTokenizer

try:
    from timm.models.registry import register_model
except ImportError:
    from .registry import register_model

model_urls = {
    'cct_7_3x1_32':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_7_3x1_32_cifar10_300epochs.pth',
    'cct_7_3x1_32_sine':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_7_3x1_32_sine_cifar10_5000epochs.pth',
    'cct_7_3x1_32_c100':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_7_3x1_32_cifar100_300epochs.pth',
    'cct_7_3x1_32_sine_c100':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_7_3x1_32_sine_cifar100_5000epochs.pth',
    'cct_7_7x2_224_sine':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_7_7x2_224_flowers102.pth',
    'cct_14_7x2_224':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_14_7x2_224_imagenet.pth',
    'cct_14_7x2_384':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/finetuned/cct_14_7x2_384_imagenet.pth',
    'cct_14_7x2_384_fl':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/finetuned/cct_14_7x2_384_flowers102.pth',
}


class ParallelMemoryCCT(nn.Module):
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
                 width_patches_lstm=2,
                 type='concatenate',  # type of union lstm output with cct output: concatenate | sum | average
                 *args, **kwargs):
        super(ParallelMemoryCCT, self).__init__()

        self.tokenizer_cct = Tokenizer(n_input_channels=n_input_channels,
                                       n_output_channels=embedding_dim,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       pooling_kernel_size=pooling_kernel_size,
                                       pooling_stride=pooling_stride,
                                       pooling_padding=pooling_padding,
                                       max_pool=True,
                                       activation=nn.ReLU,
                                       n_conv_layers=n_conv_layers,
                                       conv_bias=False)

        self.tokenizer_lstm = VerticalTokenizer(img_size=img_size, channels=n_input_channels, dim=embedding_dim,
                                                width_patches=width_patches_lstm)

        self.classifier = ParallelMemoryClassifier(
            sequence_length=self.tokenizer_cct.sequence_length(n_channels=n_input_channels,
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
            type=type
        )

    def forward(self, x):
        x_cct = self.tokenizer_cct(x)
        x_lstm = self.tokenizer_lstm(x)

        return self.classifier(x_cct, x_lstm)


class ParallelMemoryClassifier(Module):
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
                 type='concatenate'):
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
        self.attention_pool_lstm = Linear(self.embedding_dim, 1)

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

        self.lstm = LSTM(embedding_dim, embedding_dim)
        self.norm_lstm = LayerNorm(embedding_dim)
        fc_dim = embedding_dim
        if self.union_type == 'concatenate':
            fc_dim = embedding_dim * 2

        self.norm_union = LayerNorm(fc_dim)
        self.fc = Linear(fc_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x_cct, x_lstm):
        if self.positional_emb is None and x_cct.size(1) < self.sequence_length:
            x_cct = F.pad(x_cct, (0, 0, 0, self.n_channels - x_cct.size(1)), mode='constant', value=0)

        if self.positional_emb is not None:
            x_cct += self.positional_emb

        x_cct = self.dropout(x_cct)

        for blk in self.blocks:
            x_cct = blk(x_cct)

        x_cct = self.norm_cct(x_cct)
        x_cct = torch.matmul(F.softmax(self.attention_pool_cct(x_cct), dim=1).transpose(-1, -2), x_cct).squeeze(-2)

        x_lstm = self.lstm(x_lstm)[0]
        x_lstm = self.norm_lstm(x_lstm)

        x_lstm = torch.matmul(F.softmax(self.attention_pool_lstm(x_lstm), dim=1).transpose(-1, -2), x_lstm).squeeze(-2)
        if self.union_type == 'concatenate':
            x = torch.cat((x_cct, x_lstm), 1)
        elif self.union_type == 'sum':
            x = torch.sum(torch.stack([x_cct, x_lstm]), dim=0)
        elif self.union_type == 'average':
            x = torch.mean(torch.stack([x_cct, x_lstm]), dim=0)

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


def _parallel_memory_cct(arch, pretrained, progress, pretrained_arch, num_layers, num_heads, mlp_ratio, embedding_dim,
                         kernel_size=3, stride=None, padding=None, positional_embedding='learnable',
                         width_patches=2, type='concatenate',
                         *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = ParallelMemoryCCT(num_layers=num_layers,
                              num_heads=num_heads,
                              mlp_ratio=mlp_ratio,
                              embedding_dim=embedding_dim,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding, width_patches_lstm=width_patches,
                              type=type,
                              *args, **kwargs)

    if pretrained:
        if pretrained_arch in model_urls:
            state_dict = load_state_dict_from_url(model_urls[pretrained_arch],
                                                  progress=progress)
            if positional_embedding == 'learnable':
                state_dict = pe_check(model, state_dict)
            elif positional_embedding == 'sine':
                state_dict['classifier.positional_emb'] = model.state_dict()['classifier.positional_emb']

            model_dict = model.state_dict()

            state_dict = {k: v for k, v in state_dict.items() if
                                k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(state_dict)
            model.load_state_dict(state_dict, strict=False)
        else:
            raise RuntimeError(f'Variant {pretrained_arch} does not yet have pretrained weights.')

    return model


def parallel_memory_cct_4(arch, pretrained, progress, width_patches, type, *args, **kwargs):
    return _parallel_memory_cct(arch=arch, pretrained=pretrained, progress=progress,
                                num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                                width_patches=width_patches, type=type, *args, **kwargs)


def parallel_memory_cct_7(arch, pretrained, progress, width_patches, type, *args, **kwargs):
    return _parallel_memory_cct(arch=arch, pretrained=pretrained, progress=progress,
                                num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                                width_patches=width_patches, type=type, *args, **kwargs)


@register_model
def p_mcct_7_1_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                      *args, **kwargs):
    return parallel_memory_cct_7(arch='p_mcct_7_1_3x1_32', pretrained=pretrained, progress=progress,
                                 pretrained_arch="cct_7_3x1_32",
                                 kernel_size=3, n_conv_layers=1, width_patches=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='concatenate',
                                 *args, **kwargs)

@register_model
def p_mcct_7_1_3x1_32_c100(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                      *args, **kwargs):
    return parallel_memory_cct_7(arch='p_mcct_7_1_3x1_32_c100', pretrained=pretrained, progress=progress,
                                 pretrained_arch="cct_7_3x1_32_c100",
                                 kernel_size=3, n_conv_layers=1, width_patches=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='concatenate',
                                 *args, **kwargs)


@register_model
def p_mcct_7_1_3x1_32_sum(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                          *args, **kwargs):
    return parallel_memory_cct_7(arch='p_mcct_7_1_3x1_32_sum',
                                 pretrained=pretrained, progress=progress,
                                 pretrained_arch="cct_7_3x1_32",
                                 kernel_size=3, n_conv_layers=1, width_patches=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='sum',
                                 *args, **kwargs)


@register_model
def p_mcct_7_1_3x1_32_average(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                              *args, **kwargs):
    return parallel_memory_cct_7(arch='p_mcct_7_1_3x1_32_average', pretrained=pretrained, progress=progress,
                                 pretrained_arch="cct_7_3x1_32",
                                 kernel_size=3, n_conv_layers=1, width_patches=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='average',
                                 *args, **kwargs)


@register_model
def p_mcct_7_2_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                      *args, **kwargs):
    return parallel_memory_cct_7(arch='p_mcct_7_2_3x1_32', pretrained=pretrained, progress=progress,
                                 pretrained_arch="cct_7_3x1_32",
                                 kernel_size=3, n_conv_layers=1, width_patches=2,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='concatenate',
                                 *args, **kwargs)

@register_model
def p_mcct_7_2_3x1_32_c100(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                      *args, **kwargs):
    return parallel_memory_cct_7(arch='p_mcct_7_2_3x1_32_c100', pretrained=pretrained, progress=progress,
                                 pretrained_arch="cct_7_3x1_32_c100",
                                 kernel_size=3, n_conv_layers=1, width_patches=2,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='concatenate',
                                 *args, **kwargs)

@register_model
def p_mcct_7_2_3x1_32_sum(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                          *args, **kwargs):
    return parallel_memory_cct_7(arch='p_mcct_7_2_3x1_32_sum', pretrained=pretrained, progress=progress,
                                 pretrained_arch="cct_7_3x1_32",
                                 kernel_size=3, n_conv_layers=1, width_patches=2,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='sum',
                                 *args, **kwargs)


@register_model
def p_mcct_7_2_3x1_32_average(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                              *args, **kwargs):
    return parallel_memory_cct_7(arch='p_mcct_7_2_3x1_32_average', pretrained=pretrained, progress=progress,
                                 pretrained_arch="cct_7_3x1_32",
                                 kernel_size=3, n_conv_layers=1, width_patches=2,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='average',
                                 *args, **kwargs)


@register_model
def p_mcct_4_1_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                      *args, **kwargs):
    return parallel_memory_cct_4(arch='p_mcct_4_1_3x1_32', pretrained=pretrained, progress=progress,
                                 pretrained_arch="",
                                 kernel_size=3, n_conv_layers=1, width_patches=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='concatenate',
                                 *args, **kwargs)


@register_model
def p_mcct_4_1_3x1_32_sum(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                          *args, **kwargs):
    return parallel_memory_cct_4(arch='p_mcct_4_1_3x1_32_sum', pretrained=pretrained, progress=progress,
                                 pretrained_arch="",
                                 kernel_size=3, n_conv_layers=1, width_patches=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='sum',
                                 *args, **kwargs)


@register_model
def p_mcct_4_1_3x1_32_average(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                              *args, **kwargs):
    return parallel_memory_cct_4(arch='p_mcct_4_1_3x1_32_average', pretrained=pretrained, progress=progress,
                                 pretrained_arch="",
                                 kernel_size=3, n_conv_layers=1, width_patches=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='average',
                                 *args, **kwargs)


@register_model
def p_mcct_4_2_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                      *args, **kwargs):
    return parallel_memory_cct_4(arch='p_mcct_4_2_3x1_32', pretrained=pretrained, progress=progress,
                                 pretrained_arch="",
                                 kernel_size=3, n_conv_layers=1, width_patches=2,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='concatenate',
                                 *args, **kwargs)


@register_model
def p_mcct_4_2_3x1_32_sum(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                          *args, **kwargs):
    return parallel_memory_cct_4(arch='p_mcct_4_2_3x1_32_sum', pretrained=pretrained, progress=progress,
                                 pretrained_arch="",
                                 kernel_size=3, n_conv_layers=1, width_patches=2,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='sum',
                                 *args, **kwargs)


@register_model
def p_mcct_4_2_3x1_32_average(pretrained=False, progress=False, img_size=32, positional_embedding='learnable',
                              num_classes=10, *args, **kwargs):
    return parallel_memory_cct_4(arch='p_mcct_4_2_3x1_32_average', pretrained=pretrained, progress=progress,
                                 pretrained_arch="",
                                 kernel_size=3, n_conv_layers=1, width_patches=2,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='average',
                                 *args, **kwargs)