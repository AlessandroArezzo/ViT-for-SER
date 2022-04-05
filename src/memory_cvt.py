import torch.nn as nn
from .memory_cct import TransformerMemoryClassifier
from .utils.memory_tokenizer import VerticalTokenizer

try:
    from timm.models.registry import register_model
except ImportError:
    from .registry import register_model


class MemoryCVT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 width_patches=2,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 *args, **kwargs):
        super(MemoryCVT, self).__init__()
        assert img_size % width_patches == 0, f"Image size ({img_size}) has to be" \
                                              f"divisible by patch size ({width_patches})"
        self.tokenizer = VerticalTokenizer(img_size=img_size, channels=n_input_channels, dim=embedding_dim,
                                           width_patches=width_patches)

        self.classifier = TransformerMemoryClassifier(
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
            positional_embedding=positional_embedding
        )

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


def _memory_cvt(num_layers, num_heads, mlp_ratio, embedding_dim,
                width_patches=2,
                *args, **kwargs):
    model = MemoryCVT(num_layers=num_layers,
                      num_heads=num_heads,
                      mlp_ratio=mlp_ratio,
                      embedding_dim=embedding_dim,
                      width_patches=width_patches,
                      *args, **kwargs)
    return model


def memory_cvt_4(*args, **kwargs):
    return _memory_cvt(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                       *args, **kwargs)


def memory_cvt_7(*args, **kwargs):
    return _memory_cvt(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                       *args, **kwargs)


@register_model
def mcvt_4_1_32(
        img_size=32, positional_embedding='learnable', num_classes=10,
        *args, **kwargs):
    return memory_cvt_4(arch='mcvt_4_1_32',
                        width_patches=1,
                        img_size=img_size, positional_embedding=positional_embedding,
                        num_classes=num_classes,
                        *args, **kwargs)


@register_model
def mcvt_4_2_32(img_size=32, positional_embedding='learnable', num_classes=10,
                *args, **kwargs):
    return memory_cvt_4(arch='mcvt_4_2_32',
                        width_patches=2,
                        img_size=img_size, positional_embedding=positional_embedding,
                        num_classes=num_classes,
                        *args, **kwargs)


@register_model
def mcvt_7_1_32(img_size=32, positional_embedding='learnable', num_classes=10,
                *args, **kwargs):
    return memory_cvt_7(arch='mcvt_7_1_32',
                        width_patches=1,
                        img_size=img_size, positional_embedding=positional_embedding,
                        num_classes=num_classes,
                        *args, **kwargs)


@register_model
def mcvt_7_2_32(img_size=32, positional_embedding='learnable', num_classes=10,
                *args, **kwargs):
    return memory_cvt_7(arch='mcvt_7_2_32',
                        width_patches=2,
                        img_size=img_size, positional_embedding=positional_embedding,
                        num_classes=num_classes,
                        *args, **kwargs)
