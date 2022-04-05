from .utils.tokenizer import Tokenizer
from .utils.transformers import TransformerEncoderLayer
from .utils.helpers import pe_check
import torch.nn as nn

import torch
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Parameter, init, LSTM
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

try:
    from timm.models.registry import register_model, model_entrypoint
except ImportError:
    pass

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


class SpeakerVGGVoxCCT(nn.Module):
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
                 n_layers_scnd_transformer=4,
                 n_heads_scnd_transformer=2,
                 *args, **kwargs):
        super(SpeakerVGGVoxCCT, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
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

        self.classifier = SpeakerVGGVoxCCTClassifier(
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
            n_layers_scnd_transformer=n_layers_scnd_transformer,
            n_heads_scnd_transformer=n_heads_scnd_transformer,
        )

    def forward(self, x):
        x_emotion = x[0]
        x_speaker = x[1]
        x_emotion = self.tokenizer(x_emotion)
        return self.classifier(x_emotion, x_speaker)


class SpeakerVGGVoxCCTClassifier(Module):
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
                 n_layers_scnd_transformer=4,
                 n_heads_scnd_transformer=2,
                 type="no_pooling" # pooling | no_pooling: if pooling two self-attention output are pooled in one vector, without pooling architcture extracts first vector (associated to the emotion)
                 ):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.num_tokens = 0
        self.num_classes = num_classes

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        self.attention_pool_cct = Linear(self.embedding_dim, 1)
        # self.attention_pool_speaker = Linear(self.embedding_dim, 1)

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
        self.norm = LayerNorm(embedding_dim)

        self.blocks_scnd_level = ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads_scnd_transformer,
                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(n_layers_scnd_transformer)])

        self.positional_emb_scnd_level = Parameter(self.sinusoidal_embedding(3, embedding_dim),
                                                   requires_grad=False)
        self.project_vgg_features = Linear(1024, embedding_dim)

        self.type = type

        if type == 'pooling':
            self.attention_pool_speaker_emotion = Linear(2, 1)

        self.norm_after_blocks = LayerNorm(embedding_dim)
        self.fc = Linear(embedding_dim, num_classes)

        self.apply(self.init_weight)

    def forward(self, x_emotion, x_speaker):
        if self.positional_emb is None and x_emotion.size(1) < self.sequence_length:
            x_emotion = F.pad(x_emotion, (0, 0, 0, self.n_channels - x_emotion.size(1)), mode='constant', value=0)

        if self.positional_emb is not None:
            x_emotion += self.positional_emb

        x_emotion = self.dropout(x_emotion)

        for blk in self.blocks:
            x_emotion = blk(x_emotion)
        # x_spect = self.lstm(x_spect)[0]
        x_emotion = self.norm_cct(x_emotion)
        x_emotion = torch.matmul(F.softmax(self.attention_pool_cct(x_emotion), dim=1).transpose(-1, -2),
                                 x_emotion).squeeze(-2)

        x_speaker = self.project_vgg_features(x_speaker)

        x = torch.stack((x_emotion, x_speaker), 1)

        # x += self.positional_emb_scnd_level
        for blk in self.blocks_scnd_level:
            x = blk(x)

        if type == 'pooling':
            x = torch.matmul(F.softmax(self.attention_pool_cct(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0, :]

        x = self.norm_after_blocks(x)

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


def _speaker_vgg_cct(pretrained, progress, pretrained_arch,
                     num_layers, num_heads, mlp_ratio, embedding_dim,
                     kernel_size=3, stride=None, padding=None, positional_embedding='learnable',
                     *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = SpeakerVGGVoxCCT(num_layers=num_layers,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             embedding_dim=embedding_dim,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
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


def speaker_vgg_cct_7(pretrained, progress, *args, **kwargs):
    return _speaker_vgg_cct(pretrained=pretrained, progress=progress,
                            num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                            *args, **kwargs)


@register_model
def speaker_vgg_cct_7_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding='learnable',
                             num_classes=10,
                             *args, **kwargs):
    return speaker_vgg_cct_7(arch='speaker_vgg_cct_7_3x1_32', pretrained=pretrained, progress=progress,
                             pretrained_arch="cct_7_3x1_32_c100",
                             kernel_size=3, n_conv_layers=1,
                             img_size=img_size, positional_embedding=positional_embedding,
                             num_classes=num_classes,
                             n_layers_scnd_transformer=1,
                             n_heads_scnd_transformer=1,
                             *args, **kwargs)

@register_model
def speaker_vgg_cct_7_3x1_32_pooling(pretrained=False, progress=False, img_size=32, positional_embedding='learnable',
                             num_classes=10,
                             *args, **kwargs):
    return speaker_vgg_cct_7(arch='speaker_vgg_cct_7_3x1_32', pretrained=pretrained, progress=progress,
                             pretrained_arch="cct_7_3x1_32_c100",
                             kernel_size=3, n_conv_layers=1,
                             img_size=img_size, positional_embedding=positional_embedding,
                             num_classes=num_classes,
                             n_layers_scnd_transformer=1,
                             n_heads_scnd_transformer=1,
                             type='pooling',
                             *args, **kwargs)