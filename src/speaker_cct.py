import os

import torch.nn as nn
from timm.models.helpers import load_state_dict
from timm.models.layers import set_layer_config
from .utils.helpers import pe_check

from .utils.tokenizer import Tokenizer
import torch
from torch.nn import Linear, Module, ModuleList, Linear, Dropout, LayerNorm, Parameter, init, LSTM
import torch.nn.functional as F
from .utils.transformers import TransformerEncoderLayer, Attention
#import wavencoder
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from torch.hub import load_state_dict_from_url
import numpy as np

try:
    from timm.models.registry import register_model, model_entrypoint
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

class SpeakerCCT(nn.Module):
    def __init__(self,
                 speaker_embedder=None,
                 gender_embedder=None,
                 corpus_embedder=None,
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
                 type='self-attention',
                 n_layers_scnd_transformer=4,
                 n_heads_scnd_transformer=2,
                 after_fc_embedded=False,
                 *args, **kwargs):
        super(SpeakerCCT, self).__init__()

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

        self.speaker_embedder = speaker_embedder
        self.gender_embedder = gender_embedder
        self.corpus_embedder = corpus_embedder

        if self.speaker_embedder is not None:
            self.speaker_embedder.requires_grad = False
        if self.gender_embedder is not None:
            self.gender_embedder.requires_grad = False
        if self.corpus_embedder is not None:
            self.corpus_embedder.requires_grad = False
        self.after_fc_embedded=after_fc_embedded

        self.classifier = SpeakerCCTClassifier(
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
            type=type,
            n_layers_scnd_transformer=n_layers_scnd_transformer,
            n_heads_scnd_transformer=n_heads_scnd_transformer,
            after_fc_embedded=after_fc_embedded
        )

    def forward(self, x):
        x_spect = self.tokenizer(x)
        x_speaker = None
        if self.speaker_embedder:
            if self.after_fc_embedded:
                x_speaker = self.speaker_embedder(x)
            else:
                x_speaker = self.speaker_embedder.extract_memory_embedding(x)
        x_gender = None
        if self.gender_embedder:
            if self.after_fc_embedded:
                x_gender = self.gender_embedder(x)
            else:
                x_gender = self.gender_embedder.extract_memory_embedding(x)
        x_corpus = None
        if self.corpus_embedder:
            if self.after_fc_embedded:
                x_corpus = self.corpus_embedder(x)
            else:
                x_corpus = self.corpus_embedder.extract_memory_embedding(x)
        return self.classifier(x_spect, x_speaker, x_gender, x_corpus)


class SpeakerCCTClassifier(Module):
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
                 type='self-attention',
                 n_layers_scnd_transformer=4,
                 n_heads_scnd_transformer=2,
                 after_fc_embedded=False
                 ):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.num_tokens = 0
        self.num_classes = num_classes
        self.after_fc_embedded=after_fc_embedded

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        assert type == 'self-attention' or type == 'concatenate' or type == 'sum' or type == 'average', \
            f"Union output type is set to {type} that is not specified"
        self.union_type = type

        self.attention_pool_cct = Linear(self.embedding_dim, 1)
        #self.attention_pool_speaker = Linear(self.embedding_dim, 1)

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

        #self.speaker_encoder = wavencoder.models.Wav2Vec(pretrained=True)
        #self.fc_wav = Linear(5600, embedding_dim)
        self.lstm = LSTM(embedding_dim, embedding_dim)
        self.norm = LayerNorm(embedding_dim)

        fc_dim = embedding_dim

        if self.union_type == 'concatenate':
            fc_dim = embedding_dim * 2

        if self.after_fc_embedded:
            self.norm_spect = LayerNorm(fc_dim)
            self.fc_spect = Linear(fc_dim, num_classes)
            self.project_speaker = Linear(11, num_classes)
            self.project_gender = Linear(2, num_classes)
            self.project_corpus = Linear(5, num_classes)
            self.fc = Linear(num_classes, num_classes)
            if self.union_type == 'self-attention':
                """
                self.self_attn = Attention(dim=embedding_dim, num_heads=4,
                                               attention_dropout=attention_dropout, projection_dropout=dropout)
                """
                self.blocks_scnd_level = ModuleList([
                    TransformerEncoderLayer(d_model=num_classes, nhead=n_heads_scnd_transformer,
                                            dim_feedforward=dim_feedforward, dropout=dropout,
                                            attention_dropout=attention_dropout, drop_path_rate=dpr[i])
                    for i in range(n_layers_scnd_transformer)])
                self.positional_emb_scnd_level = Parameter(self.sinusoidal_embedding(3, embedding_dim),
                                                           requires_grad=False)
                self.norm_union = LayerNorm(num_classes)

        else:
            if self.union_type == 'self-attention':
                """
                self.self_attn = Attention(dim=embedding_dim, num_heads=4,
                                               attention_dropout=attention_dropout, projection_dropout=dropout)
                """
                self.blocks_scnd_level = ModuleList([
                    TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads_scnd_transformer,
                                            dim_feedforward=dim_feedforward, dropout=dropout,
                                            attention_dropout=attention_dropout, drop_path_rate=dpr[i])
                    for i in range(n_layers_scnd_transformer)])
                self.positional_emb_scnd_level = Parameter(self.sinusoidal_embedding(3, embedding_dim),
                                                           requires_grad=False)
                self.speaker_attention_pool = Linear(self.embedding_dim, 1)

            self.norm_union = LayerNorm(embedding_dim)
            self.fc = Linear(embedding_dim, num_classes)


        self.apply(self.init_weight)

    def forward(self, x_spect, x_speaker=None, x_gender=None, x_corpus=None):
        if self.positional_emb is None and x_spect.size(1) < self.sequence_length:
            x_spect = F.pad(x_spect, (0, 0, 0, self.n_channels - x_spect.size(1)), mode='constant', value=0)

        if self.positional_emb is not None:
            x_spect += self.positional_emb

        x_spect = self.dropout(x_spect)

        for blk in self.blocks:
            x_spect = blk(x_spect)
        #x_spect = self.lstm(x_spect)[0]
        x_spect = self.norm_cct(x_spect)
        x_spect = torch.matmul(F.softmax(self.attention_pool_cct(x_spect), dim=1).transpose(-1, -2), x_spect).squeeze(
            -2)

        """
        #x_speaker = self.speaker_encoder(x_wav)
        #x_speaker = x_wav

        x_speaker = torch.flatten(x_wav, start_dim=1)
        
        x_speaker = torch.matmul(F.softmax(self.attention_pool_speaker(x_wav), dim=1).transpose(-1, -2), x_wav) \
            .squeeze(-2)
        
        x_speaker = self.fc_wav(x_speaker)
        """
        """
        x_spect = self.norm_spect(x_spect)
        x_spect = self.fc_spect(x_spect)
        """
        if self.after_fc_embedded:
            x_spect = self.norm_spect(x_spect)
            x_spect = self.fc_spect(x_spect)
            if x_speaker is not None:
                x_speaker = self.project_speaker(x_speaker)
            if x_gender is not None:
                x_gender = self.project_gender(x_gender)
            if x_corpus is not None:
                x_corpus = self.project_corpus(x_corpus)

        else:
            x_speaker = torch.matmul(F.softmax(self.speaker_attention_pool(x_speaker), dim=1).transpose(-1, -2), x_speaker).squeeze(-2)

        if self.union_type == 'concatenate':
            x = torch.cat((x_spect, x_speaker), 1)
        elif self.union_type == 'sum':
            x = torch.sum(torch.stack([x_spect, x_speaker]), dim=0)
        elif self.union_type == 'average':
            x = torch.mean(torch.stack([x_spect, x_speaker]), dim=0)

        elif self.union_type == 'self-attention':

            if x_speaker is not None and x_gender is not None and x_corpus is not None:
                x = torch.stack((x_spect, x_speaker, x_gender, x_corpus), 1)
            elif x_speaker is None:
                if x_gender is not None and x_corpus is not None:
                    x = torch.stack((x_spect, x_gender, x_corpus), 1)
                elif x_gender is None and x_corpus is not None:
                    x = torch.stack((x_spect, x_corpus), 1)
                else:
                    x = torch.stack((x_spect, x_speaker), 1)
            elif x_gender is None:
                if x_speaker is not None and x_corpus is not None:
                    x = torch.stack((x_spect, x_speaker, x_corpus), 1)
                elif x_speaker is None and x_corpus is not None:
                    x = torch.stack((x_spect, x_corpus), 1)
                else:
                    x = torch.stack((x_spect, x_speaker), 1)
            elif x_corpus is None:
                if x_speaker is not None and x_gender is not None:
                    x = torch.stack((x_spect, x_speaker, x_gender), 1)
                elif x_speaker is None and x_gender is not None:
                    x = torch.stack((x_spect, x_gender), 1)
                else:
                    x = torch.stack((x_spect, x_speaker), 1)
            """
            x = self.self_attn(x)[:, 0, :]
            """

            #x += self.positional_emb_scnd_level
            for blk in self.blocks_scnd_level:
                x = blk(x)
            x = x[:, 0, :]

        x = self.norm_union(x)
        """
        #regularization
        sum_batch_sample = 0
        for batch_sample_index in np.arange(0, x.shape[0]):
            sum_batch_sample += x[batch_sample_index].norm(dim=0, p=1)
        
        self.fc.requires_grad = False
        
        regularization_vector = x.shape[0] / (self.num_classes * sum_batch_sample.item())
        for fc_weight_index in np.arange(0, self.fc.weight.shape[0]):
            #fc_column_weights = self.fc.weight[fc_weight_index, :]
            self.fc.weight[fc_weight_index, :] = torch.from_numpy(np.repeat(regularization_vector, self.fc.weight.shape[1]))
        """

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

def _read_embedder_model(embedder_path, num_embedder_classes=11):
    model_name = 'mcct_7_3x1_32_c100'
    create_fn = model_entrypoint(model_name)
    with set_layer_config(scriptable=None, exportable=None, no_jit=None):
        model = create_fn(num_classes=num_embedder_classes)

    if os.path.splitext(embedder_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(embedder_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return
    state_dict = load_state_dict(embedder_path, False)
    model.load_state_dict(state_dict, strict=True)
    return model

def _speaker_cct(pretrained, progress, pretrained_arch,
                 num_layers, num_heads, mlp_ratio, embedding_dim, speaker=True, gender=False, corpus=False,
                 speaker_embedder_path=None, gender_embedder_path=None,
                 corpus_embedder_path=None, kernel_size=3, stride=None, padding=None, positional_embedding='learnable',
                 type='self-attention', *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    speaker_embedder = None
    if speaker:
        speaker_embedder = _read_embedder_model(speaker_embedder_path)
    gender_embedder = None
    if gender:
        gender_embedder = _read_embedder_model(gender_embedder_path, num_embedder_classes=2)
    corpus_embedder = None
    if corpus:
        corpus_embedder = _read_embedder_model(corpus_embedder_path, num_embedder_classes=5)
    model = SpeakerCCT(speaker_embedder=speaker_embedder,
                       gender_embedder=gender_embedder,
                       corpus_embedder=corpus_embedder,
                       num_layers=num_layers,
                      num_heads=num_heads,
                      mlp_ratio=mlp_ratio,
                      embedding_dim=embedding_dim,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
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

def speaker_cct_4(pretrained, progress, type, *args, **kwargs):
    return _speaker_cct( pretrained=pretrained, progress=progress,
                         num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                        type=type, *args, **kwargs)

def speaker_cct_7(pretrained, progress, type, *args, **kwargs):
    return _speaker_cct( pretrained=pretrained, progress=progress,
                         num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                         type=type, *args, **kwargs)

@register_model
def speaker_cct_7_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                         speaker_embedder_path=None,
                        *args, **kwargs):
    return speaker_cct_7(arch='speaker_cct_7_3x1_32', pretrained=pretrained, progress=progress,  pretrained_arch="cct_7_3x1_32_c100",
                                 kernel_size=3, n_conv_layers=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='self-attention',
                                 speaker_embedder_path=speaker_embedder_path,
                                 n_layers_scnd_transformer=1,
                                 *args, **kwargs)

@register_model
def speaker_gender_cct_7_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                         speaker_embedder_path=None,
                         gender_embedder_path=None,
                        *args, **kwargs):
    return speaker_cct_7(arch='speaker_gender_cct_7_3x1_32', pretrained=pretrained, progress=progress,  pretrained_arch="cct_7_3x1_32_c100",
                                 kernel_size=3, n_conv_layers=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='self-attention',
                                 speaker_embedder_path=speaker_embedder_path,
                                 gender_embedder_path=gender_embedder_path,
                                 gender=True,
                                 n_layers_scnd_transformer=2,
                                 *args, **kwargs)

@register_model
def speaker_corpus_cct_7_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                         speaker_embedder_path=None,
                         corpus_embedder_path=None,
                        *args, **kwargs):
    return speaker_cct_7(arch='speaker_corpus_cct_7_3x1_32', pretrained=pretrained, progress=progress,  pretrained_arch="cct_7_3x1_32_c100",
                                 kernel_size=3, n_conv_layers=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='self-attention',
                                 speaker_embedder_path=speaker_embedder_path,
                                 corpus_embedder_path=corpus_embedder_path,
                                 corpus=True,
                                 n_layers_scnd_transformer=2,
                                 *args, **kwargs)


@register_model
def speaker_gender_corpus_cct_7_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                         speaker_embedder_path = None,
                         gender_embedder_path=None,
                         corpus_embedder_path=None,
                        *args, **kwargs):
    return speaker_cct_7(arch='speaker_gender_corpus_cct_7_3x1_32', pretrained=pretrained, progress=progress,  pretrained_arch="cct_7_3x1_32_c100",
                                 kernel_size=3, n_conv_layers=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='self-attention',
                                 speaker_embedder_path=speaker_embedder_path,
                                 gender_embedder_path=gender_embedder_path,
                                 corpus_embedder_path=corpus_embedder_path,
                                    gender=True, corpus=True,
                                    *args, **kwargs)

@register_model
def speaker_cct_7_3x1_32_concatenate(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                         speaker_embedder_path = None,
                        *args, **kwargs):
    return speaker_cct_7(arch='speaker_cct_7_3x1_32', pretrained=pretrained, progress=progress,  pretrained_arch="cct_7_3x1_32_c100",
                                 kernel_size=3, n_conv_layers=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='concatenate',
                                 speaker_embedder_path=speaker_embedder_path,
                                 *args, **kwargs)

@register_model
def speaker_cct_7_3x1_32_sum(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                             speaker_embedder_path=None,
                             *args, **kwargs):
    return speaker_cct_7(arch='speaker_cct_7_3x1_32_sum', pretrained=pretrained, progress=progress,  pretrained_arch="cct_7_3x1_32_c100",
                                 kernel_size=3, n_conv_layers=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='sum',
                                 speaker_embedder_path=speaker_embedder_path,
                                *args, **kwargs)

@register_model
def speaker_cct_7_3x1_32_average(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                                 speaker_embedder_path=None,
                                 *args, **kwargs):
    return speaker_cct_7(arch='speaker_cct_7_3x1_32_average', pretrained=pretrained, progress=progress,  pretrained_arch="cct_7_3x1_32_c100",
                                 kernel_size=3, n_conv_layers=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='average',
                                 speaker_embedder_path=speaker_embedder_path,
                                 *args, **kwargs)

@register_model
def speaker_cct_4_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                         speaker_embedder_path=None,
                         *args, **kwargs):
    return speaker_cct_4(arch='speaker_cct_7_3x1_32', pretrained=pretrained, progress=progress,  pretrained_arch="",
                                 kernel_size=3, n_conv_layers=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='concatenate',
                                 speaker_embedder_path=speaker_embedder_path,
                                 *args, **kwargs)

@register_model
def speaker_cct_4_3x1_32_sum(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                             speaker_embedder_path=None,
                             *args, **kwargs):
    return speaker_cct_4(arch='speaker_cct_7_3x1_32_sum', pretrained=pretrained, progress=progress,  pretrained_arch="",
                                 kernel_size=3, n_conv_layers=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='sum',
                                 speaker_embedder_path=speaker_embedder_path,
                                 *args, **kwargs)

@register_model
def speaker_cct_4_3x1_32_average(pretrained=False, progress=False, img_size=32, positional_embedding='learnable', num_classes=10,
                                 speaker_embedder_path=None,
                                 *args, **kwargs):
    return speaker_cct_4(arch='speaker_cct_7_3x1_32_average', pretrained=pretrained, progress=progress,  pretrained_arch="",
                                 kernel_size=3, n_conv_layers=1,
                                 img_size=img_size, positional_embedding=positional_embedding,
                                 num_classes=num_classes, type='average',
                                 speaker_embedder_path=speaker_embedder_path,
                                 *args, **kwargs)
