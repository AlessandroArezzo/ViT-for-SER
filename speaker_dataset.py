import librosa
import torch
import torch.utils.data as data
import numpy as np
import logging

import torchaudio
#import wavencoder
from PIL import Image

from timm.data.parsers import create_parser
from torchaudio.models import Wav2Vec2Model
import python_speech_features
#from wav2vec2_stt import Wav2Vec2STT

_logger = logging.getLogger(__name__)

_ERROR_RETRY = 50
class SpeakerDataset(data.Dataset):

    def __init__(
            self,
            root,
            parser=None,
            class_map=None,
            load_bytes=False,
            transform=None,
            target_transform=None,
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = bundle.get_model()

        #self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        input, target = self.parser[index]
        img = input[0]
        wav_file = input[1]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        """
        speaker_encoder = wavencoder.models.Wav2Vec(pretrained=True)
        """

        #wav_data = torch.from_numpy(wav_data[np.newaxis, ...])
        #print(wav_data.shape)
        #cp = torch.load('./wav2vec_large.pt')

        #model = Wav2Vec2Model.build_model(self.cp['args'], task=None)
        #model.load_state_dict(self.cp['model'])
        #model.eval()

        # ! SPEAKER ENCODING !

        waveform, _ = torchaudio.load(wav_file)
        with torch.inference_mode():
            wav_features, _ = self.model.feature_extractor(waveform)


        return (img, wav_features), target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)