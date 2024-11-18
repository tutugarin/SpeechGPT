from transformers import AutoProcessor
import logging
from typing import Dict
from typing import Optional

import soundfile as sf
import torch
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
)
from torch import Tensor
from transformers import AutoProcessor

try:
    from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperConfig
    has_hf = True
except ImportError:
    has_hf = False

logger = logging.getLogger(__name__)

from fairseq.models import FairseqEncoder, FairseqDecoder


# Костыль
class DummyEncoder(FairseqEncoder):
    def forward(self, *args, **kwargs):
        return None  # или возвращайте данные, если требуется.


# Костыль
class DummyDecoder(FairseqDecoder):
    def forward(self, *args, **kwargs):
        return None


@register_model("whisper-turbo")
class HuggingFaceWhisperModel(FairseqEncoderDecoderModel):
    def __init__(self, args):
        dummy_encoder = DummyEncoder(None)
        dummy_decoder = DummyDecoder(None)
        super().__init__(dummy_encoder, dummy_decoder)  # No encoder or decoder
        if not has_hf:
            raise ImportError(
                '\n\nPlease install huggingface/transformers with:'
                '\n\n  pip install transformers'
                '\n\nOr to make local edits, install the submodule:'
                '\n\n  git submodule update --init '
                'fairseq/models/huggingface/transformers'
            )
        self.args = args

        # Проверка аргумента audio_path
        if not args.audio_path:
            raise ValueError("The argument --audio-path must be specified and not empty.")

        self.load_model(args)

    def load_model(self, args):
        model_path = getattr(args, 'load_hf_whisper_from', '')
        assert model_path, "Model path must be specified in --load-hf-whisper-from"
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.config = self.model.config

    @staticmethod
    def add_args(parser):
        parser.add_argument('--load-hf-whisper-from', type=str, default='',
                            help='load Hugging Face pretrained Whisper from path')
        parser.add_argument('--audio-path', type=str, default='',
                            help='Path to the audio file for processing')
        parser.add_argument('--max-target-positions', type=int,
                            help='maximum target positions for the decoder')

    @classmethod
    def build_model(cls, args, task):
        default_architecture(args)
        return cls(args)

    def forward(
            self,
            src_tokens: Tensor,
            tgt_tokens: Optional[Tensor] = None,
            src_lengths: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        input_features = src_tokens

        if tgt_tokens is not None:
            outputs = self.model(
                input_features=input_features,
                decoder_input_ids=tgt_tokens,
            )
            logits = outputs.logits
        else:
            self.eval()
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_features=input_features
                )
            logits = generated_ids

        batch_size = input_features.size(0)
        seq_len = input_features.size(1)

        return tuple(
            (
                logits,  # seq_len x batch x hidden (logits or other output)
                None,
                None,
                None
            )
        )

    def parse_waveform(self, file: str, **kwargs):
        waveform, sampling_rate = sf.read(file)
        waveform = torch.tensor(waveform).unsqueeze(0).float()
        inputs = self.processor(waveform.squeeze(0), sampling_rate=sampling_rate, return_tensors="pt")
        waveform = inputs['input_features']

        return waveform

    def generate(self, audio_tokens=None, text=False, skip_special_tokens=True, file=None, **kwargs):
        if audio_tokens is None and file is None:
            raise Exception("audio_tokens or file must not be None")

        if file is not None:
            waveform = self.parse_waveform(file)
            audio_tokens = waveform

        self.eval()

        with torch.no_grad():
            generated_ids = self.model.generate(audio_tokens, **kwargs)

        if not text:
            return generated_ids

        return self.processor.batch_decode(generated_ids, skip_special_tokens)


def default_architecture(args):
    args.load_hf_whisper_from = getattr(args, 'load_hf_whisper_from', 'openai/whisper-large-v3-turbo')
    args.audio_path = getattr(args, 'audio_path', '')
    args.generate_text = getattr(args, 'generate_text', 'False')
