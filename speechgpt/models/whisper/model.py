import logging
from typing import Dict, Optional

import soundfile as sf
import torch
from torch import Tensor
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
    FairseqEncoder,
    FairseqDecoder
)
from transformers import AutoProcessor

try:
    from transformers import WhisperForConditionalGeneration
    has_hf = True
except ImportError:
    has_hf = False

logger = logging.getLogger(__name__)

# Заглушка для энкодера
class DummyEncoder(FairseqEncoder):
    def forward(self, *args, **kwargs):
        return None

# Заглушка для декодера
class DummyDecoder(FairseqDecoder):
    def forward(self, *args, **kwargs):
        return None

@register_model("whisper-turbo")
class HuggingFaceWhisperModel(FairseqEncoderDecoderModel):
    def __init__(self, args):
        dummy_encoder = DummyEncoder(None)
        dummy_decoder = DummyDecoder(None)
        super().__init__(dummy_encoder, dummy_decoder)
        if not has_hf:
            raise ImportError(
                "\n\nPlease install huggingface/transformers with:"
                "\n\n  pip install transformers"
                "\n\nOr to make local edits, install the submodule:"
                "\n\n  git submodule update --init fairseq/models/huggingface/transformers"
            )
        self.args = args
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

        return logits, None, None, None

    def parse_waveform(self, file: str, **kwargs):
        waveform, sampling_rate = sf.read(file)
        waveform = torch.tensor(waveform).unsqueeze(0).float()
        inputs = self.processor(waveform.squeeze(0), sampling_rate=sampling_rate, return_tensors="pt")
        return inputs['input_features']

    def generate(self, audio_tokens=None, text=False, skip_special_tokens=True, file=None, **kwargs):
        if audio_tokens is None and file is None:
            raise ValueError("audio_tokens or file must not be None")
        
        if file is not None:
            audio_tokens = self.parse_waveform(file)
            
        self.eval()
        
        with torch.no_grad():
            generated_ids = self.model.generate(audio_tokens, **kwargs)

        if not text:
            return generated_ids

        return self.processor.batch_decode(generated_ids, skip_special_tokens=skip_special_tokens)

def default_architecture(args):
    args.load_hf_whisper_from = getattr(args, 'load_hf_whisper_from', 'openai/whisper-large-v3-turbo')
    args.generate_text = getattr(args, 'generate_text', False)

if __name__ == "__main__":
    print("ok")
