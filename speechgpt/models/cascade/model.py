import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
sys.path.append('../../../..')

import torch
from torch import Tensor
from typing import Optional, Dict
from fairseq.models import BaseFairseqModel, register_model
from omegaconf import OmegaConf

from speechgpt.models.whisper.model import HuggingFaceWhisperModel
from speechgpt.models.qwen.model import HuggingFaceQwen2ForCausalLM


# @register_model("asr-llm-cascade-model")
class AsrLlmCascadeModel(BaseFairseqModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.asr = None
        self.llm = None
        self.asr_processor = None
        self.llm_tokenizer = None
        self.load_models(args)

    def load_models(self, args):
        self.asr = HuggingFaceWhisperModel.build_model(args, None)
        self.llm = HuggingFaceQwen2ForCausalLM.build_model(args, None)
        self.asr_processor = self.asr.processor
        self.llm_tokenizer = self.llm.tokenizer
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    @classmethod
    def build_model(cls, args=None, task=None):
        return cls(args)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--asr_config', type=str, default='openai/whisper-large-v3-turbo')
        parser.add_argument('--llm_config', type=str, default='Qwen/Qwen2-0.5B')
        parser.add_argument('--local_asr_weights', type=str, default=None)
        parser.add_argument('--local_llm_weights', type=str, default=None)

    def forward(
        self,
        src_tokens: Tensor,
        tgt_tokens: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        asr_output = self.asr(src_tokens, tgt_tokens, incremental_state)
        # Linear?
        # llm_output = self.llm(whisper_output, None, tgt_tokens)
        return asr_output

    @torch.no_grad()
    def generate_from_asr(
        self,
        input_tokens=None,
        text=False,
        skip_special_tokens=True,
        file=None,
        **kwargs
    ):
        if input_tokens is None and file is None:
            raise Exception("input_tokens or file must not be None")

        asr_output = self.asr.generate(input_tokens, text, skip_special_tokens, file, **kwargs)
        return asr_output

    @torch.no_grad()
    def generate(
        self,
        input_tokens=None,
        skip_special_tokens=True,
        file=None,
        **kwargs
    ):
        if input_tokens is None and file is None:
            raise Exception("input_tokens or file must not be None")

        text = True
        asr_texts = self.asr.generate(input_tokens, text, skip_special_tokens, file, **kwargs)

        llm_tok_outs = self.llm_tokenizer(asr_texts, padding=True, return_tensors="pt")

        # тут возникает ошибка что не реализован prepare_generated_input_ids
        generate_ids = self.llm.generate(llm_tok_outs.input_ids, attention_mask=llm_tok_outs.attention_mask, **kwargs)
        gen_texts = self.llm_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return gen_texts

    def get_text(self):
        return "text"

class Model:
    def predict(self, *args, **kwargs):
        return "pred"

    def generate(self, *args, **kwargs):
        return "generate"

def get_cascade_model():
    args = OmegaConf.create()
    args.llm_config = "Qwen/Qwen2-0.5B"
    args.asr_config = "openai/whisper-large-v3-turbo"
    cascade = AsrLlmCascadeModel.build_model(args)
    print("cascade model inited")

    return cascade

if __name__ == '__main__':
    _ = get_cascade_model()