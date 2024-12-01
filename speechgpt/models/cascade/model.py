from fairseq.models import BaseFairseqModel, register_model
from torch import Tensor
from typing import Optional, Dict

from speechgpt.models.whisper.model import HuggingFaceWhisperModel
# импортировать свою модель


# класс для аргументов (возможно на будущее)
class Args:
    pass

@register_model("asr-llm-cascade-model")
class AsrLlmCascadeModel(BaseFairseqModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.asr = None
        self.llm = None
        self.load_models(args)

    def load_models(self, args):
        self.asr = HuggingFaceWhisperModel.build_model(Args, None)
        # self.llm = добавить модель

    @classmethod
    def build_model(cls, args=None, task=None):
        args = args or Args()
        return cls(args)

    def forward(
        self,
        src_tokens: Tensor,
        tgt_tokens: Optional[Tensor] = None,
        src_lengths: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """Форвард пасс метод (может использоваться при обучения, но для генерации 
        использовать generate"""
        whisper_output = self.asr(src_tokens, tgt_tokens, src_lengths, incremental_state)
        # добавть работу с моделью llm_output = self.llm(whisper_output, ...)

        # возвращать llm output
        return whisper_output


    def generate(self, input_tokens=None, text=False, skip_special_tokens=True, file=None, **kwargs):

        if input_tokens is None and file is None:
            raise Exception("input_tokens or file must not be None")
            
        whisper_output = self.asr.generate(input_tokens, text, skip_special_tokens, file, **kwargs)
        # добавть работу с моделью llm_output = self.llm(whisper_output, ...)

        # возвращать llm output
        return whisper_output
