import torch
from fairseq.models import FairseqEncoderDecoderModel, FairseqEncoder, FairseqDecoder
from transformers import AutoModelForCausalLM, AutoTokenizer


class DummyEncoder(FairseqEncoder):
    def forward(self, *args, **kwargs):
        return None

    def reorder_encoder_out(self, encoder_out, new_order):
        return encoder_out


class DummyDecoder(FairseqDecoder):
    def forward(self, *args, **kwargs):
        return None

    def extract_features(self, prev_output_tokens, encoder_out=None, **kwargs):
        return None

    def output_layer(self, features, **kwargs):
        return None


# Основной класс модели
class FairseqQwenAudioModel(FairseqEncoderDecoderModel):
    def __init__(self, args=None, task=None):
        super().__init__(DummyEncoder(None), DummyDecoder(None))
        model_path = args.qwen_audio_model_path if args and args.qwen_audio_model_path else "Qwen/Qwen-Audio"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        ).eval()

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--qwen_audio_model_path',
            type=str,
            default='Qwen/Qwen-Audio',
            help='Path to the Qwen-Audio model'
        )

    def forward(self, src_tokens, src_lengths=None, prev_output_tokens=None, **kwargs):
        query = kwargs.get("query")
        audio_info = self.tokenizer.process_audio(query)
        inputs = self.tokenizer(query, return_tensors="pt", audio_info=audio_info).to(self.model.device)
        return self.model.generate(**inputs, audio_info=audio_info)

    def generate(self, query):
        audio_info = self.tokenizer.process_audio(query)
        inputs = self.tokenizer(query, return_tensors="pt", audio_info=audio_info).to(self.model.device)
        pred = self.model.generate(**inputs, audio_info=audio_info)
        return self.tokenizer.decode(pred[0], skip_special_tokens=False, audio_info=audio_info)

    def build_model(args, task):
        return FairseqQwenAudioModel(args, task)
