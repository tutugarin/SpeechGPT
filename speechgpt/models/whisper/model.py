import soundfile as sf
import torch
from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqEncoder,
    FairseqDecoder
)
from speechgpt.logger import get_logger

try:
    from transformers import WhisperForConditionalGeneration, AutoProcessor
    has_hf = True
except ImportError:
    has_hf = False

logger = get_logger()


# Заглушка для энкодера
class DummyEncoder(FairseqEncoder):
    """
    A dummy encoder for the Hugging Face Whisper model.
    """
    def forward(self, *args, **kwargs):
        _ = args, kwargs

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Dummy implementation of reorder_encoder_out.
        """
        _ = encoder_out, new_order

# Заглушка для декодера
class DummyDecoder(FairseqDecoder):
    """
    A dummy decoder for the Hugging Face Whisper model.
    """
    def forward(self, *args, **kwargs):
        _ = args, kwargs

    def extract_features(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Dummy implementation of extract_features.
        """
        _ = prev_output_tokens, encoder_out, kwargs

    def output_layer(self, features, **kwargs):
        """
        Dummy implementation of output_layer.
        """
        _ = features, kwargs


DEFAULT_ASR_WEIGHTS = "openai/whisper-large-v3-turbo"


class HuggingFaceWhisperModel(FairseqEncoderDecoderModel):
    """
    A wrapper class for the Hugging Face Whisper model.

    Args:
    - args (dict): a dictionary of arguments, including the path to the model weights
    - task (Task): a Fairseq task object

    Attributes:
    - model (WhisperForConditionalGeneration): the Whisper model
    - processor (AutoProcessor): the processor for the Whisper model
    - config (dict): the configuration of the Whisper model
    """

    def __init__(self, args=None, task=None):
        dummy_encoder = DummyEncoder(None)
        dummy_decoder = DummyDecoder(None)
        _ = task # для pylint'а
        super().__init__(dummy_encoder, dummy_decoder)
        if not has_hf:
            raise ImportError(
                "\n\nPlease install huggingface/transformers with:"
                "\n\n  pip install transformers"
                "\n\nOr to make local edits, install the submodule:"
                "\n\n  git submodule update --init fairseq/models/huggingface/transformers"
            )
        self.load_model(args)

    def load_model(self, args):
        """
        Load the Whisper model from the given path.

        Args:
        - args (dict): a dictionary of arguments, including the path to the model weights
        """
        if args and args.asr_config:
            model_path = args.asr_config
        else:
            model_path = DEFAULT_ASR_WEIGHTS

        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.config = self.model.config
        logger.info("Model loaded from %s", model_path)

    @staticmethod
    def add_args(parser):
        """
        Add arguments to the parser for the Hugging Face Whisper model.

        Args:
        - parser (argparse.ArgumentParser): the parser object
        """
        parser.add_argument('--load-hf-whisper-from', type=str, default='',
                            help='load Hugging Face pretrained Whisper from path')
        parser.add_argument('--max-target-positions', type=int,
                            help='maximum target positions for the decoder')

    @classmethod
    def build_model(cls, args=None, task=None):
        """
        Build the Hugging Face Whisper model.

        Args:
        - args (dict): a dictionary of arguments, including the path to the model weights
        - task (Task): a Fairseq task object

        Returns:
        - HuggingFaceWhisperModel: the built model
        """
        return cls(args, task)

    def forward(self, src_tokens, src_lengths=None, prev_output_tokens=None, **kwargs):
        """
        The forward pass for the Hugging Face Whisper model.

        Args:
        - src_tokens (torch.Tensor): the input tokens
        - src_lengths (torch.Tensor): the length of the input tokens
        - prev_output_tokens (torch.Tensor): the previous output tokens

        Returns:
        - logits (torch.Tensor): the output logits
        - None, None, None (tuple): the output lengths, padding mask, and attention mask
        """
        input_features = src_tokens

        if prev_output_tokens is not None:
            outputs = self.model(
                input_features=input_features,
                decoder_input_ids=prev_output_tokens,
            )
            logits = outputs.logits
        else:
            self.eval()
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_features=input_features
                )
            logits = generated_ids

        logger.debug("Forward pass complete, logits generated")
        return logits, None, None, None

    def parse_waveform(self, file: str, **kwargs):
        """
        Parse the waveform from the given file.

        Args:
        - file (str): the path to the file

        Returns:
        - input_features (torch.Tensor): the input features
        """
        _ = kwargs # для pylint'а

        waveform, sampling_rate = sf.read(file)
        waveform = torch.tensor(waveform).unsqueeze(0).float()
        inputs = self.processor(waveform.squeeze(0), sampling_rate=sampling_rate, return_tensors="pt")
        logger.info("Parsed waveform from file %s", file)
        return inputs['input_features']

    def generate(self, audio_tokens=None, text=False, skip_special_tokens=True, file=None, **kwargs):
        """
        Generate speech output from the given audio tokens.

        Args:
        - audio_tokens (torch.Tensor): the input audio tokens
        - text (bool): whether to return the output as text
        - skip_special_tokens (bool): whether to skip special tokens when generating text
        - file (str): the path to the file

        Returns:
        - generated_ids (torch.Tensor): the generated output tokens
        - generated_text (str): the generated output text
        """
        if audio_tokens is None and file is None:
            raise ValueError("audio_tokens or file must not be None")

        if file is not None:
            audio_tokens = self.parse_waveform(file)

        self.eval()

        with torch.no_grad():
            generated_ids = self.model.generate(audio_tokens, **kwargs)

        logger.debug("Generated speech output, length of generated tokens:%s", len(generated_ids))

        if not text:
            return generated_ids

        return self.processor.batch_decode(generated_ids, skip_special_tokens=skip_special_tokens)
