from dataclasses import dataclass
from datasets import load_dataset


@dataclass
class LJSpeechItem:
    id: str
    audio: bytes
    sampling_rate: int
    file: str
    text: str
    normalized_text: str
    duration: float


def get_dataset_iterator_ljspeech(load_audio=True):
    print("Загружаем датасет 'LJSpeech'...")
    ds = load_dataset("keithito/lj_speech")
    print("Датасет 'LJSpeech' был успешно загружен")

    for item in ds['train']:
        audio_dict = item['audio'] # np.array
        yield LJSpeechItem(
            id=item['id'],
            audio=audio_dict['array'].tobytes() if load_audio else None,
            sampling_rate=audio_dict['sampling_rate'],
            file=item['file'],
            text = item['text'],
            normalized_text=item['normalized_text'],
            duration=audio_dict['array'].shape[0] / audio_dict['sampling_rate']
        )
