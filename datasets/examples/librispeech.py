from dataclasses import dataclass
from tqdm import tqdm
from datasets import load_dataset


@dataclass
class LibrispeechItem:
    file: str
    audio: dict
    id: str
    text: str


def get_dataset_iterator_librispeech():
    print("Loading Librispeech dataset...")
    ds = load_dataset("openslr/librispeech_asr", split='validation.clean', streaming=True, trust_remote_code=True)

    for item in tqdm(ds, desc="Processing samples"):
        yield LibrispeechItem(
            file=item['file'],
            audio=item['audio'],
            id=item['id'],
            text=item['text'],
        )
        