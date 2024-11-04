from dataclasses import dataclass
from tqdm import tqdm
from datasets import load_dataset


@dataclass
class AishellItem:
    file: str
    audio: dict
    file2: str
    instruction: str
    label: str


def get_dataset_iterator_aishell():
    print("Loading AISHELL dataset...")
    ds = load_dataset("DynamicSuperbPrivate/SpeakerVerification_Aishell1Train", streaming=True)
    print("AISHELL dataset successfully loaded")

    for item in tqdm(ds['train'], desc="Processing samples"):
        yield AishellItem(
            file=item['file'],
            audio=item['audio'],
            file2=item['file2'],
            instruction=item['instruction'],
            label=item['label']
        )
