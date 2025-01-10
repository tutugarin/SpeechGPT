import hashlib

from dataclasses import dataclass

@dataclass
class ExampleItem:
    id: str
    text: str


def get_dataset_iterator_example(n_rows: int = 100):
    for i in range(n_rows):
        text = f"This is an example text #{i}"
        row_id = hashlib.sha256(text.encode()).hexdigest()
        row = ExampleItem(id=row_id, text=text)
        yield row
