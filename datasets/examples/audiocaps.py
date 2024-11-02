from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class Audiocap:
    """
    Класс, представляющий одну запись из датасета Audiocaps.

    Атрибуты:
        audiocap_id (str): Уникальный идентификатор аудиокапса.
        youtube_id (str): Идентификатор видео на YouTube.
        start_time (float): Время начала аудио в секундах.
        caption (str): Описание аудио.
    """
    audiocap_id: str
    youtube_id: str
    start_time: float
    caption: str

def get_dataset_iterator_audiocaps():
    """
    Функция для загрузки датасета Audiocaps и итерации по его записям.
    
    Каждая запись представлена объектом класса `Audiocap`, 
    содержащим следующие поля:
    - audiocap_id (уникальный идентификатор аудиокапса),
    - youtube_id (идентификатор видео на YouTube),
    - start_time (время начала аудио в секундах),
    - caption (описание аудио).

    Возвращает:
        generator: Итератор по объектам типа `Audiocap`.
    """

    print("Загрузка датасета Audiocaps...")

    ds = load_dataset("d0rj/audiocaps")
    print("Датасет Audiocaps успешно загружен")

    for item in ds['train']:
        yield Audiocap(
            audiocap_id=item['audiocap_id'],
            youtube_id=item['youtube_id'],
            start_time=item['start_time'],
            caption=item['caption']
        )
        