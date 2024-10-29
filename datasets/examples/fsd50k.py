from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class FSD50KItem:
    """
    Класс, представляющий одну запись из датасета FSD50K.

    Атрибуты:
        index (int): Уникальный индекс аудиофайла в датасете.
        datasetname (str): Название датасета (обычно "FSD50K").
        audio (str): Путь или идентификатор аудиофайла (можно загрузить как аудио).
        audio_len (float): Длительность аудиофайла в секундах.
        text (str): Текстовая метка или описание, связанное с аудио.
        raw_text (str): Сырые текстовые данные, связанные с записью аудио.
    """
    index: int
    datasetname: str
    audio: dict
    audio_len: float
    text: str
    raw_text: str

def get_dataset_iterator_fsd50k():
    """
    Функция для загрузки датасета FSD50K и итерации по его записям.
    
    Каждая запись представлена объектом класса `FSD50KItem`, 
    содержащим следующие поля:
    - index (уникальный индекс аудиофайла),
    - datasetname (название датасета, например, "FSD50K"),
    - audio (путь или идентификатор аудиофайла),
    - audio_len (длительность аудиофайла),
    - text (текстовое описание),
    - raw_text (сырые текстовые данные).

    Возвращает:
        generator: Итератор по объектам типа `FSD50KItem`.
    """

    print("Загрузка датасета FSD50K...")
    # Загрузка датасета с использованием библиотеки datasets
    ds = load_dataset("CLAPv2/FSD50K")
    print("Датасет FSD50K успешно загружен")

    # Итерация по записям датасета
    for item in ds['train']:
        # Возвращаем каждую запись как объект FSD50KItem
        yield FSD50KItem(
            index=item['index'],
            datasetname=item['datasetname'],
            audio=item['audio'],  # Это может быть путь или идентификатор аудиофайла
            audio_len=item['audio_len'],
            text=item['text'],
            raw_text=item['raw_text']
        )
