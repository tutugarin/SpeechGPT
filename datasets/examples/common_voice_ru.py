from dataclasses import dataclass
from typing import Iterator, Optional
from datasets import load_dataset


@dataclass
class CommonVoiceRuItem:
    """
    Класс, представляющий одну запись из датасета 'Common Voice' на русском языке.

    Атрибуты:
        client_id (str): Уникальный идентификатор клиента, предоставившего запись.
        audio_path (str): Путь до аудио.
        audio (bytes): Аудиоданные в формате байтов.
        duration (float): Продолжительность аудио в секундах.
        sentence (str): Предложение, произнесенное в аудиозаписи.
        up_votes (int): Количество голосов "за" запись.
        down_votes (int): Количество голосов "против" записи.
        age (Optional[str]): Возраст говорящего (если доступно).
        gender (Optional[str]): Пол говорящего (если доступно).
        accent (Optional[str]): Акцент говорящего (если доступно).
        locale (str): Локализация, в которой была сделана запись.
        segment (Optional[str]): Чаще всего пустая строка (если доступно).
        sampling_rate (int): Частота дискретизации аудиозаписи.
    """
    client_id: str
    audio_path: str
    audio: Optional[bytes]
    duration: float
    sentence: str
    up_votes: int
    down_votes: int
    age: Optional[str]
    gender: Optional[str]
    accent: Optional[str]
    locale: str
    segment: Optional[str]
    sampling_rate: int


def get_dataset_iterator_common_voice_ru(
        token: str = "",
        load_audio: bool = False
) -> Iterator[CommonVoiceRuItem]:
    """
    Функция для загрузки датасета 'Common Voice' на русском языке и итерации по его записям.
    
    Каждая запись представлена объектом класса `CommonVoiceRuItem`, содержащим информацию о клиенте,
    аудиозаписи и произнесенном предложении.

    Параметры:
        token (str): Токен доступа для загрузки датасета. Должен быть предоставлен.
        load_audio (bool): Нужно ли загружать массив с аудиотреком. Требует очень много памяти.

    Возвращает:
        Iterator[CommonVoiceRuItem]: Итератор по объектам типа `CommonVoiceRuItem`.
    
    Исключения:
        ValueError: Если токен не предоставлен.
    """
    if not token:
        raise ValueError("Token must be provided to access the dataset.")

    print("Начинаем загрузку датасета Common Voice...")

    # Загружаем датасет
    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "ru",
        trust_remote_code=True,
        split='train',
        token=token
    )

    print("Загрузка датасета 'Common Voice' завершена.")

    for record in ds:
        audio_array = record['audio']['array']
        audio_path = record['audio']['path']
        sampling_rate = record['audio']['sampling_rate']

        yield CommonVoiceRuItem(
            client_id=record['client_id'],
            audio=audio_array.to_bytes() if load_audio else None,
            audio_path=audio_path,
            duration=audio_array.shape[0] / sampling_rate,
            sentence=record['sentence'],
            up_votes=record['up_votes'],
            down_votes=record['down_votes'],
            age=record.get('age'),
            gender=record.get('gender'),
            accent=record.get('accent'),
            locale=record['locale'],
            segment=record.get('segment'),
            sampling_rate=sampling_rate
        )
