from dataclasses import dataclass
from datasets import load_dataset


@dataclass
class AlpacaItem:
    """
    Класс, представляющий одну запись из датасета 'alpaca'.

    Атрибуты:
        instruction (str): описывает задачу, которую должна выполнить модель. 
                            Каждая из 52 тысяч инструкций уникальна.
        input (str): необязательный контекст или ввод для задачи.
                    Например, если инструкция — "Суммируйте следующую статью",
                                                то вводом является сама статья.
                    Примерно 40% примеров содержат поле input.
        output (str): Исправленный текст или ответ на задачу.
    """
    instruction: str
    input: str
    output: str

def get_dataset_iterator_alpaca():
    """
    Функция для загрузки датасета 'alpaca' и итерации по его записям.
    
    Каждая запись представлена объектом класса `AlpacaItem`, содержащим три поля:
    - instruction (инструкция для выполнения задачи),
    - input (входные данные, например, предложение с ошибками),
    - output (правильный вариант или ответ).

    Возвращает:
        generator: Итератор по объектам типа `AlpacaItem`.
    """

    print("Загружаем датасет 'alpaca'...")
    # Загрузка датасета через библиотеку datasets
    ds = load_dataset("yahma/alpaca-cleaned")
    print("Датасет 'alpaca' был успешно загружен")

    # Итерация по записям датасета
    for item in ds['train']:
        # Возвращаем каждую запись как объект AlpacaItem
        yield AlpacaItem(
            instruction=item['instruction'],
            input=item['input'],
            output=item['output']
        )
