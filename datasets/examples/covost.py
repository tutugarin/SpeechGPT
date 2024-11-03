import hashlib

import pandas as pd
import hashlib
from dataclasses import dataclass
from typing import Iterator

@dataclass
class DatasetItem:
    """Класс, представляющий элемент датасета."""
    id: str
    path: str
    sentence: str
    translation: str
    client_id: str

class CoVoST2Dataset:
    """Класс для работы с основными файлами датасета CoVoST2 (train, dev, test)."""

    def __init__(self, file_paths: dict):
        """
        Инициализация класса с указанием путей к файлам для различных типов датасетов.

        :param file_paths: Словарь с путями к файлам для 'train', 'dev', 'test'.
        """
        self.file_paths = file_paths

    def load_data(self, dataset_type: str, n_rows: int = None) -> Iterator[DatasetItem]:
        """
        Построчное чтение данных из файла в зависимости от типа датасета.
        
        :param dataset_type: Тип датасета ('train', 'dev', 'test').
        :param n_rows: Ограничение на количество строк для чтения.
        :return: Итератор с элементами DatasetItem.
        """
        file_path = self.file_paths.get(dataset_type)
        if not file_path:
            raise ValueError(f"Путь для {dataset_type} не найден. Проверьте file_paths.")
        
        # Открываем файл и читаем построчно
        with open(file_path, 'r', encoding='utf-8') as file:
            # Пропускаем заголовок
            header = file.readline().strip().split('\t')
            
            # Проверяем, что файл содержит нужные колонки
            expected_columns = ['path', 'sentence', 'translation', 'client_id']
            if not all(col in header for col in expected_columns):
                raise ValueError(f"Файл {file_path} не содержит ожидаемых столбцов: {expected_columns}")

            # Определяем индексы нужных колонок
            indices = {col: header.index(col) for col in expected_columns}

            # Счетчик строк
            row_count = 0

            # Чтение строк с ограничением по n_rows
            for line in file:
                row = line.strip().split('\t')

                # Ограничение по количеству строк, если задано
                if n_rows is not None and row_count >= n_rows:
                    break

                # Получаем значения для каждой колонки
                path = row[indices['path']]
                sentence = row[indices['sentence']]
                translation = row[indices['translation']]
                client_id = row[indices['client_id']]
                
                # Генерация уникального ID на основе sentence и translation
                unique_text = f"{sentence} {translation}"
                row_id = hashlib.sha256(unique_text.encode()).hexdigest()

                # Возвращаем элемент DatasetItem
                yield DatasetItem(
                    id=row_id,
                    path=path,
                    sentence=sentence,
                    translation=translation,
                    client_id=client_id
                )

                row_count += 1

# file_paths = {
#     'train': "covost_v2.en_de.train.tsv",
#     'dev': "covost_v2.en_de.dev.tsv",
#     'test': "covost_v2.en_de.test.tsv"
# }

# dataset = CoVoST2Dataset(file_paths)

# # Выбор типа датасета для итерации
# dataset_type = 'train'  # Измените на 'dev' или 'test' при необходимости

# for item in dataset.load_data(dataset_type, n_rows=1):
#     print(item)