# pylint: disable=R0903

from dataclasses import dataclass
from typing import Iterator

@dataclass
class CovostItem:
    """Класс, представляющий элемент датасета."""
    id: int
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

    def load_data(self, dataset_type: str, n_rows: int = None) -> Iterator[CovostItem]:
        """
        Построчное чтение данных из файла в зависимости от типа датасета.
        
        :param dataset_type: Тип датасета ('train', 'dev', 'test').
        :param n_rows: Ограничение на количество строк для чтения.
        :return: Итератор с элементами CovostItem.
        """
        file_path = self.file_paths.get(dataset_type)
        if not file_path:
            raise ValueError(f"Путь для {dataset_type} не найден. Проверьте file_paths.")
        # Открываем файл и читаем построчно
        with open(file_path, 'r', encoding='utf-8') as file:
            # Пропускаем заголовок и проверяем нужные колонки
            header = file.readline().strip().split('\t')
            indices = {col: header.index(col)
                        for col in ['path', 'sentence', 'translation', 'client_id']}

            # Чтение строк с ограничением по n_rows
            for row_count, line in enumerate(file):
                if n_rows is not None and row_count >= n_rows:
                    break

                # Получаем значения для каждой колонки
                row = line.strip().split('\t')
                yield CovostItem(
                    id=row_count,
                    path=row[indices['path']],
                    sentence=row[indices['sentence']],
                    translation=row[indices['translation']],
                    client_id=row[indices['client_id']]
                )
