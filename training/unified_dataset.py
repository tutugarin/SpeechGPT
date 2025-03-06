from datasets import Dataset, load_dataset, concatenate_datasets, Audio
from typing import Optional, List, Dict
import itertools
import gc


class UnifiedSpeechDataset:
    def __init__(self, token: Optional[str] = None, lang: str = "ru",
                 split: str = "train", subset: str = ''):
        self.token = token
        self.split = split
        self.subset = subset
        self.prompt_lang = lang
        self.datasets = {}
        self.task_datasets = {
            "asr": {},
            "translation": {}
        }

        self._language_names = {
            'ru': {'ru': 'русский', 'en': 'Russian'},
            'en': {'ru': 'английский', 'en': 'English'}
        }

        self._load(lang)

    def _load(self, lang: str = "ru"):
        if lang == "ru":
            self._load_all_russian_datasets()
        else:
            self._load_all_english_datasets()

    def _get_language_name(self, lang_code: str) -> str:
        primary_lang = lang_code.split('_')[0].lower()
        return self._language_names.get(primary_lang, {}).get(self.prompt_lang, lang_code)

    def _add_fleurs_asr(self, language_code: str = "ru_ru"):
        dataset_name = f"fleurs_asr_{language_code}"
        if dataset_name not in self.datasets:
            print(f"Загрузка датасета FLEURS ASR для {language_code}, сплит {self.split}...")
            dataset = self._load_dataset_from_hf(
                "google/xtreme_s",
                f"fleurs.{language_code}",
                self.subset)

            def process(example):
                lang_name = self._get_language_name(example['language'])
                if self.prompt_lang == "ru":
                    prompt = f"Распознай эту речь на {lang_name}"
                else:
                    prompt = f"Transcribe this speech in {lang_name}"

                return {
                    "text_prompt": prompt,
                    "speech_input": example['audio']['array'],
                    "text_response": example['raw_transcription'],
                    # "metadata": {
                    #     "source": "fleurs",
                    #     "language": example['language'],
                    #     "original_id": example.get('id', None),
                    #     "gender": example.get('gender', None)
                    # }
                }

            print("Загрузка создание промптов для Fleurs ASR ...")
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
            dataset = dataset.map(
                process,
                remove_columns=[col for col in dataset.column_names
                                if col not in ["text_prompt", "speech_input", "text_response"]],
                num_proc=1
            )

            self.datasets[dataset_name] = dataset
            self.task_datasets["asr"][dataset_name] = dataset
            gc.collect()
            print(f"Добавлен датасет FLEURS ASR с {len(dataset)} примерами")

    def _add_common_voice(self, language_code: str = "ru"):
        dataset_name = f"common_voice_{language_code}"
        if dataset_name not in self.datasets:
            print(f"Загрузка датасета Common Voice для {language_code}, сплит {self.split}...")
            dataset = self._load_dataset_from_hf(
                "mozilla-foundation/common_voice_17_0",
                language_code,
                self.subset)

            def process(example):
                lang_name = self._get_language_name(language_code)
                if self.prompt_lang == "ru":
                    prompt = f"Распознай эту речь на {lang_name}"
                else:
                    prompt = f"Transcribe this speech in {lang_name}"

                return {
                    "text_prompt": prompt,
                    "speech_input": example['audio']['array'],
                    "text_response": example['sentence'],
                    # "metadata": {
                    #     "source": "common_voice",
                    #     "language": language_code,
                    #     "client_id": example.get('client_id', None),
                    #     "gender": example.get('gender', None),
                    #     "age": example.get('age', None),
                    #     "accent": example.get('accent', None)
                    # }
                }

            print("Загрузка создание промптов для Common Voice ...")
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
            dataset = dataset.map(
                process,
                remove_columns=[col for col in dataset.column_names
                                if col not in ["text_prompt", "speech_input", "text_response"]],
                num_proc=1
            )

            self.datasets[dataset_name] = dataset
            self.task_datasets["asr"][dataset_name] = dataset
            print(f"Добавлен датасет Common Voice с {len(dataset)} примерами")

    def _add_covost2(self, src_lang: str = "ru", tgt_lang: str = "en"):
        dataset_name = f"covost2_{src_lang}_{tgt_lang}"
        if dataset_name not in self.datasets:
            print(f"Загрузка датасета CoVoST2 для перевода с {src_lang} на {tgt_lang}, сплит {self.split}...")
            dataset = self._load_dataset_from_hf(
                "fixie-ai/covost2",
                f"{src_lang}_{tgt_lang}",
                self.subset)

            def process_covost2(example):
                src_name = self._get_language_name(src_lang)
                tgt_name = self._get_language_name(tgt_lang)

                if self.prompt_lang == "ru":
                    prompt = f"Переведи эту речь с {src_name} на {tgt_name}"
                else:
                    prompt = f"Translate this speech from {src_name} to {tgt_name}"

                return {
                    "text_prompt": prompt,
                    "speech_input": example['audio']['array'],
                    "text_response": example['translation'],
                    # "metadata": {
                    #     "source": "covost2",
                    #     "source_language": src_lang,
                    #     "target_language": tgt_lang,
                    #     "original_sentence": example.get('sentence', None),
                    #     "id": example.get('id', None)
                    # }
                }

            print("Загрузка создание промптов для CoVoST2 ...")
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
            dataset = dataset.map(
                process_covost2,
                remove_columns=[col for col in dataset.column_names
                                if col not in ["text_prompt", "speech_input", "text_response"]],
                num_proc=1
            )

            self.datasets[dataset_name] = dataset
            self.task_datasets["translation"][dataset_name] = dataset
            print(f"Добавлен датасет CoVoST2 с {len(dataset)} примерами")

    def _load_dataset_from_hf(self, dataset: str, lang_code: str, slice: Optional[str] = ''):
        return load_dataset(
            dataset,
            lang_code,
            trust_remote_code=True,
            split=f'{self.split}{slice}',
            token=self.token
        )

    def get_unified_dataset(self, task: Optional[str] = None) -> Dataset:
        if task is not None:
            if task not in self.task_datasets:
                raise ValueError(f"Задача {task} не поддерживается. Выберите из 'asr' или 'translation'")

            if not self.task_datasets[task]:
                raise ValueError(f"Нет загруженных датасетов для задачи {task}. Сначала добавьте датасеты.")

            task_datasets_list = list(self.task_datasets[task].values())
            if len(task_datasets_list) == 1:
                return task_datasets_list[0]
            else:
                return concatenate_datasets(task_datasets_list)
        else:
            all_datasets_list = list(self.datasets.values())
            if not all_datasets_list:
                raise ValueError("Нет загруженных датасетов. Сначала добавьте датасеты.")

            if len(all_datasets_list) == 1:
                return all_datasets_list[0]
            else:
                return concatenate_datasets(all_datasets_list)

    def get_task_datasets(self) -> Dict[str, List[str]]:
        return {
            task: list(datasets.keys())
            for task, datasets in self.task_datasets.items()
            if datasets
        }

    def task_iterator(self, task: str, batch_size: int = 16):
        if task not in self.task_datasets:
            raise ValueError(f"Задача {task} не поддерживается. Выберите из 'asr' или 'translation'")

        if not self.task_datasets[task]:
            raise ValueError(f"Нет загруженных датасетов для задачи {task}. Сначала добавьте датасеты.")

        task_iterators = []
        for dataset in self.task_datasets[task].values():
            task_iterators.append(self._batched_iterator(dataset, batch_size))

        return itertools.chain.from_iterable(task_iterators)

    def task_dataset(self, task: str, dataset_name: Optional[str] = None) -> Dataset:
        if task is None or task == "":
            return self.get_unified_dataset()

        if task not in self.task_datasets:
            raise ValueError(f"Задача {task} не поддерживается. Выберите из 'asr' или 'translation'")

        if not self.task_datasets[task]:
            raise ValueError(f"Нет загруженных датасетов для задачи {task}. Сначала добавьте датасеты.")

        # Если конкретный датасет не указан, возвращаем объединенный датасет для задачи
        if dataset_name is None:
            task_datasets_list = list(self.task_datasets[task].values())
            if len(task_datasets_list) == 1:
                return task_datasets_list[0]
            else:
                return concatenate_datasets(task_datasets_list)

        # Если указан конкретный датасет
        if dataset_name not in self.task_datasets[task]:
            available_datasets = list(self.task_datasets[task].keys())
            raise ValueError(f"Датасет {dataset_name} не найден для задачи {task}. "
                             f"Доступные датасеты: {available_datasets}")

        return self.task_datasets[task][dataset_name]

    def _batched_iterator(self, dataset, batch_size):
        total_examples = len(dataset)
        for i in range(0, total_examples, batch_size):
            yield dataset[i:min(i + batch_size, total_examples)]

    def __iter__(self):
        batch_size = 16
        iterators = []
        for dataset in self.datasets.values():
            iterators.append(self._batched_iterator(dataset, batch_size))
        return itertools.chain.from_iterable(iterators)

    def __getitem__(self, index):
        unified_dataset = self.get_unified_dataset()
        return unified_dataset[index]

    def __len__(self):
        unified_dataset = self.get_unified_dataset()
        return len(unified_dataset)

    def _load_all_russian_datasets(self):
        self._add_fleurs_asr("ru_ru")
        self._add_common_voice("ru")
        self._add_covost2("ru", "en")

    def _load_all_english_datasets(self):
        self._add_fleurs_asr("en_en")
        self._add_common_voice("en")
        self._add_covost2("en", "ru")
        return self