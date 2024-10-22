import re
from dataclasses import dataclass
from datasets import load_dataset


@dataclass
class SpeechInstructCrossModalItem:
    """
    Класс, представляющий одну запись из датасета 'SpeechInstruct', срез Cross-Modal.
    Каждая запись представляет собой тройку (D, U, T), соответственно description, unit_sequence, transcription
    Задачи: ASR, TTS

    NOTE: unit_sequence может быть как текстом, так и аудио-токенами в зависимости от задачи:
    - For a discrete unit sequence U and its associated transcription T we determine whether
      it will be used for constructing an ASR task or a TTS task based on the probability p.

    Такие айтемы имеют следующий формат:
    [Human]: {D} This is input: {U}<eoh> [SpeechGPT]: <sosp>{T}<eosp><eoa>.

    Атрибуты:
        prefix (str): system prompts.
        plain_text (str): полный текст
        description (str): пользовательская инструкция.
        unit_sequence (str): пользовательский запрос.
        transcription (str): ответ модели.
    """
    prefix: str
    plain_text: str
    description: str
    unit_sequence: str
    transcription: str


@dataclass
class SpeechInstructCoMItem:
    """
    Класс, представляющий одну запись из датасета 'SpeechInstruct', срез Chain-of-Modality.
    Каждая запись представляет собой речевую инструкцию пользователя, ее расшифровку tq,
    текстовый и аудио ответ модели, соответственно, ta и ua
    Задача: Speech Instruction-Speech Response, другие в наборе не представлены

    NOTE: в статье представлен следующий шаблон:
        [Human]: This is a speech instruction: {SpeechI}. And your response should be speech.
        You can do it step by step. You can first transcribe the instruction and get the text Instruction.
        Then you can think about the instruction and get the text response. Last, you should speak the
        response aloud <eoh>. [SpeechGPT]: [tq] {TextI}; [ta] {TextR}; [ua] {SpeechR}<eoa>.
    Однако в реальных данных форматирование другое

    Такие айтемы имеют следующий формат:
    [Human]: <sosp>{SpeechI}<eosp><eoh> [SpeechGPT]: {tq}; [ta] {ta}; [ua] <sosp>{ua}<eosp><eoa>.

    Атрибуты:
        prefix (str): system prompts.
        plain_text (str): полный текстю
        speech_instruction (str): пользовательская речевая инструкция.
        text_query (str): похоже на транскрипт пользовательской инструкции,
        text_answer (str): ta
        unit_answer (str): ua
    """
    prefix: str
    plain_text: str
    speech_instruction: str
    text_query: str
    text_answer: str
    unit_answer: str


@dataclass
class SpeechInstructChatItem:
    """
    Класс, представляющий одну запись из датасета 'SpeechInstruct', срез диалоги.
    Каждая запись представляет собой простые текстовые диалоги пользователя и модели
    Задача: диалоги

    NOTE: являются частью Cross-Modal

    Атрибуты:
        prefix (str): system prompts.
        plain_text (str): полный текст
    """
    prefix: str
    plain_text: str


def get_dataset_iterator_speechinstruct():
    print("Загружаем датасет 'SpeechInstructor'...")
    # Загрузка датасета через библиотеку datasets
    ds = load_dataset("fnlp/SpeechInstruct")
    print("Датасет 'SpeechInstructor' был успешно загружен")

    cross_modal_pat = r"\[Human\]: (.*) This is input: (.*) \[SpeechGPT\]: (.*)"
    chain_of_modality_pat = r"\[Human\]: (.*) \[SpeechGPT\]: (.*); \[ta\] (.*); \[ua\] (.*)"
    for item in ds['train']:
        prefix = item['prefix']
        plain_text = item['plain_text']

        g = re.search(cross_modal_pat, plain_text)
        if g is not None:
            yield SpeechInstructCrossModalItem(
                prefix=prefix,
                plain_text=plain_text,
                description=g.group(1),
                unit_sequence=g.group(2),
                transcription=g.group(3),
            )
            continue

        g = re.search(chain_of_modality_pat, plain_text)
        if g is not None:
            yield SpeechInstructCoMItem(
                prefix=prefix,
                plain_text=plain_text,
                speech_instruction=g.group(1),
                text_query=g.group(2),
                text_answer=g.group(3),
                unit_answer=g.group(4),
            )
            continue

        yield SpeechInstructChatItem(
            prefix=prefix,
            plain_text=plain_text,
        )
