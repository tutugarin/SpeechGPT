import os
import dotenv
import librosa
import soundfile

import asyncio
from aiogram import Bot
from aiogram import types
from aiogram import Dispatcher
from aiogram.filters import Command

from speechgpt.models import AsrLlmCascadeModel


os.makedirs("audio", exist_ok=True)
dotenv.load_dotenv()
bot = Bot(token=os.environ.get("BOT_TOKEN"))
dp = Dispatcher()


print("\nInitializing model...")
model = AsrLlmCascadeModel.build_model()
print("Model is running!\n")


# Заглушка для вашей модели (замените на вызов вашей модели)
def process_audio_with_model(wav_file: str) -> str:
    """На будущее: запрос к модели на другой машине"""
    print(f"Processing {wav_file}...")
    answ = model.generate(file=wav_file, max_new_tokens=64, top_p=0.95)[0]
    answ = ". ".join(answ.split(". ")[:-1]) + "." # Обрезка до последнего завершенного предложения
    print(f"Successfully processed {wav_file}")
    return answ


async def download(file_id, save_name):
    file = await bot.get_file(file_id)
    await bot.download_file(file.file_path, save_name)


def convert(save_name, wave_name):
    audio, sample_rate = librosa.load(save_name, sr=16000)
    soundfile.write(wave_name, audio, samplerate=16000)


@dp.message()
async def handle_audio(message: types.Message):
    save_name = None
    wave_name = None
    try:
        if message.voice:  # Голосовые сообщения
            file_id = message.voice.file_id
            save_name = f"{file_id}.ogg"
            wave_name = f"{file_id}.wav"

        elif message.document:  # Файлы
            file_id = message.document.file_id
            save_name = f"{message.document.file_name}"
            name, ext = os.path.splitext(save_name)
            if ext in [".mp3", ".wav", ".ogg"]:
                wave_name = f"{os.path.splitext(save_name)[0]}.wav"
            else:
                await message.reply("Message doesn't seem to be an audio. Please send audio.")

        else:
            await message.reply("Message doesn't seem to be an audio. Please send audio.")

        if save_name and wave_name:
            save_name = f"audio/{save_name}"
            wave_name = f"audio/{wave_name}"

            await download(file_id, save_name)
            convert(save_name, wave_name)

            result_text = process_audio_with_model(wave_name)
            await message.reply(result_text)

            for name in set([save_name, wave_name])
                os.remove(name)
        else:
            await message.reply("File error! Maybe you have sent something different from audio?")

    except Exception as e:
        print(f"Got exception:\n\t{e}")
        await message.reply("Sorry, I failed to process your audio :(")


async def start():
    await dp.start_polling(bot, skip_updates=True)


if __name__ == "__main__":
    print("Bot is now running")
    asyncio.run(start())
