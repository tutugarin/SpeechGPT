import logging
import pandas as pd
import plotly.express as px
import streamlit as st
from datasets import load_dataset

# Настройка логирования
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("streamlit_logger")
logger.info("Приложение запущено.")

# Streamlit UI
st.title("Dataset EDA and Inference App")
logger.info("UI загружен.")

# Выбор датасета
dataset_option = st.selectbox(
    "Выберите датасет для анализа",
    ("FSD50K", "AISHELL", "Alpaca", "Audiocaps")
)
logger.info("Выбран датасет: %s", dataset_option)

try:
    if dataset_option == "FSD50K":
        logger.info("Загрузка FSD50K dataset...")
        ds = load_dataset("CLAPv2/FSD50K")['train']
        df = pd.DataFrame(ds)
        st.write("### EDA для FSD50K")
        st.write(df.head())
        logger.info("Размер данных FSD50K: %s", df.shape)

        # Графики
        fig = px.histogram(df, x="audio_len", title="Распределение длительности аудио")
        st.plotly_chart(fig)
        logger.info("Гистограмма аудио длительности создана.")

    elif dataset_option == "AISHELL":
        logger.info("Загрузка AISHELL dataset...")
        ds = load_dataset("DynamicSuperbPrivate/SpeakerVerification_Aishell1Train")['train']
        df = pd.DataFrame(ds)
        st.write("### EDA для AISHELL")
        st.write(df.head())
        logger.info("Размер данных AISHELL: %s", df.shape)

        # Графики
        fig = px.histogram(df, x="label", title="Распределение меток")
        st.plotly_chart(fig)
        logger.info("Гистограмма меток создана.")

    elif dataset_option == "Alpaca":
        logger.info("Загрузка Alpaca dataset...")
        ds = load_dataset("yahma/alpaca-cleaned")['train']
        df = pd.DataFrame(ds)
        st.write("### EDA для Alpaca")
        st.write(df.head())
        logger.info("Размер данных Alpaca: %s", df.shape)

        # Графики
        fig = px.histogram(df, x="instruction", title="Распределение инструкций")
        st.plotly_chart(fig)
        logger.info("Гистограмма инструкций создана.")

    elif dataset_option == "Audiocaps":
        logger.info("Загрузка Audiocaps dataset...")
        ds = load_dataset("d0rj/audiocaps")['train']
        df = pd.DataFrame(ds)
        st.write("### EDA для Audiocaps")
        st.write(df.head())
        logger.info("Размер данных Audiocaps: %s", df.shape)

        # Графики
        fig = px.histogram(df, x="start_time", title="Распределение начальных времен аудиоклипов")
        st.plotly_chart(fig)
        logger.info("Гистограмма начальных времен аудиоклипов создана.")

    # Инференс семпла
    st.write("### Инференс семпла")
    selected_sample = st.number_input("Выберите индекс семпла", min_value=0, step=1, value=0)
    if st.button("Инференс семпла"):
        sample = df.iloc[selected_sample]
        st.write("Выбранный семпл:", sample)
        logger.info("Инференс семпла: %s", sample)

except KeyError as e:
    logger.error("Ошибка при доступе к ключу набора данных: %e", e)
    st.error("Ошибка при доступе к ключу данных: %e", e)

except ValueError as e:
    logger.error("Ошибка при создании графика: %e", e)
    st.error("Ошибка при создании графика: %e", e)

except IndexError as e:
    logger.error("Ошибка при доступе к индексу данных: %e", e)
    st.error("Ошибка при доступе к индексу данных: %e", e)

st.write("Приложение завершило работу.")
logger.info("Приложение завершено.")
