import tempfile

import librosa
import pandas as pd
import scipy
import streamlit as st

from src.filtered_spec import plot_audio
from src.predict import predict, temp_predict
from src.utils import (
    create_audio_player,
    image_to_bytes,
    plot_melspec,
    plot_wave,
    get_table_download_link,
)

st.image("app/assets/Elephants-header.png")

st.header("Elephant Age Classification")

# Sidebar
st.sidebar.image("app/assets/header.png", use_column_width=True)
st.sidebar.header("About")
st.sidebar.markdown(
    "Elephant age classification application developed by Harvard Extension School"
)
st.sidebar.title("Created by")
st.sidebar.text("Yikun Shen\nLucy Tan\nKarma Tarap")
st.sidebar.title("With thanks to")
st.sidebar.text("Elephant Listening Project\nZambezi Partners\nMicrosoft Project 15")

uploaded_files = st.file_uploader(
    "Choose a wav file",
    accept_multiple_files=True,
    type=["wav"],
    help="Upload a rumble in wav format",
)
from pathlib import Path

import torch

if len(uploaded_files) == 1:
    st.write("Single file detected")
    y, sr = librosa.load(uploaded_files[0])
    st.subheader("Audio Player")
    st.audio(create_audio_player(y, sr))
    st.subheader("Raw Waveform")
    st.pyplot(plot_wave(y, sr))
    st.subheader("Mel Spectrogram")
    st.pyplot(plot_melspec(y, sr))
    st.subheader("Processed Spectrogram")
    st.pyplot(plot_audio(y, sr))

    prediction = temp_predict(uploaded_files[0])
    st.subheader(f"Prediction: {prediction}")

elif len(uploaded_files) > 1:
    st.write("Multiple files detected")
    preds_df = pd.DataFrame(
        [[f.name, temp_predict(f)] for f in uploaded_files],
        columns=["name", "prediction"],
    )
    if len(preds_df):
        if st.checkbox("Show Predictions"):
            preds_df
        download_button_str = get_table_download_link(
            preds_df, "predictions.xlsx", f"Download Predictions"
        )
        st.markdown(download_button_str, unsafe_allow_html=True)
