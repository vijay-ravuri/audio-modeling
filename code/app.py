import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import librosa

from song_splitter import audio_framer, plot_decomp, get_recommendations

import pickle


df = pd.concat([pd.read_csv('./data/decomposed/df_{}.csv'.format(x)) for x in range(10)]).reset_index(drop = True)

with open('./data/pickle/preprocessing.pkl', 'rb') as f:
    pre = pickle.load(f)


st.write('Got data read in... what ya got?')

input_song = st.file_uploader("gimme a song")

if input_song:
    song, sr = librosa.load(input_song)
    st.audio(song, sample_rate = sr)
    fig, ax = plt.subplots()
    wp = librosa.display.waveshow(song, ax = ax)
    st.write(fig)

    mel = librosa.power_to_db(librosa.feature.melspectrogram(y = song, sr = sr), ref = np.max)

    fig, ax = plt.subplots()
    librosa.display.specshow(mel, x_axis = 'time', y_axis = 'mel', ax = ax)
    st.write(fig)

    song_frame = audio_framer(mel.T, -1, mel.shape[1] // 129)
    
    decomped = pd.DataFrame(pre.transform(song_frame.loc[:, list(range(256))]))

    fig, ax = plot_decomp(decomped)

    st.write(fig)

    st.write(get_recommendations(decomped, df))