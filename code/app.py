import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import librosa

from song_splitter import audio_framer, plot_decomp, get_recommendations

import pickle
if 'reset' not in st.session_state:
    st.session_state['reset'] = False

if st.session_state['reset']:
    for key in st.session_state.keys():
        del st.session_state[key]


if 'mel' not in st.session_state:
    st.session_state['mel'] = False

if 'pca' not in st.session_state:
    st.session_state['pca'] = False

if 'rec' not in st.session_state:
    st.session_state['rec'] = False

if 'comp1' not in st.session_state:
    st.session_state['comp1'] = False

if 'comp2' not in st.session_state:
    st.session_state['comp2'] = False

if 'generated' not in st.session_state or st.session_state['rec']:
    st.session_state['generated'] = False


st.write(st.session_state['pca'], st.session_state['mel'], st.session_state['rec'])

st.markdown(
    """
    ## Song Analysis and Recommendations
    """
    )

df = pd.concat(
    [pd.read_csv('./data/decomposed/df_{}.csv'.format(x)) for x in range(10)]
    ).reset_index(drop = True)

with open('./data/pickle/preprocessing.pkl', 'rb') as f:
    pre = pickle.load(f)

input_song = st.file_uploader("Upload your song:", key = 'input')

if input_song:
    song, sr = librosa.load(input_song)
    st.audio(song, sample_rate = sr)


    st.markdown("### Waveplot")
    st.write('''
            The typical way to view a song is a waveplot like this:
            ''')
    fig, ax = plt.subplots()
    wp = librosa.display.waveshow(song, ax = ax)
    st.write(fig)
    st.write('''
            This isn't particularly helpful outside of telling us 
            how much noise is at each timestamp. So instead we can...
            ''')
    view_spec = st.button('Turn it into a spectogram!', key = 'mel')
    st.write(st.session_state['mel'])
    if st.session_state['mel'] or st.session_state['pca'] or st.session_state['rec'] or (st.session_state['comp1'] or st.session_state['comp2']): 
        mel = librosa.feature.melspectrogram(y = song, sr = sr)

        fig, ax = plt.subplots()
        librosa.display.specshow(librosa.power_to_db(mel, ref = np.max), x_axis = 'time', y_axis = 'mel', ax = ax)
        st.write(fig)

        st.write(
            '''
            A spectogram lets us view our song in terms of frequency (Hz), loudness (dB), and time. 
            You can see the same general trends that we saw in the waveplot and much more. This allows
            us to understand which frequency ranges our song is louder at for any given span of time.
            '''
        )

        st.write(
            '''
            For our purposes, we will be turning our song into 3 second chunks then aggregating that and decomposing it.
            Basically, we'll turn this song which was {} seconds long into {} 3-second chunks instead and then
            apply PCA to it. That means we can visualize our song in terms of its principal components:
            '''.format(song.shape[0] // sr, mel.shape[1] // 129)
        )
        
        decomp_song = st.button('View decomposed song', key = 'pca')
        st.write(st.session_state['pca'], not st.session_state['mel'])

        if  st.session_state['pca'] and not st.session_state['mel'] or st.session_state['rec'] or (st.session_state['comp1'] or st.session_state['comp2']):
            song_frame = audio_framer(librosa.power_to_db(mel.T), -1, mel.shape[1] // 129)

            decomped = pd.DataFrame(pre.transform(song_frame.loc[:, list(range(256))]))
            comp_1 = st.selectbox(
                'X-axis component',
                (1, 2, 3, 4, 5),
                index = 0,
                key = 'comp1'
            )
            comp_2 = st.selectbox(
                'Y-axis component',
                (1, 2, 3, 4, 5),
                index = 1,
                key = 'comp2'
            )
            if comp_1 and comp_2:
                fig, ax = plot_decomp(decomped, comp_1-1, comp_2-1)
            else:
                fig, ax = plot_decomp(decomped)

            st.write(fig)

            st.write('With these components we can finally get recommendations.')
            st.write(not st.session_state['pca'], not st.session_state['mel'], st.session_state['rec'])
            recs = st.button('Generate Recommendations', key = 'rec')

            st.write('generated?')
            st.write(st.session_state['generated'])
            

            if ((not st.session_state['pca'] and not st.session_state['mel']) and st.session_state['rec'] and (
                st.session_state['comp1'] or st.session_state['comp2']) or st.session_state['generated']):

                recos = get_recommendations(decomped, df)
                st.session_state['generated'] = True

                md  = """|Rank|Song|Genre|Score|\n|-|-|-|-|"""
                for idx in range(recos.shape[0]):
                    row = recos.iloc[idx, :]

                    md += "\n|{}|{} by {} on {}|{}|{}|".format(
                        idx + 1,
                        row['title'],
                        row['name'],
                        row['album'],
                        row['genre_top'],
                        round(row['cos'], 4)
                    )

                st.markdown(md)

                st.button("RESET", key = 'reset')