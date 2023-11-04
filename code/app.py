import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import librosa

from song_utils import audio_framer, plot_decomp, get_recommendations

import pickle

from audioread.exceptions import NoBackendError
from soundfile import LibsndfileError

"""
Streamlit allows for easy deployment of models for use but is not particularly friendly towards
walking through a process in the way that this app was written to do. As a result, this code has
a lot of redundancy and complex conditional statements in order to allow for a linear narrative as
a song is uploaded that culminates in generating recommendations. For future purposes, a different
library would be ideal for building an app of this nature.
"""

# Defining a number of session states to allow for interactability
if 'reset' not in st.session_state:
    st.session_state['reset'] = False

# Resets the app to as close to its original state as possible
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


st.markdown(
    """
    ## Song Analysis and Recommendations

    This app allows you to upload a song of your choosing and see how that song is processed and then generate
    recommendations for similar songs in our dataset. The process for how this system works is documented in the
    Recommender System Notebook. This applet does not save the song uploaded in any way once it is closed and any
    data uploaded or generated is deleted.
    """
    )

# Read in the 10-piece dataframe of our decomposed song data
df = pd.concat(
    [pd.read_csv('./data/decomposed/df_{}.csv'.format(x)) for x in range(10)]
    ).reset_index(drop = True)

# Open preprocessing pipeline for the input song
with open('./data/pickle/preprocessing.pkl', 'rb') as f:
    pre = pickle.load(f)

# Allow user to upload their song
input_song = st.file_uploader("Upload your song:", key = 'input')

# In a try-except block to catch if users upload a file that isn't a song
# or otherwise breaks the librosa load function.
try:
    if input_song:
        # Turn song into workable data
        song, sr = librosa.load(input_song)


        st.markdown("### Waveplot")
        st.write('''
                The typical way to view a song is a waveplot like this:
                ''')
        # Plot waveplot
        fig, ax = plt.subplots()
        wp = librosa.display.waveshow(song, ax = ax, alpha = 0.8)
        st.write(fig)
        st.write('''
                This isn't particularly helpful outside of telling us 
                how much noise is at each timestamp and some basic general
                trends in our song.
                
                 So instead we can...
                ''')
        view_spec = st.button('Turn it into a spectogram!', key = 'mel')

        if st.session_state['mel'] or st.session_state['pca'] or st.session_state['rec'] or (st.session_state['comp1'] or st.session_state['comp2']): 
            # Create and plot spectrogram
            mel = librosa.feature.melspectrogram(y = song, sr = sr)

            fig, ax = plt.subplots()
            gram = librosa.display.specshow(librosa.power_to_db(mel, ref = np.max), x_axis = 'time', y_axis = 'mel', ax = ax)
            plt.colorbar(gram)
            plt.title("Mel-spectrogram")
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


            if  st.session_state['pca'] and not st.session_state['mel'] or st.session_state['rec'] or (st.session_state['comp1'] or st.session_state['comp2']):
                # Break song into 3-second chunks 
                song_frame = audio_framer(librosa.power_to_db(mel.T), -1, mel.shape[1] // 129)

                # Decompose song using processing pipeline then plot components
                decomped = pd.DataFrame(pre.transform(song_frame.loc[:, list(range(256))]))
                # Allows user to select which component for the x-axis
                comp_1 = st.selectbox(
                    'X-axis component',
                    (1, 2, 3, 4, 5),
                    index = 0,
                    key = 'comp1'
                )
                # Allows user to select which component for the y-axis
                comp_2 = st.selectbox(
                    'Y-axis component',
                    (1, 2, 3, 4, 5),
                    index = 1,
                    key = 'comp2'
                )

                # Ensures the plot doesn't break in edge cases
                if comp_1 and comp_2:
                    fig, ax = plot_decomp(decomped, comp_1-1, comp_2-1)
                else:
                    fig, ax = plot_decomp(decomped)

                st.write(fig)
                st.write('''We can see various trends in the 3-second segments for our song using the different components.
                            Depending on the song we will see different trends in the different dimensions. Some dimensions will
                            show much more pronounced trends than others. Additionally, from how clumped together our points are we
                            can tell how self-similar our song is. A song with segments more similar to each other will tend to have
                            higher scores for whatever songs are recommended as similar to it.
                        ''')
                st.write('With these components we can finally get recommendations.')
                recs = st.button('Generate Recommendations', key = 'rec')
                

                if ((not st.session_state['pca'] and not st.session_state['mel']) and st.session_state['rec'] and (
                    st.session_state['comp1'] or st.session_state['comp2']) or st.session_state['generated']):


                    # Generates recommendations
                    recos = get_recommendations(decomped, df)
                    # This variable helps us keep the generated recommendations visible if
                    # something from earlier has been changed.
                    st.session_state['generated'] = True

                    # Creates markdown table for the recommendations
                    md  = """|Rank|Song|Genre|Score|\n|-|-|-|-|""" + "".join([
                        "\n|{}|{} by {} on {}|{}|{}|".format(
                            idx + 1,
                            recos.iloc[idx, 1],
                            recos.iloc[idx, 2],
                            recos.iloc[idx, 3],
                            recos.iloc[idx, 4],
                            round(recos.iloc[idx, 5], 4)) for idx in range(recos.shape[0])
                        ])
                    # Below is the for-loop that does the same thing as above for readability
                    # The list generation was made to make the app feel slightly less sluggish
                    # for idx in range(recos.shape[0]):
                    #     row = recos.iloc[idx, :]

                    #     md += "\n|{}|{} by {} on {}|{}|{}|".format(
                    #         idx + 1,
                    #         row['title'],
                    #         row['name'],
                    #         row['album'],
                    #         row['genre_top'],
                    #         round(row['cos'], 4)
                    #     )

                    st.markdown(md)

                    st.markdown('''
                            These recommendations are limited to only songs in the set of data we had access to which is a subset
                            of royalty free songs available on the [Free Music Archive](https://freemusicarchive.org/). 
                            ''')

                    # Reset button
                    st.write("###### If we want to try a different song we can:")
                    st.button("RESET", key = 'reset', type = 'primary')
                    
                    
except (NoBackendError, LibsndfileError):
    st.write('Failed to read in file, please check filetype and try again.')
    input_song = None