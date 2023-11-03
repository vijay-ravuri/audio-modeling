import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity


# Splits a full song into even segments, by default 10 segments of 3s
def audio_framer(song, idx, splits = 10):
    # Creating columns for returned dataframe
    index_list = [idx] * splits # Maintains index of original song
    timestamps = list(range(splits)) # Orders each segment for potential future use
    df  = pd.DataFrame({'song_index' : index_list, 'ts' : timestamps}) 
    

    # Each slice is aggregated then added into a list to create a dataframe
    slices = []
    for slice in np.array_split(song, splits, axis = 0):
        sliced = pd.concat(
            [
                pd.Series(
                    np.mean(slice, axis = 0)
                ),
                pd.Series(
                    np.std(slice, axis = 0)
                )
            ],
            axis = 0
        ).reset_index(drop = True)
        
        slices.append(sliced)

    # Returns a dataframe of size splits x 258
    return pd.concat([df, pd.DataFrame(slices)], axis = 1)


# Returns top 5 songs by segment cosine similarity
# For a step by step explanation see 05-Recommender_System
def get_recommendations(song_frame, df):
    similarities = pd.DataFrame(cosine_similarity(song_frame, df[[str(x) for x in range(100)]]))

    to_stack = []
    for idx in similarities.index:
        # Find top 5 similar for the row
        top5 = similarities.loc[idx, :].sort_values().tail()
        topper5 = pd.DataFrame(zip(top5.index, top5.values), columns = ['segment', 'cos'])
        
        # Slightly de-weighting the first ~9s of each song
        topper5['cos'] = topper5['cos'].apply(lambda x: 0.8 * x if idx < 3 else x)

        to_stack.append(topper5)

    stacked_sim = pd.concat(to_stack).reset_index(drop = True)

    # Merge the song indices for the most similar segments
    similarities_with_index = pd.merge(
        stacked_sim,
        df[['song_index']],
        how = 'inner',
        left_on = 'segment',
        right_index = True
    )

    # Sum similarities for each song with a segment in our top 5 similarities
    most_similar = similarities_with_index.groupby(
        'song_index')['cos'].sum().sort_values(
            ascending = False
            ).head()


    # Gather song info for most similar songs
    recommendations = pd.merge(
        df[(df['song_index'].isin(most_similar.index)) & (df['ts'] == 0)][ #Selecting just one timestamp for the merge
            ['song_index', 'title', 'name', 'album', 'genre_top']
            ],
        most_similar,
        how = 'inner',
        left_on = 'song_index',
        right_index = True
    ).sort_values(by = 'cos', ascending = False)

    return recommendations


# Plots a decomposed song in the first two principal components
def plot_decomp(decomposed, component_1 = 0, component_2 = 1):
    fig, ax = plt.subplots()
    cm = plt.cm.ScalarMappable(cmap = 'winter', 
                               norm  = plt.Normalize(
                                   0, 
                                   decomposed.shape[0]*3
                                )
                            )

    plot = sns.scatterplot(
        x = decomposed[component_1],
        y = decomposed[component_2],
        hue = decomposed.index * 3,
        palette = 'winter',
        ax = ax
    )

    ax.legend().remove()

    plt.colorbar(cm, ax = plot, label = 'Time (s)')

    labels = ['First', 'Second', 'Third', 'Fourth', 'Fifth']
    ax.set_xlabel('{} Principal Component'.format(labels[component_1]))
    ax.set_ylabel('{} Principal Component'.format(labels[component_2]))

    return fig, ax