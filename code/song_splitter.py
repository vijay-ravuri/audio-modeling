import pandas as pd
import numpy as np


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