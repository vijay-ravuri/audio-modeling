# Song Classification and Recommendations
The web app where you can find recommendations for a song of your choosing can be found [here](https://audio-modeling.streamlit.app/). 

Slides for the presentation can be found [here](./presentation.pdf).
## Background

This project uses song and song metadata gathered from the [Free Music Archive](https://freemusicarchive.org/) by a group of researchers at École Polytechnique Fédérale de Lausanne (EPFL). The full repo with their data and information about the source of the data can be found [here](https://github.com/mdeff/fma). Specifically, we used the FMA-Small subset of data which is a set of 8000 30-second clips from songs with an even genre distribution. Genre here refers to the "top genre" of each song as determined by the researchers who created this dataset. Songs of course can be any number of genres so there is naturally a lot of overlap in songs between genres. 

Two goals were set for this project. Firstly, build a model to classify songs into genres based on the raw audio data in some way. Secondly, build a recommender system to find similar songs within our data to a song of a users choosing. 

This repository has no audio data included. Please see the FMA repository for the dataset used.

## Outline
1. [Processing](./code/01-Processing.ipynb) - Reading in and transforming the songs and track data.
2. [Exploratory Data Analysis](./code/02-EDA.ipynb) - Visual representations of our song data
3. [Convolutional Neural Network](./code/03-CNN.ipynb) - An attempt at applying deep-learning to classify songs into genres
4. [Decomposition Modeling](./code/04-Decomposition_Modeling.ipynb) - Using PCA on time-aggregated segments then applying LightGBM to classify songs into genres
5. [Recommender System](./code/05-Recommender_System.ipynb) - Details the process of generating recommendations for a user-input song as used in the streamlit app.

## Software Requirements
This project was done using Python 3.10.2, the versions of relevant libraries are listed in [requirements-complete.txt](./requirements-complete.txt). The [requirements.txt](./requirements.txt) file is slightly different, omitting tensorflow as it is not necessary for the streamlit app to run. There may be a slight dependency issue downloading the exact version of tensorflow but there should be no changes to the analysis or process if run using a newer version of the library.
## Data Processing
The [Librosa](https://librosa.org/doc/latest/index.html) library allows for processing raw audio data into other forms using various transformations. For our purposes we converted our raw audio for each file into a [mel-spectograms](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0). This allows us to view our songs in 3 key dimension: time, frequency (Hz), and loudness (dB). The process of transforming the songs is done through what [Fourier Transforms](https://en.wikipedia.org/wiki/Fourier_transform), more specifically a number of consecutive [Fast Fourier transforms (FFT)](https://en.wikipedia.org/wiki/Fast_Fourier_transform).
## Modeling
Two different approaches to classifying songs into genres were attempted, the first using a [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network) (as documented in [03-CNN](./code/03-CNN.ipynb) and the second using [Principle Component Analysis (PCA)](https://royalsocietypublishing.org/doi/10.1098/rsta.2015.0202) combined with [LightGBM (LGBM)](https://lightgbm.readthedocs.io/en/stable/). Both approaches aimed to accurately classify songs into their corresponding top genre as defined above. 
### CNN
A CNN is a type of feed-forward Neural Network that uses [convolutional layers](https://en.wikipedia.org/wiki/Convolution) in order to identify relevant features in our inputs. In our case, we employed two-dimensional convolutional layers to identify relevant trends in frequency that define our genres.

Our primary metric for evaluating the CNN was [categorical cross-entropy (Log Loss)](https://en.wikipedia.org/wiki/Cross-entropy) in our case this equation is as follows:

$$
\displaylines{
    \text{Categorical Cross-Entropy} = -\frac{1}{N}\sum_{i = 1}^{N} y_i \sum_{j=1}^{k}y_{ij}\ln{p_{ij}}
\newline
\text{Where N is the number of observations and k is the number of categories.}
}
$$

In essence, Categorical Cross-Entropy takes the average of how far off our predicted probability of each category is for each song. 

### LightGBM
LightGBM is a gradient-boosting tree-based model that builds a number of successive simple decision trees that are trained on the error of the previous tree. A more thorough overview of gradient boosting and boosting in general can be found [here](https://www.mygreatlearning.com/blog/gradient-boosting/). Our LGBM model was built on the first 100 principle components of our songs. The exact process used here is detailed in the [04-Decomposition_Modeling Notebook](./code/04-Decomposition_Modeling.ipynb). In short, songs were split into 3-second segments and then reduced down using the average and standard deviations of the values at the 128 mel-bands for each segment. Those averages and standard deviations were then reduced further using PCA. 

Our primary metric for this approach was Accuracy where:
```math
\text{Accuracy} = \frac{\# \text{ of correct predictions}}{N}
```
Additionally, [04](./code/04-Decomposition_Modeling.ipynb) does briefly assess our models performance with various other metrics such as log-loss, [precision, recall](https://en.wikipedia.org/wiki/Precision_and_recall), and [F-1 Score](https://en.wikipedia.org/wiki/F-score). 
## Recommender System
Our recommender system works on the same aggregated 3-second segments of songs as the LightGBM model. New inputs are broken into 3-second segments, average and standard deviations for each band are calculated and then decomposed, and those principal components are used to find similar segments in our dataset using [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity). The top 10 most similar segments in our data are gathered for each segment in the user-input song. Then, those similarities are grouped by whichever song they originally came from and summed together. The recommendations are then the top 5 songs with highest summed cosine similarities. The exact process of taking in a new song and finding similar songs is detailed in [05](./code/05-Recommender_System.ipynb) and the function that is used for the applet can be found in [song_utils.py](./code/song_utils.py).

The cosine similarity between two vectors $A$ and $B$ is defined as follows:


```math
cos(\theta) = \frac{A \cdot B}{\lVert A\rVert \lVert B \rVert}
```



This means that song segments with similar values in each principal components will have high scores and segments with very different values will have low scores.
## Conclusion & Future Work
Overall, our attempts at modeling struggled quite a bit at accurately classifying songs into genres. In particular, our attempt at a CNN had very poor performance and generally can be considered a failed attempt. The LightGBM approach showed a bit more promise, reaching an accuracy of 42.23% on our test set. Our model performed best at classifying Hip-Hop songs with a respectable 57% accuracy. We observed high intra-genre variance along with relatively low inter-genre variance which explains why classification proved to be such a difficult task. Likely the difficulties our models faced are due to the "top genre" being one of a number of genres that can describe a song. For example, our LGBM model struggled the most to classify Pop songs correct (only 20% accuracy) which makes sense given that Pop is a much looser definition that often takes strong inspiration from other genres. Our recommender system is a bit simplistic but is functional and can be scaled up to a much more complex implementation. A potential avenue for future work could look into finding a better balance between the number of times a song comes up as similar and how similar the segments individually are. Another avenue could be to explore which genres are more self-similar and measure self-similarity for individual songs. From our analysis it is clear that electronic songs are often much more spread out while more acoustic songs tended to be clumped a little bit more. Lastly, future work with regards to the CNN approach would be prudent. A deep-dive into the failings of our approach is included at the end of [03](./code/03-CNN.ipynb) and a better implementation would likely yield results that match or exceed our decomposition approach. 

On a personal note, this project was my first attempt working with audio data of any kind and if I were to work on data of this nature (whether a second attempt with this data or another project) I would want to focus a lot more on different approaches to processing the audio data. With little familiarity I fell back on what seemed to be the tried and true approaches but there are a number of other ways to approach modeling audio data including applying PCA directly onto the songs and processing the songs in their raw audio state (as seen in [Wave2Vec](https://ai.meta.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/)).

## References
All song data for modeling and analysis purposes was sourced from [this](https://github.com/mdeff/fma) repo which is released under [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). I hold no copyright over the audio used in this project.
