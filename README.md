# Song Classification and Recommendations

## Background

This project uses song and song metadata gathered from the [Free Music Archive](https://freemusicarchive.org/) by a group of Swiss researchers at École Polytechnique Fédérale de Lausanne (EPFL). The full repo with their data and information about the source of the data can be found [here](https://github.com/mdeff/fma). Specifically, we used the FMA-Small subset of data which is a set of 8000 30-second clips from songs with an even genre distribution. Genre here refers to the "top genre" of each song as determined by the researchers who created this dataset. Songs of course can be any number of genres so there is naturally a lot of overlap in songs between genres.

Two goals were set for this project. Firstly, build a model to classify songs into genres based on the raw audio data in some way. Secondly, build a recommender system to find similar songs within our data to a song of a users choosing. 


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
$$
\text{Accuracy} = \frac{\# \text{ of correct predictions}}{N}
$$
Additionally, [04](./code/04-Decomposition_Modeling.ipynb) does briefly assess our models performance with various other metrics such as log-loss, [precision, recall](https://en.wikipedia.org/wiki/Precision_and_recall), and [F-1 Score](https://en.wikipedia.org/wiki/F-score). 
## Recommender System
Our recommender system works on the same aggregated 3-second segments of songs as the LightGBM model. New inputs are broken into 3-second segments, average and standard deviations for each band are calculated and then decomposed, and those principal components are used to find similar segments in our dataset using [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity). The top 10 most similar segments in our data are gathered for each segment in the user-input song. Then, those similarities are grouped by whichever song they originally came from and summed together. The recommendations are then the top 5 songs with highest summed cosine similarities. The exact process of taking in a new song and finding similar songs is detailed in [05](./code/05-Recommender_System.ipynb) and the function that is used for the applet can be found in [song_utils.py](./code/song_utils.py).

The cosine similarity between two vectors $A$ and $B$ is defined as follows:
$$
cos(\theta) = \frac{A \cdot B}{\lVert A\rVert \lVert B \rVert}
$$
This means that song segments with similar values in each principal components will have high scores and segments with very different values will have low scores.
## Conclusion & Future Work

At the moment my first goal is finish this recommender system and get a working applet up and running.

Once I have that, clean up my notebooks and get them presentable then consider going further in depth with any of them.

Once I feel the project is complete I'll reassess whether I have time to get the streamlit app running remotely and do that if I have time.

Otherwise, readme then presentation as always.
