# Song Classification and Recommendations

## Background

## Outline

## Software Requirements

## Data Processing

## Modeling
Two different approaches to classifying songs into genres were attempted, the first using a [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network) (as documented in [03-CNN](./code/03-CNN.ipynb) and the second using [Principle Component Analysis (PCA)](https://royalsocietypublishing.org/doi/10.1098/rsta.2015.0202) combined with [LightGBM (LGBM)](https://lightgbm.readthedocs.io/en/stable/). Both approaches aimed to accurately classify songs into their corresponding top genre as defined above. 
### CNN
A CNN is a type of feed-forward Neural Network that uses [convolutional layers](https://en.wikipedia.org/wiki/Convolution) in order to identify relevant features in our inputs. In our case, we employed two-dimensional convolutional layers to identify relevant trends in frequency that define our genres.

Our primary metric for evaluating the CNN was [categorical cross-entropy (Log Loss)](https://en.wikipedia.org/wiki/Cross-entropy) in our case this equation is as follows:

$$
\text{Categorical Cross-Entropy} = -\frac{1}{N}\sum_{i = 1}^{N} y_i \sum_{j=1}^{k}y_{ij}\ln{p_{ij}}
\newline
\text{Where N is the number of observations and k is the number of categories.}
$$

In essence, Categorical Cross-Entropy takes the average of how far off our predicted probability of each category is for each song. 

### LightGBM
LightGBM is a gradient-boosting tree-based model that builds a number of successive simple decision trees that are trained on the error of the previous tree. A more thorough overview of gradient boosting and boosting in general can be found [here](https://www.mygreatlearning.com/blog/gradient-boosting/). Our LGBM model was built on the first 100 principle components of our songs. The exact process used here is detailed in the [04-Decomposition_Modeling Notebook](./code/04-Decomposition_Modeling.ipynb). In short, songs were split into 3-second segments and then reduced down using the average and standard deviations of the values at the 128 mel-bands for each segment. Those averages and standard deviations were then reduced further using PCA. 

Our primary metric for this approach was Accuracy where:
$$
\text{Accuracy} = 
$$
## Recommender System

## Conclusion & Future Work

At the moment my first goal is finish this recommender system and get a working applet up and running.

Once I have that, clean up my notebooks and get them presentable then consider going further in depth with any of them.

Once I feel the project is complete I'll reassess whether I have time to get the streamlit app running remotely and do that if I have time.

Otherwise, readme then presentation as always.
