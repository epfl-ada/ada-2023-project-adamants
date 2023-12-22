# Introduction

<h2> Analyzing player strategies in the Wikispeedia game </h2>

## Deliverables

In this project milestone, we have completed an extended exploration and visualisation of our data based on the principles discussed in milestone 2.

- From the features we previous gathered for milestone 2, we have tried multiple clustering methods, which we have visualised, compared, and then validated within [this Jupyter book](https://c-achard.github.io/ada-2023-project-adamants/README_p3.html) (which you might already be reading). All of the code used in order to perform our analysis, and its associated outputs / results can also be found within.

- We have created an interactive, story-based exploration of our findings in a [website-based](https://epfl-ada.github.io/ada-2023-project-adamants/) format. It summarises the essential components of our exploration and findings, in a fun and interesting way.

- In case you are looking for a way back, our repository is [located here](https://github.com/epfl-ada/ada-2023-project-adamants/tree/main).

---

## Summary of previous goals (Milestone P2)

The Wikispeedia dataset provides a rich set of player navigation paths within Wikipedia’s graph.
By leveraging those paths, it has been proposed to use these player-created features to derive semantic distance between articles.

In the following, we aim to validate this approach by finding how player behaviors in-game can be described quantitatively by newly derived features, and assessing how these variables relate to semantic content.
To achieve this, we did the following:

- Extracting several features that aim to quantify player behavior from the Wikispeedia dataset and applying meaningful transformations to them.
- Testing different clustering algorithms and comparing them to assess their meaningfulness in terms of player strategies
- Iteratively selecting and refining features based on their contribution to the clustering
- Assessing the robustness of our chosen clustering algorithm by comparing the results obtained on different subsets of the data

## Features used for clustering

To achieve this goal we came up with 10 different features that can either tell us about the entire path followed by the player, the path  of pages followed by the players or the players themselves. Then we will run clustering algorithms to discuss the different strategies used by these groups. Here is a list and a description of the different features extracted:  

|Feature Name | Description|
|-------------|------------|
|durationInSec|Time taken by the player to complete the path (directly provided in the dataset).|
|backtrack|Number of backtracks made by the player in this path.|
|numberOfPath|Number of paths previously played by the player, in total.|
|position_mean|Character position of succeding clicked link on article page, averaged across all edges of path.|
|position_std|Standard deviation of character position of succeding clicked link within a path.|
|path_length|Simple measure of number of edges in path (pages clicked on) from beginning to end. Does not contains player backtracks, and so counts the final version of player path.|
|coarse_mean_time|Measure of amount of time, on average, that the player took to travel from one edge to another in the path. It is called "coarse" because it is a global measure, that does not take into account outlier edges like the median could. This is especially noticeable on paths where the player disconnected, for example.|
|semantic_similarity|This is a measure of the average similarity between two successive terms in the given path. The feature is calculated by computing the cosine similarity (normalised dot product) between embeddings (one-hot vectors projected to a lower dimensionality subspace) given by a sentence transformer, which is a model specifically trained in order to give embeddings which have "semantically relevant" / "semantically informative" relative positions. As such, the cosine similarity gives a measure of logical similarity of the words. A value of -1 represents opposite meanings / topics, around 0 means unrelated, and around 1 means highly similar meanings / topics. The mean of this is supposed to approximately capture how related topics globally were on a given path.|
|ratio|The ratio between the user path and the shortest path between the start and end page.|
|path_closeness_abs_sum|As each page in the path has a closeness centrality associated to it from the full network of Wikipedia, this features is obtained by looking at the derivative of the closeness centrality of each page in the path (that is, the delta between the closeness centrality of a page and the previous one in the path) and summing the absolute value of these deltas before and after the maximal degree page in the path. <br />This is meant to quantify if the player is strongly looking for a hub page in the path, reaches it (as the maximal degree page is assumed to be a hub page) and then goes to a specific page from there, i.e. the target page. Indeed, a good player is expected to have a strongly positive delta before the maximal degree page and a strongly negative delta after it. We take the absolute value then sum them to have a single feature that aims at quantifying this behaviour.|

---

## Clustering code and figures

You will find our code in the following sections:

- {ref}`section:data-loading` : Data loading and pre-processing (features are already computed, we simply analyze and apply transforms to them)
- {ref}`section:leiden-clustering` : Leiden clustering (without timeouts, see before last section for details)
- {ref}`section:clustering-validation` : Assessement of the clustering robustness
- {ref}`section:clustering-timeouts` : Leiden clustering with timeouts
- {ref}`section:hierarchical-clustering` : Hierarchical clustering and other alternatives that were tested

## Group contributions

- Yves: backtrack, numberOfPath, clustering and stastical analysis
- Mathilde: position, path length, site presentation and redaction
- Cyril: Graph measures features, initial data preprocessing, clustering validation, Jupyter book
- David: ratio, site presentation and coding
- Laurent: semantic similarity, coarse mean and clustering