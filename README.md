# Analyzing player strategies in the Wikispeedia game and assessing correlations with semantic content

## Abstract
The Wikispeedia dataset provides a rich set of player navigation paths within Wikipedia’s graph. 
By leveraging those paths, it has been proposed to use these player-created features to derive semantic distance between articles. 
In the following, we aim to validate this approach by finding how player behaviors in-game can be described quantitatively by newly derived features, and assessing how these variables relate to semantic content.
To achieve this, we will :  
- Extract several features that aim to quantify player behavior
- Create a ranking based on these, perform clustering to validate that we achieve a meaningful set of latent variables for player paths 
- Iteratively select and refine features based on their contribution to the clustering
- Compare them to BERT scores of articles pairs to see whether they correlate with semantic content and if better players retain this semantic content in their paths or overfit Wikipedia’s underlying structure instead.

## Research questions
The goal of this project is to unravel and analyze the different strategies used by the players of Wikispeedia by extracting features from player path data. By doing so we want and expect to find strategies that do not only rely on the semantic distances between words but are more tightly dependent on the Wikipedia graph and on the player strategies. Additionally, after having clustered the players and unraveling the strategies explaining these clusters, we want to verify if these strategies have an impact on the mean BERT score of these clusters.

## Methods
To achieve this goal we came up with 10 different metrics that can either tell us about the entire path followed by the player, each choice of pages done by the players or the players themselves. Then we want to run clustering algorithms to discuss the different strategies used by these groups. Here is a list and a description of the different metrics used:  
- The number of backtracks per path: this metric can be used to inform us on how the player navigated its way from the starting page. Was it straightforward or more meandering? As expected it is positively correlated with the rating given by the player and with the time taken to complete the path.  
- The number of paths attempted by the player before the one played: this metric can inform us how much training the player had before he played the path recorded. It is negatively correlated with the time taken to finish the path recorded, even though ~50% of the IP addresses appear only 1 time (heavily right-skewed distribution).  
- Next, we look at the graph properties of the articles within Wikipedia: using well-known measures such as degree, local clustering, and centrality, we can assess the properties of a given article within the graph. We can then relate these to player paths directly and use them as features for the clustering.  
- In addition, in an attempt to quantify the strategy of the users, we look at the degree distribution within a path, take the maximum and fit a line (max. not included) before and after said maximum to one of the graph measures mentioned previously within the path.  
This lets us see how much a player increases or decreases certain measures along the path; it would for example be expected that a good player would have a sharply positive slope in page degree before the maximum (finding a hub from the start) and then a sharply negative slope once they hone into their target.  
- The player performance metric integrates several key factors: the average ratio between the user's path and the shortest path to the goal, the proportion of completed versus incomplete paths, and the average duration spent on a page. Additionally, it considers the average time taken before a player decides to abandon the path. This metric aims to determine which of these elements are most indicative of a player's skill and effectiveness.  
- The position of the clicked link on the article: the position of the next Wikipedia link could allow insight into the answer to know whether players scroll to find a word that popped into their heads, or let their search be guided by the options under your sight, aka the first links visible on the article. We could then characterize the players’ rationality or influenceability.
- The path length: this metric would allow us to know whether a longer optimal path distance between the start article and target article makes a game more difficult (leading the players to rate the game as difficult, or leading them to restart or even give up!).  
- Path length and player persistence: Investigate if longer or more difficult paths prompt players to abandon the game.  
- Distribution of average time per edge: Explore the distribution of average time spent per edge during navigation. Seeing if this data has particular outliers may give indicators of any trouble they might have had.  
- BERTscore semantic similarity metric: Apply the BERTscore semantic similarity metric to evaluate path difficulties, and compare this to player ratings. What sort of link is there between the two? Comparison of per-edge and per-path values may give different insights (although per path is more relevant). These metrics will be selected and improved based on their contribution to the clustering.  

## Proposed timeline

|         Title        | Description                                                                                                                                                                | Estimated time | Start date | Due date |
|:--------------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------:|:----------:|:--------:|
|                      |                                                                                                                                                                            |                |            |          |
|    **Milestone 1**   | Data formatting                                                                                                                                                            |     6 days     |   18 nov   |  30 nov  |
|       Filtering      | Choose as a group which data we will exclude  from the analysis using the previous  analysis of the metrics                                                                |      1 day     |   18 nov   |  22 nov  |
|      Dataframes      | Create 2 dataframes, containing either 1 row per path, 1 column per metric or 1 row per  connection, 1 column per metric, choosing the metrics that make sense.            |     5 days     |   22 nov   |  30 nov  |
|                      |                                                                                                                                                                            |                |            |          |
|    **Milestone 2**   | Computation and visualization                                                                                                                                              |     7 days     |    1 dec   |   8 dec  |
|   Plots generation   | Complete the notebook(s) with plots showing  the clusters, the correlations between  clusters, the mean BERTscore; for multiple  clustering algorithm and the different df |     7 days     |    1 dec   |   8 dec  |
|                      |                                                                                                                                                                            |                |            |          |
|    **Milestone 3**   | Interpretation and visualization                                                                                                                                           |     12 days    |    8 dec   |  20 dec  |
| Story-writing medium | Choose as a group how we want to write the data story                                                                                                                      |     2 days     |    8 dec   |  10 dec  |
| Write the data story | Write the data story using the notebook, extend it if more plots are needed                                                                                                |     10 days    |   10 dec   |  20 dec  |
|                      |                                                                                                                                                                            |                |            |          |
|    **Milestone 4**   | Final check                                                                                                                                                                |      1 day     |   20 dec   |  22 dec  |
|   Data story check   | Check for spelling mistakes, formatting,...                                                                                                                                |      1 day     |   20 dec   |  22 dec  |
| Final notebook check | Check for spelling mistakes, formatting,...                                                                                                                                |      1 day     |   20 dec   |  22 dec  |

## Organization within the team
See the milestones above
