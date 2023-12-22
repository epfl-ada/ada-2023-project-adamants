# Extra text for data story

## Validation

In order to check the Leiden clustering's robustness, we used 5-fold cross-validation.
Due to the clusters not having a fixed ID across runs, we performed matching using the clustroids nearest to those in a given fold across other folds.
Once this is done, we can plot the original clustroids and their nearest neighbors across folds, and see how close they are in the UMAP embedding space.

From the following plot, we see they are quite close to each other !