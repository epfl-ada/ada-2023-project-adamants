from feature_26_generation import DATA_FOLDER, PATHS_AND_GRAPH, PATHS_FINISHED, PATHS_UNFINISHED, SHORTEST_PATH_MATRIX , ARTICLES, load, load_articles, add_path_length, add_link_position
from feature_13_generation import add_number_of_backtracks, add_number_of_paths_previously_played
from feature_49_generation import add_time_per_edge, split_into_edges, add_sentence_similarity_metric
from feature_78_generation import GRAPH_METRICS_PATH, get_paths_pairs_metrics, compute_metrics_slopes, add_computed_graph_features
from feature_5_generation import  add_average_time_on_page, add_paths_ratio
import pandas as pd
from pandas.errors import SettingWithCopyWarning
import sys
import warnings
sys.path.append("../book")
sys.path.append("book/") # Attend to more cases
from graph_measures import *
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning) # Huge amounts of warnings generated in loops. Ignore them here.
import numpy as np

# Setup

print('Setup')
paths_finished_raw = load('paths_finished')
paths_unfinished_raw = load('paths_unfinished', drop_timeouts=True)
print(f"Shape of paths_finished_raw: {paths_finished_raw.shape}")
print(f"Shape of paths_unfinished_raw: {paths_unfinished_raw.shape}")
paths_finished = paths_finished_raw.copy()
paths_unfinished = paths_unfinished_raw.copy()
shortest_path_distance_matrix = load('shortest_path_matrix')
articles = load_articles()


# 1: Number of backtracks in path
print('Feature 1')
paths_finished_copy = paths_finished.copy()
paths_finished = add_number_of_backtracks(paths_finished_copy)
paths_unfinished_copy = paths_unfinished.copy()
paths_unfinished = add_number_of_backtracks(paths_unfinished_copy)

print(f"Shape of paths_finished: {paths_finished.shape} after adding feature 1")
print(f"Shape of paths_unfinished: {paths_unfinished.shape} after adding feature 1")

# # 3: Number of paths previously played per player
print('Feature 3')
paths_finished_copy, paths_unfinished_copy = paths_finished.copy(), paths_unfinished.copy()
paths_finished, paths_unfinished = add_number_of_paths_previously_played(paths_finished_copy, paths_unfinished_copy)

print(f"Shape of paths_finished: {paths_finished.shape} after adding feature 3")
print(f"Shape of paths_unfinished: {paths_unfinished.shape} after adding feature 3")

# # 4: Time per edge
print('Feature 4')
paths_finished_copy = paths_finished.copy()
paths_finished = add_time_per_edge(paths_finished_copy)
paths_unfinished_copy = paths_unfinished.copy()
paths_unfinished = add_time_per_edge(paths_unfinished_copy)

print(f"Shape of paths_finished: {paths_finished.shape} after adding feature 4")
print(f"Shape of paths_unfinished: {paths_unfinished.shape} after adding feature 4")

# 6: Position of clicked link in article
# print('Feature 6')
paths_finished_copy = paths_finished.copy()
paths_finished = add_link_position(paths_finished_copy, True)
paths_unfinished_copy = paths_unfinished.copy()
paths_unfinished = add_link_position(paths_unfinished_copy, False)

print(f"Shape of paths_finished: {paths_finished.shape} after adding feature 6")
print(f"Shape of paths_unfinished: {paths_unfinished.shape} after adding feature 6")

# # For compatibility with next functions, this has to be done
paths_finished['path'] = paths_finished['path'].map(lambda x: ';'.join(x))
paths_unfinished['path'] = paths_unfinished['path'].map(lambda x: ';'.join(x))

# 2: Path length and player persistence
print('Feature 2')
paths_finished_copy = paths_finished.copy()
paths_finished = add_path_length(paths_finished_copy, articles, shortest_path_distance_matrix, True)[0]
paths_unfinished_copy = paths_unfinished.copy()
paths_unfinished = add_path_length(paths_unfinished_copy, articles, shortest_path_distance_matrix, False)[0]

print(f"Shape of paths_finished: {paths_finished.shape} after adding feature 2")
print(f"Shape of paths_unfinished: {paths_unfinished.shape} after adding feature 2")

# For compatibility with next functions, this has to be done
paths_finished['path'] = paths_finished['path'].map(lambda x: ';'.join(x))
paths_unfinished['path'] = paths_unfinished['path'].map(lambda x: ';'.join(x))

# Intermediary: split into edges (necessary for later)
finished_edge_df = split_into_edges(paths_finished)
unfinished_edge_df = split_into_edges(paths_unfinished)

# 5: Player performance
print('Feature 5')
paths_finished_copy = paths_finished.copy()
paths_finished = add_paths_ratio(paths_finished_copy,shortest_path_distance_matrix,articles)
paths_finished_copy = paths_finished.copy()
paths_finished = add_average_time_on_page(paths_finished_copy)
paths_unfinished_copy = paths_unfinished.copy()
paths_unfinished = add_paths_ratio(paths_unfinished_copy,shortest_path_distance_matrix,articles,finished=False)
paths_unfinished_copy = paths_unfinished.copy()
paths_unfinished = add_average_time_on_page(paths_unfinished_copy)


print(f"Shape of paths_finished: {paths_finished.shape} after adding feature 5")
print(f"Shape of paths_unfinished: {paths_unfinished.shape} after adding feature 5")

# For compatibility with next functions, this has to be done
paths_finished['path'] = paths_finished['path'].map(lambda x: ';'.join(x))
paths_unfinished['path'] = paths_unfinished['path'].map(lambda x: ';'.join(x))

# 9: Sentence-transformers-based word-pair simililarity metric
print("Feature 9")
paths_finished_copy, finished_edge_df_copy = paths_finished.copy(), finished_edge_df.copy()
paths_finished, finished_edge_df = add_sentence_similarity_metric(paths_finished_copy, finished_edge_df_copy, finished=True)
paths_unfinished_copy, unfinished_edge_df_copy = paths_unfinished.copy(), unfinished_edge_df.copy()
paths_unfinished, unfinished_edge_df = add_sentence_similarity_metric(paths_unfinished_copy, unfinished_edge_df_copy, finished=False)

print

# For compatibility with next functions, this has to be done
paths_finished["path"] = paths_finished["path"].map(lambda x: x.split(";"))
paths_unfinished["path"] = paths_unfinished["path"].map(lambda x: x.split(";"))

# # 7 & 8 : Path pair metrics and metrics slopes
print('Features 7 & 8')
paths_finished_copy, paths_unfinished_copy = paths_finished.copy(), paths_unfinished.copy()
paths_finished_modif = add_computed_graph_features(
    paths_finished_raw,
    paths_finished_copy, 
    metrics_pickle_path="../data/p3_extra_data/finished_path_metrics.pkl"
)
paths_unfinished_modif = add_computed_graph_features(
    paths_unfinished_raw, 
    paths_unfinished_copy, 
    metrics_pickle_path="../data/p3_extra_data/unfinished_path_metrics.pkl"
)

# Save
paths_unfinished_modif.to_csv(DATA_FOLDER + 'combined_metrics_unfinished_paths.csv')
paths_finished_modif.to_csv(DATA_FOLDER + 'combined_metrics_finished_paths.csv')
unfinished_edge_df.to_csv(DATA_FOLDER + 'combined_metrics_unfinished_edges.csv')
finished_edge_df.to_csv(DATA_FOLDER + 'combined_metrics_finished_edges.csv')
