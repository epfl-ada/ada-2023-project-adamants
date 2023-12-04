from feature_26_generation import DATA_FOLDER, PATHS_AND_GRAPH, PATHS_FINISHED, PATHS_UNFINISHED, SHORTEST_PATH_MATRIX , ARTICLES, load, load_articles, add_path_length, add_link_position
from feature_13_generation import add_number_of_backtracks, add_number_of_paths_previously_played
from feature_49_generation import add_time_per_edge, split_into_edges, add_sentence_similarity_metric
from feature_78_generation import GRAPH_METRICS_PATH, get_paths_pairs_metrics, compute_metrics_slopes
import pandas as pd
from pandas.errors import SettingWithCopyWarning
import sys
import warnings
sys.path.append("../book")
sys.path.append("book/") # Attend to more cases
from graph_measures import *
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning) # Huge amounts of warnings generated in loops. Ignore them here.

# Setup

print('Setup')
paths_finished = load('paths_finished')
paths_unfinished = load('paths_unfinished')
shortest_path_distance_matrix = load('shortest_path_matrix')
articles = load_articles()

# 6: Position of clicked link in article
print('Feature 6')
paths_finished_copy = paths_finished.copy()
paths_finished = add_link_position(paths_finished_copy, True)[0]
paths_unfinished_copy = paths_unfinished.copy()
paths_unfinished = add_link_position(paths_unfinished_copy, False)[0]

# For compatibility with next functions, this has to be done
paths_finished['path'] = paths_finished['path'].map(lambda x: ';'.join(x))
paths_unfinished['path'] = paths_unfinished['path'].map(lambda x: ';'.join(x))

# 2: Path length and player persistence
print('Feature 2')
paths_finished_copy = paths_finished.copy()
paths_finished = add_path_length(paths_finished_copy, articles, shortest_path_distance_matrix, True)[0]
paths_unfinished_copy = paths_unfinished.copy()
paths_unfinished = add_path_length(paths_unfinished_copy, articles, shortest_path_distance_matrix, False)[0]

# For compatibility with next functions, this has to be done
paths_finished['path'] = paths_finished['path'].map(lambda x: ';'.join(x))
paths_unfinished['path'] = paths_unfinished['path'].map(lambda x: ';'.join(x))

# 4: Time per edge
print('Feature 4')
paths_finished_copy = paths_finished.copy()
paths_finished = add_time_per_edge(paths_finished_copy)
paths_unfinished_copy = paths_unfinished.copy()
paths_unfinished = add_time_per_edge(paths_unfinished_copy)

# Intermediary: split into edges (necessary for later)
finished_edge_df = split_into_edges(paths_finished)
unfinished_edge_df = split_into_edges(paths_unfinished)

# 9: Sentence-transformers-based word-pair simililarity metric
print("Feature 9")
paths_finished_copy, finished_edge_df_copy = paths_finished.copy(), finished_edge_df.copy()
paths_finished, finished_edge_df = add_sentence_similarity_metric(paths_finished_copy, finished_edge_df_copy, finished=True)
paths_unfinished_copy, unfinished_edge_df_copy = paths_unfinished.copy(), unfinished_edge_df.copy()
paths_unfinished, unfinished_edge_df = add_sentence_similarity_metric(paths_unfinished_copy, unfinished_edge_df_copy, finished=False)

# 1: Number of backtracks in path
print('Feature 1')
paths_finished_copy = paths_finished.copy()
paths_finished = add_number_of_backtracks(paths_finished_copy)
paths_unfinished_copy = paths_unfinished.copy()
paths_unfinished = add_number_of_backtracks(paths_unfinished_copy)

# 3: Number of paths previously played per player
print('Feature 3')
paths_finished_copy, paths_unfinished_copy = paths_finished.copy(), paths_unfinished.copy()
paths_finished, paths_unfinished = add_number_of_paths_previously_played(paths_finished_copy, paths_unfinished_copy)

# For compatibility with next functions, this has to be done
paths_finished["path"] = paths_finished["path"].map(lambda x: x.split(";"))
paths_unfinished["path"] = paths_unfinished["path"].map(lambda x: x.split(";"))

# 7 & 8 : Path pair metrics and metrics slopes
print('Metrics 7 & 8')
nodes = pd.read_csv(GRAPH_METRICS_PATH)
paths_unfinished_copy = paths_unfinished.copy()
unfinished_metrics_dict = compute_path_metrics_w_nodes(
    nodes, paths_unfinished_copy
)
unfinished_slopes = compute_metrics_slopes(unfinished_metrics_dict)

paths_finished_copy = paths_finished.copy()
finished_metrics_dict = compute_path_metrics_w_nodes(
    nodes, paths_finished_copy 
)
finished_slopes = compute_metrics_slopes(finished_metrics_dict)


slopes_unfin_df = pd.DataFrame()
for k,v in unfinished_slopes.items():
    slopes_unfin_df = pd.concat([slopes_unfin_df, v], axis=1)
    
paths_unfinished_modif = paths_unfinished.copy()
paths_unfinished_modif = pd.concat([paths_unfinished_modif, slopes_unfin_df], axis=1)

slopes_fin_df = pd.DataFrame()
for k,v in finished_slopes.items():
    slopes_fin_df = pd.concat([slopes_fin_df, v], axis=1)
    
paths_finished_modif = paths_finished.copy()
paths_finished_modif = pd.concat([paths_finished_modif, slopes_fin_df], axis=1)

# Save
paths_unfinished_modif.to_csv(DATA_FOLDER + 'combined_metrics_unfinished_paths.csv')
paths_finished_modif.to_csv(DATA_FOLDER + 'combined_metrics_finished_paths.csv')
unfinished_edge_df.to_csv(DATA_FOLDER + 'combined_metrics_unfinished_edges.csv')
finished_edge_df.to_csv(DATA_FOLDER + 'combined_metrics_finished_edges.csv')
