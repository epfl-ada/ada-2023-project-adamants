import sys
import pandas as pd
from pathlib import Path
import numpy as np
sys.path.append("../book")
from graph_measures import *
from utils import load_data, load_paths_pairs, load_paths
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

GRAPH_METRICS_PATH = Path("../data/p3_extra_data/nodes_w_graph_metrics.csv").resolve()

def append_nan_to_dict(d):
    """Adds nan values to a dict for the given keys."""
    for k, v in d.items():
        d[k].append(np.nan)

def get_paths_pairs_metrics(paths_pairs, nodes_path=GRAPH_METRICS_PATH):
    """Adds the graph metrics to path pairs as extra columns.
    
    Makes a deep copy of the data, the modified copy is returned."""
    paths_pairs_metrics = paths_pairs.copy(deep=True)
    print("Computing graph metrics...")
    if Path(nodes_path).is_file():
        nodes_w_graph_metrics = pd.read_csv(GRAPH_METRICS_PATH)
        print("Succesfully loaded")
    else:
        _, links, _ = load_data(unquote_names=False)
        nodes_w_graph_metrics = compute_node_metrics(links, save_csv=True)
    
    page_degree_dict= dict(zip(nodes_w_graph_metrics["node_name"], nodes_w_graph_metrics["degree"]))
    page_clustering_dict = dict(zip(nodes_w_graph_metrics["node_name"], nodes_w_graph_metrics["clustering"]))
    page_degree_centrality_dict= dict(
        zip(nodes_w_graph_metrics["node_name"], nodes_w_graph_metrics["degree_centrality"])
    )
    page_betweenness_dict = dict(zip(nodes_w_graph_metrics["node_name"], nodes_w_graph_metrics["betweenness"]))
    page_closeness_dict = dict(zip(nodes_w_graph_metrics["node_name"], nodes_w_graph_metrics["closeness"]))
    
    
    mapping = {
        "degree": page_degree_dict,
        "clustering": page_clustering_dict,
        "degree_centrality": page_degree_centrality_dict,
        "betweenness": page_betweenness_dict,
        "closeness": page_closeness_dict,
    }
    
    metrics_dict_from = {
        "degree_from": [],
        "clustering_from": [],
        "closeness_from": [],
        "betweenness_from": [],
        "degree_centrality_from": [],
    }
    metrics_dict_to = {
        "degree_to": [],
        "clustering_to": [],
        "closeness_to": [],
        "betweenness_to": [],
        "degree_centrality_to": [],
    }
    print("_"*20)
    print("Adding graph metrics to path pairs...")
    total = len(paths_pairs_metrics)
    current = 0
    for row in paths_pairs_metrics.itertuples():
        page_from = str(row[1])
        page_to = str(row.to)
        
        if page_from != "<":
            for k,v in mapping.items():
                try:
                    k += "_from"
                    metrics_dict_from[k].append(v[page_from])
                except KeyError:
                    print(f"Key error: {page_from} not found in nodes in metric {k}")
                    metrics_dict_from[k].append(np.nan)
        else:
            append_nan_to_dict(metrics_dict_from)
        
        if page_to != "<":
            for k,v in mapping.items():
                try:
                    k += "_to"
                    metrics_dict_to[k].append(v[page_to])
                except KeyError:
                    print(f"Key error: {page_to} not found in nodes in metric {k}")
                    metrics_dict_to[k].append(np.nan)
        else:
            append_nan_to_dict(metrics_dict_to)
        
        current += 1
        print(f"{current}/{total}", end="\r")
    print("_"*20)
    print("Adding columns to dataframe...")
    for k, v in metrics_dict_from.items():
        paths_pairs_metrics[k] = v
    for k, v in metrics_dict_to.items():
        paths_pairs_metrics[k] = v
        
    return paths_pairs_metrics


if __name__ == "__main__":
    
    finished_metrics_pairs_csv_path = Path("../data/p3_extra_data/finished_paths_pairs_metrics.csv").resolve()
    unfinished_metrics_pairs_csv_path = Path("../data/p3_extra_data/unfinished_paths_pairs_metrics.csv").resolve()
    
    finished_paths, unfinished_paths = load_paths_pairs()
    if not finished_metrics_pairs_csv_path.is_file():
        print("Computing finished paths pairs metrics...")
        finished_paths_metrics = get_paths_pairs_metrics(finished_paths)
        finished_paths_metrics.to_csv(finished_metrics_pairs_csv_path)
    if not unfinished_metrics_pairs_csv_path.is_file():
        print("Computing unfinished paths pairs metrics...")
        unfinished_paths_metrics = get_paths_pairs_metrics(unfinished_paths)
        unfinished_paths_metrics.to_csv(unfinished_metrics_pairs_csv_path)

    finished_paths_appended_metrics_path = Path("../data/p3_extra_data/finished_paths_appended_metrics.csv").resolve()
    unfinished_paths_appended_metrics_path = Path("../data/p3_extra_data/unfinished_paths_appended_metrics.csv").resolve()
    
    finished_paths, unfinished_paths = load_paths(unquote_names=False, drop_timeouts=True)
    nodes = pd.read_csv(GRAPH_METRICS_PATH)
    if not finished_paths_appended_metrics_path.is_file():
        print("Computing finished paths metrics...")
        finished_paths_appended_metrics = append_features_to_paths(finished_paths, nodes)
        finished_paths_appended_metrics.to_csv(finished_paths_appended_metrics_path)
    if not unfinished_paths_appended_metrics_path.is_file():
        print("Computing unfinished paths metrics...")
        unfinished_paths = unfinished_paths.copy(deep=True)
        unfinished_paths_appended_metrics = append_features_to_paths(unfinished_paths, nodes)
        unfinished_paths_appended_metrics.to_csv(unfinished_paths_appended_metrics_path)
    print("Done")
    