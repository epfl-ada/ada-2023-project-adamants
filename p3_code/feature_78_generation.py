import sys
import pandas as pd
from pathlib import Path
import numpy as np
sys.path.append("../book")
from graph_measures import *
from utils import load_data, load_paths_pairs

GRAPH_METRICS_PATH = Path("../data/p3_extra_data/links_w_graph_metrics.csv").resolve()

def append_nan_to_dict(d):
    """Adds nan values to a dict for the given keys."""
    for k, v in d.items():
        d[k].append(np.nan)

def get_paths_pairs_metrics(paths_pairs, links_path=GRAPH_METRICS_PATH):
    """Adds the graph metrics to path pairs as extra columns.
    
    Makes a deep copy of the data, the modified copy is returned."""
    paths_pairs_metrics = paths_pairs.copy(deep=True)
    print("Computing graph metrics...")
    if Path(links_path).is_file():
        links_w_graph_metrics = pd.read_csv(GRAPH_METRICS_PATH)
        print("Succesfully loaded")
    else:
        _, links, _ = load_data(unquote_names=False)
        links_w_graph_metrics = compute_graph_metrics(links, save_csv=True)
    
    page_degree_dict_from = dict(zip(links_w_graph_metrics["from"], links_w_graph_metrics["degree"]))
    page_clustering_dict_from = dict(zip(links_w_graph_metrics["from"], links_w_graph_metrics["clustering"]))
    page_degree_centrality_dict_from = dict(
        zip(links_w_graph_metrics["from"], links_w_graph_metrics["degree_centrality"])
    )
    page_betweenness_dict_from = dict(zip(links_w_graph_metrics["from"], links_w_graph_metrics["betweenness"]))
    page_closeness_dict_from = dict(zip(links_w_graph_metrics["from"], links_w_graph_metrics["closeness"]))
    
    page_degree_dict_to = dict(zip(links_w_graph_metrics["to"], links_w_graph_metrics["degree"]))
    page_clustering_dict_to = dict(zip(links_w_graph_metrics["to"], links_w_graph_metrics["clustering"]))
    page_degree_centrality_dict_to = dict(
        zip(links_w_graph_metrics["to"], links_w_graph_metrics["degree_centrality"])
    )
    page_betweenness_dict_to = dict(zip(links_w_graph_metrics["to"], links_w_graph_metrics["betweenness"]))
    page_closeness_dict_to = dict(zip(links_w_graph_metrics["to"], links_w_graph_metrics["closeness"]))
        
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
        page_from = row[1]
        page_to = row.to
        
        if page_from != "<":
            try:
                metrics_dict_from["degree_from"].append(page_degree_dict_from[page_from])
                metrics_dict_from["clustering_from"].append(page_clustering_dict_from[page_from])
                metrics_dict_from["closeness_from"].append(page_closeness_dict_from[page_from])
                metrics_dict_from["betweenness_from"].append(page_betweenness_dict_from[page_from])
                metrics_dict_from["degree_centrality_from"].append(page_degree_centrality_dict_from[page_from])
            except KeyError:
                print(f"Key error: {page_from}")
                append_nan_to_dict(metrics_dict_from)
        else:
            append_nan_to_dict(metrics_dict_from)
        
        if page_to != "<":
            try:
                metrics_dict_to["degree_to"].append(page_degree_dict_to[page_to])
                metrics_dict_to["clustering_to"].append(page_clustering_dict_to[page_to])
                metrics_dict_to["closeness_to"].append(page_closeness_dict_to[page_to])
                metrics_dict_to["betweenness_to"].append(page_betweenness_dict_to[page_to])
                metrics_dict_to["degree_centrality_to"].append(page_degree_centrality_dict_to[page_to])
            except KeyError:
                print(f"Key error: {page_to}")
                append_nan_to_dict(metrics_dict_to)
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
    finished_paths, unfinished_paths = load_paths_pairs()
    finished_paths_metrics = get_paths_pairs_metrics(finished_paths)
    unfinished_paths_metrics = get_paths_pairs_metrics(unfinished_paths)
    finished_paths_metrics.to_csv("../data/finished_paths_pairs_metrics.csv")
    unfinished_paths_metrics.to_csv("../data/unfinished_paths_pairs_metrics.csv")