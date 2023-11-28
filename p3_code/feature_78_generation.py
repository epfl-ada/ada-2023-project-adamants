import sys
import pandas as pd
from pathlib import Path

sys.path.append("../book")
from graph_measures import *
from utils import load_data, load_paths_pairs

GRAPH_METRICS_PATH = Path("../data/links_w_graph_metrics.csv")

def get_paths_pairs_metrics(paths_pairs, links):
    """Adds the graph metrics to path pairs as extra columns.
    
    Makes a deep copy of the data, the modified copy is returned."""
    print("Computing graph metrics...")
    links_w_graph_metrics = compute_graph_metrics(links)
    paths_pairs_metrics = paths_pairs.copy(deep=True)
    metrics_dict = {
        "degree": [],
        "clusering": [],
        "closeness": [],
        "betweenness": [],
        "degree_centrality": [],
    }
    degree_diff = []
    print("_"*20)
    print("Adding graph metrics to path pairs...")
    for row in paths_pairs_metrics.iterrow():
        # locate matching from/to columns in the links
        # and copy the graph metrics to the paths_pairs_metrics dataframe
        metrics_row = links_w_graph_metrics.loc[
            (links_w_graph_metrics["from"] == row["from"]) &
            (links_w_graph_metrics["to"] == row["to"])
        ]
        for k, v in metrics_dict.items():
            v.append(metrics_row[k].values[0])
        
        degree_from = links_w_graph_metrics.loc[
            links_w_graph_metrics["from"] == row["from"]
        ]["degree"].values[0]
        degree_to = links_w_graph_metrics.loc[
            links_w_graph_metrics["to"] == row["to"]
        ]["degree"].values[0]
        degree_diff.append(degree_from - degree_to)
    print("_"*20)
    print("Adding columns to dataframe...")
    metrics_dict["degree_diff"] = degree_diff
    for k, v in metrics_dict.items():
        paths_pairs_metrics[k] = v
        
    return paths_pairs_metrics


if __name__ == "__main__":
    finished_paths, unfinished_paths = load_paths_pairs()
    _, links, _ = load_data()
    finished_paths_metrics = get_paths_pairs_metrics(finished_paths, links)
    unfinished_paths_metrics = get_paths_pairs_metrics(unfinished_paths, links)
    # finished_paths_metrics.to_csv("../data/finished_paths_pairs_metrics.csv")
    # unfinished_paths_metrics.to_csv("../data/unfinished_paths_pairs_metrics.csv")