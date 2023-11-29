import sys
import pandas as pd
from pathlib import Path
import numpy as np
sys.path.append("../book")
from graph_measures import *
from utils import load_data, load_paths_pairs

GRAPH_METRICS_PATH = Path("../data/p3_extra_data/links_w_graph_metrics.csv").resolve()

def get_paths_pairs_metrics(paths_pairs, links_path=GRAPH_METRICS_PATH):
    """Adds the graph metrics to path pairs as extra columns.
    
    Makes a deep copy of the data, the modified copy is returned."""
    paths_pairs_metrics = paths_pairs.copy(deep=True)
    print("Computing graph metrics...")
    if Path(links_path).is_file():
        links_w_graph_metrics = pd.read_csv(GRAPH_METRICS_PATH)
        print("Succesfully loaded")
    else:
        _, links, _ = load_data()
        links_w_graph_metrics = compute_graph_metrics(links)
    metrics_dict = {
        "degree": [],
        "clustering": [],
        "closeness": [],
        "betweenness": [],
        "degree_centrality": [],
    }
    degree_diff = []
    print("_"*20)
    print("Adding graph metrics to path pairs...")
    total = len(paths_pairs_metrics)
    current = 0
    for row in paths_pairs_metrics.itertuples():
        if row[1] == "<" or row.to == "<":
            for k, v in metrics_dict.items():
                v.append(np.nan)
            continue
        metrics_row = links_w_graph_metrics.loc[
            (links_w_graph_metrics["from"] == row[1]) &
            (links_w_graph_metrics["to"] == row.to)
        ]
        for k, v in metrics_dict.items():
            try:
                v.append(metrics_row[k].values)
            except:
                print("Error in metrics:", metrics_row[k].values)
                v.append(np.nan)
        try:
            degree_from = links_w_graph_metrics.loc[
                links_w_graph_metrics["from"] == row[1]
            ]["degree"].values[0]
            degree_to = links_w_graph_metrics.loc[
                links_w_graph_metrics["to"] == row.to
            ]["degree"].values[0]
            degree_diff.append(degree_from - degree_to)
        except:
            print("Error in degree diff:", row[1], row.to)
            degree_diff.append(np.nan)
        current += 1
        print(f"{current}/{total}", end="\r")
    print("_"*20)
    print("Adding columns to dataframe...")
    metrics_dict["degree_diff"] = degree_diff
    for k, v in metrics_dict.items():
        paths_pairs_metrics[k] = v
        
    return paths_pairs_metrics


if __name__ == "__main__":
    finished_paths, unfinished_paths = load_paths_pairs()
    finished_paths_metrics = get_paths_pairs_metrics(finished_paths)
    unfinished_paths_metrics = get_paths_pairs_metrics(unfinished_paths)
    # finished_paths_metrics.to_csv("../data/finished_paths_pairs_metrics.csv")
    # unfinished_paths_metrics.to_csv("../data/unfinished_paths_pairs_metrics.csv")