import sys
import pandas as pd
from pathlib import Path
import numpy as np
import os
sys.path.append("../book")
sys.path.append("book/") # Attend to more cases
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

def compute_metrics_slopes(metrics_dict):
    slopes = {}
    slope_bef = "slope_before_max"
    slope_aft = "slope_after_max"
    for k,v in metrics_dict.items():
        v.index = range(len(v))
        slopes[k] = {} 
        slopes[k][slope_bef] = []
        slopes[k][slope_aft] = []
        for i, val in enumerate(v):
            if len(val) < 3:
                slopes[k][slope_bef].append(np.nan)
                slopes[k][slope_aft].append(np.nan)
                continue
            if np.isnan(val).all():
                slopes[k][slope_bef].append(np.nan)
                slopes[k][slope_aft].append(np.nan)
                continue
            max_idx = np.nanargmax(metrics_dict["path_degree"][i])
            if max_idx == 0:
                slopes[k][slope_bef].append(np.nan)
                slopes[k][slope_aft].append(np.nan)
                continue
            if max_idx == len(val) - 1:
                slopes[k][slope_bef].append(np.nan)
                slopes[k][slope_aft].append(np.nan)
                continue
            before_max = val[:max_idx]
            after_max = val[max_idx+1:]
            
            before_too_short = len(before_max) < 2
            after_too_short = len(after_max) < 2
        
            try:
                slopes[k][slope_bef].append(np.polyfit(range(len(before_max)), before_max, 1)[0] if not before_too_short else np.nan)
            except np.linalg.LinAlgError:
                slopes[k][slope_bef].append(np.nan)
            try:
                slopes[k][slope_aft].append(np.polyfit(range(len(after_max)), after_max, 1)[0] if not after_too_short else np.nan)
            except np.linalg.LinAlgError:
                slopes[k][slope_aft].append(np.nan)
        slopes[k] = pd.DataFrame.from_dict(slopes[k], orient="index").T
        slopes[k].columns = [str(k + "_slope_before"), str(k + "_slope_after")]
        
    return slopes


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
        metrics_dict_fin = compute_path_metrics_w_nodes(
            nodes, finished_paths 
        )
        slopes_fin = compute_metrics_slopes(metrics_dict_fin)
        finished_paths = finished_paths.copy(deep=True)
        
        slopes_fin_df = pd.DataFrame()
        for k,v in slopes_fin.items():
            slopes_fin_df = pd.concat([slopes_fin_df, v], axis=1)

        finished_paths_modif = finished_paths.copy()
        finished_paths_modif = pd.concat([finished_paths_modif, slopes_fin_df], axis=1)
        finished_paths_modif.to_csv(finished_paths_appended_metrics_path)
        
    if not unfinished_paths_appended_metrics_path.is_file():
        print("Computing unfinished paths metrics...")
        metrics_dict_unfin = compute_path_metrics_w_nodes(
            nodes, unfinished_paths 
        )
        slopes_unfin = compute_metrics_slopes(metrics_dict_unfin)

        slopes_unfin_df = pd.DataFrame()
        for k,v in slopes_unfin.items():
            slopes_unfin_df = pd.concat([slopes_unfin_df, v], axis=1)
            
        paths_unfinished_modif = unfinished_paths.copy()
        paths_unfinished_modif = pd.concat([paths_unfinished_modif, slopes_unfin_df], axis=1)
        paths_unfinished_modif.to_csv(unfinished_paths_appended_metrics_path)
    print("Done")
