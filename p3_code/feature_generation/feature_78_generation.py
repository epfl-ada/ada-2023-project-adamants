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
import pickle
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

GRAPH_METRICS_PATH = Path("../data/p3_extra_data/nodes_w_graph_metrics.csv").resolve()


def compute_path_metrics_w_nodes(nodes, paths_df, pickle_path = None):
    """Computes the metrics for each path."""
    pickle_path = Path(pickle_path).resolve()
    # pre-compute metrics for paths to avoid costly np.isin in the plots
    if not pickle_path.is_file() and pickle_path is not None:
        path_degree = pd.Series(dtype=object)
        path_clustering = pd.Series(dtype=object)
        path_degree_centrality = pd.Series(dtype=object)
        path_betweenness = pd.Series(dtype=object)
        path_closeness = pd.Series(dtype=object)

        page_degree_dict = dict(zip(nodes["node_name"], nodes["degree"]))
        page_clustering_dict = dict(zip(nodes["node_name"], nodes["clustering"]))
        page_degree_centrality_dict = dict(
            zip(nodes["node_name"], nodes["degree_centrality"])
        )
        page_betweenness_dict = dict(zip(nodes["node_name"], nodes["betweenness"]))
        page_closeness_dict = dict(zip(nodes["node_name"], nodes["closeness"]))

        not_found_count = 0

        total = len(paths_df)
        print("Total of paths: ", total)
        current = 0
        
        for i, path in enumerate(paths_df["path"]):
            page_degrees = []
            page_clustering = []
            page_degree_centrality = []
            page_betweenness = []
            page_closeness = []
            
            # if len(path) == 1:
            #     print(f"Path of len 1 : {path}\nSkipping...")
            #     continue
            
            for page in path:
                try:
                    page_degrees.append(page_degree_dict[page])
                    page_clustering.append(page_clustering_dict[page])
                    page_degree_centrality.append(page_degree_centrality_dict[page])
                    page_betweenness.append(page_betweenness_dict[page])
                    page_closeness.append(page_closeness_dict[page])
                except KeyError:
                    if page == "<":
                        continue
                    print(f"Page {page} not found in links")
                    not_found_count += 1
                    page_degrees.append(np.nan)
                    page_clustering.append(np.nan)
                    page_degree_centrality.append(np.nan)
                    page_betweenness.append(np.nan)
                    page_closeness.append(np.nan)

            path_degree[i] = np.array(page_degrees)
            path_clustering[i] = np.array(page_clustering)
            path_degree_centrality[i] = np.array(page_degree_centrality)
            path_betweenness[i] = np.array(page_betweenness)
            path_closeness[i] = np.array(page_closeness)
            
            current += 1
            print(f"Progress: {current}/{total}", end="\r")

        metrics_dict = {
            "path_degree": path_degree,
            "path_clustering": path_clustering,
            "path_degree_centrality": path_degree_centrality,
            "path_betweenness": path_betweenness,
            "path_closeness": path_closeness,
        }
        print(f"Total of pages not found in links: {not_found_count}")
        
        # save as pickle
        with Path(pickle_path).open("wb") as f:
            pickle.dump(metrics_dict, f)
        
        return metrics_dict
    elif Path(pickle_path).is_file():
        print("Loading metrics from pickle...")
        with Path(pickle_path).open("rb") as f:
            return pickle.load(f)

    else:
        raise ValueError(f"Invalid pickle path: {pickle_path}")


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

def compute_metrics_slopes(metrics_dict, invalid_values=np.nan):
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
                slopes[k][slope_bef].append(invalid_values)
                slopes[k][slope_aft].append(invalid_values)
                continue
            if np.isnan(val).all():
                slopes[k][slope_bef].append(invalid_values)
                slopes[k][slope_aft].append(invalid_values)
                continue
            max_idx = np.nanargmax(metrics_dict["path_degree"][i])
            if max_idx == 0:
                slopes[k][slope_bef].append(invalid_values)
                try:
                    slopes[k][slope_aft].append(np.polyfit(range(len(val)), val, 1)[0])
                except np.linalg.LinAlgError:
                    slopes[k][slope_aft].append(invalid_values)
                continue
            if max_idx == len(val) - 1:
                try:
                    slopes[k][slope_bef].append(np.polyfit(range(len(val)), val, 1)[0])
                except:
                    slopes[k][slope_bef].append(invalid_values)
                slopes[k][slope_aft].append(invalid_values)
                continue
            before_max = val[:max_idx]
            after_max = val[max_idx+1:]
            
            before_too_short = len(before_max) < 2
            after_too_short = len(after_max) < 2
        
            try:
                slope = np.polyfit(range(len(before_max)), before_max, 1)[0] / len(before_max)
                slopes[k][slope_bef].append(slope if not before_too_short else invalid_values)
            except np.linalg.LinAlgError:
                slopes[k][slope_bef].append(invalid_values)
            try:
                slope = np.polyfit(range(len(after_max)), after_max, 1)[0] / len(after_max)
                slopes[k][slope_aft].append(slope if not after_too_short else invalid_values)
            except np.linalg.LinAlgError:
                slopes[k][slope_aft].append(invalid_values)
        slopes[k] = pd.DataFrame.from_dict(slopes[k], orient="index").T
        slopes[k].columns = [str(k + "_slope_before"), str(k + "_slope_after")]
        
    return slopes

def compute_metrics_mean_diff(metrics_dict, invalid_values=np.nan):
    mean_diff = {}
    mean_diff_bef = "slope_before_max"
    mean_diff_aft = "slope_after_max"
    for k,v in metrics_dict.items():
        v.index = range(len(v))
        mean_diff[k] = {} 
        mean_diff[k][mean_diff_bef] = []
        mean_diff[k][mean_diff_aft] = []
        for i, val in enumerate(v):
            if len(val) < 3:
                mean_diff[k][mean_diff_bef].append(invalid_values)
                mean_diff[k][mean_diff_aft].append(invalid_values)
                continue
            if np.isnan(val).all():
                mean_diff[k][mean_diff_bef].append(invalid_values)
                mean_diff[k][mean_diff_aft].append(invalid_values)
                continue
            max_idx = np.nanargmax(metrics_dict["path_degree"][i])
            
            # take path with max included
            before_max = val[:max_idx+1]
            after_max = val[max_idx:]
            
            if len(before_max) < 2:
                mean_diff[k][mean_diff_bef].append(invalid_values)
                mean_diff[k][mean_diff_aft].append(invalid_values)
                continue
            if len(after_max) < 2:
                mean_diff[k][mean_diff_bef].append(invalid_values)
                mean_diff[k][mean_diff_aft].append(invalid_values)
                continue
            
            before_diff = np.diff(before_max)
            after_diff = np.diff(after_max)
            before_mean_diff = np.mean(before_diff) / len(before_diff)
            after_mean_diff = np.mean(after_diff) / len(after_diff)
            
            mean_diff[k][mean_diff_bef].append(before_mean_diff)
            mean_diff[k][mean_diff_aft].append(after_mean_diff)
        mean_diff[k] = pd.DataFrame.from_dict(mean_diff[k], orient="index").T
        mean_diff[k].columns = [str(k + "_slope_before"), str(k + "_slope_after")]
        
    return mean_diff

def compute_metrics_delta_norm(metrics_dict, invalid_values=np.nan):
    delta_norm = {}
    delta_norm_bef = "slope_before_max"
    delta_norm_aft = "slope_after_max"
    for k,v in metrics_dict.items():
        v.index = range(len(v))
        delta_norm[k] = {}
        delta_norm[k][delta_norm_bef] = []
        delta_norm[k][delta_norm_aft] = []
        for i, val in enumerate(v):
            if len(val) < 3:
                delta_norm[k][delta_norm_bef].append(invalid_values)
                delta_norm[k][delta_norm_aft].append(invalid_values)
                continue
            if np.isnan(val).all():
                delta_norm[k][delta_norm_bef].append(invalid_values)
                delta_norm[k][delta_norm_aft].append(invalid_values)
                continue
            max_idx = np.nanargmax(metrics_dict["path_degree"][i])
            
            # take path with max included
            before_max = val[:max_idx+1]
            after_max = val[max_idx:]
            
            if len(before_max) < 2:
                delta_norm[k][delta_norm_bef].append(invalid_values)
                delta_norm[k][delta_norm_aft].append(invalid_values)
                continue
            if len(after_max) < 2:
                delta_norm[k][delta_norm_bef].append(invalid_values)
                delta_norm[k][delta_norm_aft].append(invalid_values)
                continue
            
            before_delta = before_max[-1] - before_max[0]
            after_delta = after_max[-1] - after_max[0]
            
            before_norm = before_delta / len(before_max)
            after_norm = after_delta / len(after_max)
            
            delta_norm[k][delta_norm_bef].append(before_norm)
            delta_norm[k][delta_norm_aft].append(after_norm)
        delta_norm[k] = pd.DataFrame.from_dict(delta_norm[k], orient="index").T
        delta_norm[k].columns = [str(k + "_slope_before"), str(k + "_slope_after")]
        
    return delta_norm

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

def add_computed_graph_features(path_dataframe, target_dataframe, metrics_pickle_path, use_log=True,drop_duplicates_from_merge=True):
    """Adds the computed graph features to the dataframe."""
    path_dataframe = path_dataframe.copy(deep=True)
    nodes = pd.read_csv(GRAPH_METRICS_PATH)
    
    metrics_dict = compute_path_metrics_w_nodes(nodes, path_dataframe, metrics_pickle_path)
    
    # slopes = compute_metrics_slopes(metrics_dict)
    slopes = compute_metrics_mean_diff(metrics_dict)
    # slopes = compute_delta_norm(metrics_dict)

    slopes_temp = slopes.copy()
    for k, v in slopes.items():
        print(f"Metric {k}")
        abs_sum = v[str(k)+"_slope_before"].abs() + v[str(k)+"_slope_after"].abs()
        if use_log:
            abs_sum = abs_sum.apply(lambda x : np.log(x+1e-10))
        slopes_temp[str(k)+"_abs_sum"] = abs_sum
    slopes = slopes_temp
    slopes_abs_sum = {}
    for k,v in slopes.items():
        if "abs_sum" in k:
            slopes_abs_sum[k] = v
    slopes_abs_sum  = pd.DataFrame(slopes_abs_sum)
    assert len(slopes_abs_sum) == len(path_dataframe), f"Length of slopes_abs_sum ({len(slopes_abs_sum)}) and path_dataframe ({len(path_dataframe)}) do not match"
    concat_df = pd.concat([path_dataframe, slopes_abs_sum], axis=1)
    merged_df = pd.merge(target_dataframe, concat_df, how="left", on=["hashedIpAddress", "timestamp", "durationInSec"])
    
    if drop_duplicates_from_merge:
        merged_df = merged_df.loc[:,~merged_df.columns.str.endswith("_y")]
        merged_df = merged_df.rename(columns={col: col[:-2] for col in merged_df.columns if col.endswith("_x")})
    return merged_df