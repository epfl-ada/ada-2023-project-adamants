from urllib.parse import unquote
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from functools import partial

DATA_PATH = Path.cwd() / "../data/wikispeedia_paths-and-graph/"
SHORTEST_PATH = (DATA_PATH / "shortest-path-distance-matrix.txt").resolve()
PATHS_FINISHED = (DATA_PATH / "paths_finished.tsv").resolve()
PATHS_UNFINISHED = (DATA_PATH / "paths_unfinished.tsv").resolve()

# pickle with metrics per path
PATH_METRICS_PICKLE_PATH = (DATA_PATH / "../path_metrics.pkl").resolve()

####################
# STATS AND UTILS  #
####################


def convert_to_matrix(data):
    """Replaces each row (string) with the integer values of the string and replaces _ with NaN"""
    data = np.array([[int(i) if i != "_" else np.nan for i in row] for row in data])
    return data


def parse_paths(dataframe):
    """Parses the path column into a list of strings"""
    try:
        dataframe["path"] = dataframe["path"].map(lambda x: x.split(";"))
        return dataframe
    except:
        print("The dataframe does not contain a path column")


# compute the tendency of the path degree to first increase and then decrease
def compute_path_degree_evolution(path_degrees):
    """Computes the tendency of the path degree to first increase and then decrease.

    Args :
        path_degrees (np.array) : array of degrees of the pages in the path
    """
    if len(path_degrees) == 0:
        return np.nan
    return np.diff(path_degrees)


def compute_slope(path_metric_evolution):
    """Quantifies if the slope of a metric in a path is increasing or decreasing.

    This is done by fitting a line to the metric and returning the slope.
    """
    idx = np.isfinite(path_metric_evolution)
    if len(path_metric_evolution[idx]) < 2:
        return np.nan
    fitted_line = np.polyfit(
        np.arange(len(path_metric_evolution))[idx], path_metric_evolution[idx], 1
    )
    return fitted_line[0]


def estimate_strategy(metric_array, max_array, metric=compute_slope):
    """Uses the maximum of the degree in the path to estimate the strategy of the user.

    The goal is to see if the degree tends to increase before reaching the maximum and then decrease.

    Args:
        metric_array (np.array): array of the metric for the pages in the path
        max_array (np.array): array of which the maximum is used to estimate the strategy, usually the degree
        metric (function): function to use to compute the metric of the degree evolution before and after the maximum. Can use compute_slope or np.mean for example.

    Returns:
        m_before (float): the metric of the degree evolution before the maximum
        m_after (float): the metric of the degree evolution after the maximum
    """
    # assert metric_array.shape == max_array.shape
    # print(degree_evolution)
    if len(metric_array) < 3:
        print("Path too short")
        return np.nan, np.nan
    
    if np.isnan(metric_array).all():
        print("All NaN")
        return np.nan, np.nan
    max_id = np.nanargmax(max_array)
    if max_id == 0:
        print("Max at 0")
        return np.nan, np.nan
    elif max_id == len(metric_array) - 1:
        print("Max at end")
        return np.nan, np.nan
    # elif len(metric_array) == 1:
    #     return np.nan, np.nan
    else:
        m_before = metric(metric_array[:max_id])
        m_after = metric(metric_array[max_id - 1 :])
        return m_before, m_after


def compute_path_metric_around_max(metric_array, max_array):
    """Computes the metric of the metric array before and after the maximum of the max array.

    Args:
        metric_array (np.array): array of the metric for the pages in the path
        max_array (np.array): array of the metric for the pages in the path, usually the degree

    Returns:
        metric_before_max (np.array): array of the metric for the pages in the path before the maximum
        metric_after_max (np.array): array of the metric for the pages in the path after the maximum
    """
    metric_before_max = []
    metric_after_max = []
    for i, path_metric in enumerate(metric_array):
        if np.isnan(max_array[i]).all():
            continue
        max_id = np.nanargmax(max_array[i])
        if max_id == 0:
            continue
        metric_before_max.append(path_metric[:max_id])
        metric_after_max.append(path_metric[max_id - 1 :])

    metric_before_max = np.concatenate(metric_before_max)
    metric_after_max = np.concatenate(metric_after_max)
    return metric_before_max, metric_after_max


#############################
# DATA LOADING AND ANALYSIS #
#############################


def load_data_graph_metrics(data_path=DATA_PATH, unquote_names=True):
    """Loads the Wikispeedia data for graph metrics analysis.

    Returns:
        shortest_path_df (pd.DataFrame): a dataframe with the shortest path distances between each article
        links (pd.DataFrame): a dataframe with the links between articles
        articles (pd.DataFrame): a dataframe with the article names
    """
    shortest_path = (data_path / "shortest-path-distance-matrix.txt").resolve()
    assert shortest_path.is_file()
    shortest_path = np.loadtxt(shortest_path, dtype=str)

    links_path = (data_path / "links.tsv").resolve()
    assert links_path.is_file()
    links = pd.read_csv(
        links_path,
        sep="\t",
        header=None,
        names=["from", "to"],
        skiprows=11,
        skip_blank_lines=True,
    )

    articles_path = (data_path / "articles.tsv").resolve()
    assert articles_path.is_file()
    articles = pd.read_csv(
        articles_path,
        sep="\t",
        header=None,
        names=["name"],
        skiprows=11,
        skip_blank_lines=True,
    )

    shortest_path_matrix = convert_to_matrix(shortest_path)
    if unquote_names:
        articles.name = articles.name.apply(unquote)
        links = links.map(lambda x: unquote(x))
    shortest_path_df = pd.DataFrame(
        shortest_path_matrix, index=articles.name, columns=articles.name
    )
    return shortest_path_df, links, articles


def build_links_graph(links):
    """Builds a directed graph from the links dataframe.

    Args:
        links (pd.DataFrame): a dataframe with the links between articles

    Returns:
        G (nx.DiGraph): a directed graph"""
    G = nx.DiGraph()
    G.add_edges_from(links.values)
    print(G)
    return G


def compute_graph_metrics(links, save_csv=False):
    """Compute degree, clustering, local efficiency, modularity, closeness centrality, betweenness centrality, and degree centrality.

    Args:
        links (pd.DataFrame): a dataframe with the links between articles

    Returns:
        links (pd.DataFrame): a dataframe with the links between articles and the graph metrics added as extra columns
    """
    # create a directed graph
    links = links.copy()
    G = build_links_graph(links)
    print("Computing degree...")
    degree = dict(G.degree(G.nodes()))
    nx.set_node_attributes(G, degree, "degree")
    # local efficiency
    # efficiency = nx.local_efficiency(G.to_undirected())
    # efficiency # 0.55814
    print(f"Efficiency (pre-calculated as it takes long to compute) : 0.55814")
    # modularity
    modularity = nx.algorithms.community.modularity(
        G.to_undirected(),
        nx.algorithms.community.label_propagation.label_propagation_communities(
            G.to_undirected()
        ),
    )
    print(f"Modularity: {modularity}")

    print("Computing metrics...")
    links["degree"] = links["from"].map(degree)
    # compute local clustering coefficient for each node
    print("Local clustering coefficient...")
    clustering = nx.clustering(G)
    links["clustering"] = links["from"].map(clustering)
    # centrality
    # closeness centrality
    print("Closeness centrality...")
    closeness = nx.closeness_centrality(G)
    links["closeness"] = links["from"].map(
        closeness
    )  # represents the average distance to all other nodes
    # betweenness centrality
    print("Betweenness centrality...")
    betweenness = nx.betweenness_centrality(G)
    links["betweenness"] = links["from"].map(
        betweenness
    )  # represents the number of shortest paths that pass through a node
    # degree centrality
    print("Degree centrality...")
    centrality = nx.degree_centrality(G)
    links["degree_centrality"] = links["from"].map(
        centrality
    )  # represents the number of shortest paths that pass through a node (ie is the node a hub)
    if save_csv:
        links.to_csv((DATA_PATH / "../p3_extra_data/links_w_graph_metrics.csv").resolve())
    return links

def compute_node_metrics(links, save_csv=False):
    """Compute degree, clustering, local efficiency, modularity, closeness centrality, betweenness centrality, and degree centrality for each unique node.
    
    Args:
        links (pd.DataFrame): a dataframe with the links between articles
        
    Returns:
            links (pd.DataFrame): a dataframe with the links between articles and the graph metrics added as extra columns"""
    links = links.copy()
    G = build_links_graph(links)
    print("Computing degree...")
    degree = dict(G.degree(G.nodes()))
    nx.set_node_attributes(G, degree, "degree")
    clustering = nx.clustering(G)
    print("Local clustering coefficient...")
    nx.set_node_attributes(G, clustering, "clustering")
    closeness = nx.closeness_centrality(G)
    print("Closeness centrality...")
    nx.set_node_attributes(G, closeness, "closeness")
    betweenness = nx.betweenness_centrality(G)
    print("Betweenness centrality...")
    nx.set_node_attributes(G, betweenness, "betweenness")
    centrality = nx.degree_centrality(G)
    print("Degree centrality...")
    nx.set_node_attributes(G, centrality, "degree_centrality")
    # find all unique node names in the graph, from and to
    unique_nodes = set(links["from"].unique()).union(set(links["to"].unique()))
    nodes_df = pd.DataFrame(list(unique_nodes), columns=["node_name"])
    nodes_df["degree"] = nodes_df["node_name"].map(degree)
    nodes_df["clustering"] = nodes_df["node_name"].map(clustering)
    nodes_df["closeness"] = nodes_df["node_name"].map(closeness)
    nodes_df["betweenness"] = nodes_df["node_name"].map(betweenness)
    nodes_df["degree_centrality"] = nodes_df["node_name"].map(centrality)
    
    if save_csv:
        nodes_df.to_csv((DATA_PATH / "../p3_extra_data/nodes_w_graph_metrics.csv").resolve())
    return nodes_df
    

def load_and_prepare_paths_dfs_for_metrics(
    path_finished=PATHS_FINISHED, path_unfinished=PATHS_UNFINISHED
):
    """Loads and creates a path list in the path column of the paths_finished and paths_unfinished dataframes."""
    # data exploration
    path_finished = Path(path_finished).resolve()
    assert path_finished.is_file()
    paths_finished = pd.read_csv(
        path_finished,
        sep="\t",
        header=None,
        names=["hashedIpAddress", "timestamp", "durationInSec", "path", "rating"],
        encoding="utf-8",
        skiprows=16,
    ).copy(deep=True)
    paths_finished["timestamp"] = pd.to_datetime(paths_finished["timestamp"], unit="s")

    paths_finished = parse_paths(paths_finished)
    paths_finished["path"] = paths_finished["path"].apply(
        lambda x: [unquote(i) for i in x]
    )

    path_unfinished = Path(path_unfinished).resolve()
    assert path_unfinished.is_file()
    paths_unfinished = pd.read_csv(
        path_unfinished,
        sep="\t",
        header=None,
        names=[
            "hashedIpAddress",
            "timestamp",
            "durationInSec",
            "path",
            "target",
            "type",
        ],
        encoding="utf-8",
        skiprows=17,
    ).copy(deep=True)
    paths_unfinished["timestamp"] = pd.to_datetime(
        paths_unfinished["timestamp"], unit="s"
    )

    paths_unfinished = parse_paths(paths_unfinished)
    paths_unfinished["path"] = paths_unfinished["path"].apply(
        lambda x: [unquote(i) for i in x]
    )

    return paths_finished, paths_unfinished


def compute_path_metrics(links, paths, pickle_path=PATH_METRICS_PICKLE_PATH):
    """Computes the metrics for each path."""
    pickle_path = Path(pickle_path).resolve()
    print(f"Attemping to load metrics from {pickle_path}")
    if Path(pickle_path).is_file():
        with open(pickle_path, "rb") as f:
            pickle_dict = pickle.load(f)
        print(f"Loaded metrics successfully from {pickle_path}")
    else:
        print(f"Pickle file not found at {pickle_path}. Computing metrics...")
        # pre-compute metrics for paths to avoid costly np.isin in the plots
        path_degree = pd.Series(dtype=object)
        path_clustering = pd.Series(dtype=object)
        path_degree_centrality = pd.Series(dtype=object)
        path_betweenness = pd.Series(dtype=object)
        path_closeness = pd.Series(dtype=object)

        page_degree_dict = dict(zip(links["from"], links["degree"]))
        page_clustering_dict = dict(zip(links["from"], links["clustering"]))
        page_degree_centrality_dict = dict(
            zip(links["from"], links["degree_centrality"])
        )
        page_betweenness_dict = dict(zip(links["from"], links["betweenness"]))
        page_closeness_dict = dict(zip(links["from"], links["closeness"]))

        not_found_count = 0

        for i, path in enumerate(paths["path"]):
            page_degrees = []
            page_clustering = []
            page_degree_centrality = []
            page_betweenness = []
            page_closeness = []
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

        pickle_dict = {
            "path_degree": path_degree,
            "path_clustering": path_clustering,
            "path_degree_centrality": path_degree_centrality,
            "path_betweenness": path_betweenness,
            "path_closeness": path_closeness,
        }
        print(f"Total of pages not found in links: {not_found_count}")
        with open(pickle_path, "wb") as f:
            pickle.dump(pickle_dict, f)
    return pickle_dict


def get_metrics_around_max(metrics_dict):
    """Extracts the metrics before and after the maximum of the degree."""
    path_degree = metrics_dict["path_degree"]
    path_clustering = metrics_dict["path_clustering"]
    path_degree_centrality = metrics_dict["path_degree_centrality"]
    path_betweenness = metrics_dict["path_betweenness"]
    path_closeness = metrics_dict["path_closeness"]

    path_degree_evolutions = pd.Series(dtype=object)
    path_degree_evolutions = path_degree.map(compute_path_degree_evolution)

    degree_before_max, degree_after_max = compute_path_metric_around_max(
        path_degree, path_degree
    )
    (
        degree_evolutions_before_max,
        degree_evolutions_after_max,
    ) = compute_path_metric_around_max(path_degree_evolutions, path_degree)
    clustering_before_max, clustering_after_max = compute_path_metric_around_max(
        path_clustering, path_degree
    )
    (
        degree_centrality_before_max,
        degree_centrality_after_max,
    ) = compute_path_metric_around_max(path_degree_centrality, path_degree)
    betweenness_before_max, betweenness_after_max = compute_path_metric_around_max(
        path_betweenness, path_degree
    )
    closeness_before_max, closeness_after_max = compute_path_metric_around_max(
        path_closeness, path_degree
    )

    return {
        "degree": [degree_before_max, degree_after_max],
        "degree_evolutions": [
            degree_evolutions_before_max,
            degree_evolutions_after_max,
        ],
        "clustering": [clustering_before_max, clustering_after_max],
        "degree_centrality": [
            degree_centrality_before_max,
            degree_centrality_after_max,
        ],
        "betweenness": [betweenness_before_max, betweenness_after_max],
        "closeness": [closeness_before_max, closeness_after_max],
    }


def plot_histograms_of_metrics_before_and_after(metrics_dict):
    """Plotting helper for the metrics before and after the maximum of the degree."""
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    fig.suptitle("Metrics before and after the maximum of the degree")

    for ax in axs.flatten():
        ax.set_xlabel("Metric")
        ax.set_ylabel("Frequency")

    metric_titles = {
        "degree": "Degree",
        "degree_evolutions": "Degree evolution",
        "clustering": "Clustering coefficient",
        "degree_centrality": "Degree centrality",
        "betweenness": "Betweenness centrality",
        "closeness": "Closeness centrality",
    }

    for i, metric in enumerate(metrics_dict.keys()):
        cur_ax = axs.flatten()[i]
        sns.histplot(metrics_dict[metric][0], ax=cur_ax, kde=True)
        sns.histplot(metrics_dict[metric][1], ax=cur_ax, kde=True)
        cur_ax.legend(["Before max", "After max"])
        cur_ax.set_title(metric_titles[metric])

    fig.tight_layout()
    plt.show()


def compute_metric_slope_before_and_after(metrics, drop_na=True):
    """Computes the slope of the metric before and after the maximum of the degree."""
    path_slopes = pd.Series(dtype=object)
    path_slopes = metrics.map(compute_slope)
    path_slopes = pd.DataFrame(
        path_slopes.tolist(), columns=["slope_before", "slope_after"]
    )
    if drop_na:
        na_count = path_slopes.isna().sum().sum()
        path_slopes.dropna(inplace=True)
        print(f"Dropped {na_count} NaN values")


def compute_metric_slopes(metrics_dict, drop_na=True):
    """Computes the slope of the metric before and after the maximum of the degree.

    Args:
        metrics_dict (dict): a dictionary with the metrics before and after the maximum of the degree

    Returns:
        metrics_slopes (pd.DataFrame): a dataframe with the slope of the metric before and after the maximum of the degree
    """
    na_count = 0

    metrics_slopes_list = []
    for i, metric in enumerate(metrics_dict.keys()):
        print(f"Adding slopes for {metric}")
        metric_slopes = pd.Series(dtype=object)
        metric_slopes = metrics_dict[metric].map(
            partial(estimate_strategy, metric=compute_slope, max_array=metrics_dict["path_degree"][i])
        )
        try:
            metric_slopes = pd.DataFrame(
                metric_slopes.tolist(),
                columns=[f"slope_{metric}_before", f"slope_{metric}_after"],
            )
            if drop_na:
                # drop NaN but keep the index
                na_count += metric_slopes.isna().sum().sum()
                metric_slopes.dropna(inplace=True)
        except ValueError as e:
            print(f"Could not compute slopes for {metric} : {e}")
            print(metric_slopes)
        metrics_slopes_list.append(metric_slopes)
    print(f"Dropped {na_count} NaN values")
    return metrics_slopes_list

def add_slopes_to_paths_df(paths_df, metrics_slopes):
    """Adds the slopes of the metrics before and after the maximum of the degree to the paths dataframe."""
    for metric_slopes in metrics_slopes:
        paths_df = pd.concat([paths_df, metric_slopes], axis=1)
    return paths_df

def compute_path_metrics_w_nodes(nodes, paths_df):
    """Computes the metrics for each path."""
    # pre-compute metrics for paths to avoid costly np.isin in the plots
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
        
        if len(path) == 1:
            print(f"Path of len 1 : {path}\nSkipping...")
            continue
        
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
    return metrics_dict

def append_features_to_paths(paths_df, nodes):
    """Adds the slopes of the metrics before and after the maximum of the degree to the paths dataframe."""
    print("- Metrics")
    metrics_dict = compute_path_metrics_w_nodes(nodes, paths_df)
    print("- Slopes")
    metrics_slopes = compute_metric_slopes(metrics_dict, drop_na=False)
    print("- Adding slopes to dataframe")
    # print(metrics_slopes)
    paths_df = add_slopes_to_paths_df(paths_df, metrics_slopes)
    return paths_df