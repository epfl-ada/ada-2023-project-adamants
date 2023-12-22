"""Utilities for P3 data loading."""
from urllib.parse import unquote
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
from functools import partial
# from tqdm.notebook import tqdm
from tqdm import tqdm

DATA_PATH = Path.cwd() / "../data/wikispeedia_paths-and-graph/"
SHORTEST_PATH = (DATA_PATH / "shortest-path-distance-matrix.txt").resolve()
PATHS_FINISHED = (DATA_PATH / "paths_finished.tsv").resolve()
PATHS_UNFINISHED = (DATA_PATH / "paths_unfinished.tsv").resolve()
PATHS_PAIRS_FINISHED = (DATA_PATH / "../paths_finished_pairs.csv").resolve()
PATHS_PAIRS_UNFINISHED = (DATA_PATH / "../paths_unfinished_pairs.csv").resolve()



##############
# DATA UTILS #
##############

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

def load_data(data_path=DATA_PATH, unquote_names=True):
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
    # add a column for each article name, and fill the dataframe with shortest path distances
    shortest_path_df = pd.DataFrame(
        shortest_path_matrix, index=articles.name, columns=articles.name
    )
    return shortest_path_df, links, articles

def load_paths(
    path_finished=PATHS_FINISHED, path_unfinished=PATHS_UNFINISHED, unquote_names=True, drop_timeouts=True
):
    """Loads and creates a path list in the path column of the paths_finished and paths_unfinished dataframes."""
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
    
    if unquote_names:
        paths_finished["path"] = paths_finished["path"].apply(
            lambda x: [unquote(i) for i in x]
        )
        paths_unfinished["path"] = paths_unfinished["path"].apply(
            lambda x: [unquote(i) for i in x]
        )
    
    if drop_timeouts:
        # filter all with type timeout and 0 ; characters in path
        paths_unfinished = paths_unfinished[
            not "timeout" in paths_unfinished["type"]
            and paths_unfinished["path"].count(";") > 0
            ]
        paths_unfinished.reset_index(inplace=True)

    return paths_finished, paths_unfinished

def load_paths_pairs():
    """Loads the paths_finished dataframe and converts the path column to a list of page pairs."""
    finished_pairs_df = pd.read_csv(
        PATHS_PAIRS_FINISHED,
        index_col="index"
    )
    unfinished_pairs_df = pd.read_csv(
        PATHS_PAIRS_UNFINISHED,
        index_col="index"
    )
    return finished_pairs_df, unfinished_pairs_df

def convert_paths_to_pages_pairs(paths_df):
    """Converts the path column of the paths_finished dataframe to a list of page pairs."""
    path_lengths = paths_df["path"].map(len)
    path_lengths = path_lengths[path_lengths > 1]
    paths_df = paths_df.loc[path_lengths.index]
    path_lengths = path_lengths.map(lambda x: x - 1)
    
    from_series = paths_df["path"].map(lambda x: x[:-1])
    to_series = paths_df["path"].map(lambda x: x[1:])
    
    from_series = from_series.explode().to_numpy()
    to_series = to_series.explode().to_numpy()
    
    total_length = len(from_series)
    indices = np.arange(total_length)
    
    from_series = pd.Series(from_series, index=indices)
    to_series = pd.Series(to_series, index=indices)

    # rating_series = paths_df["rating"].repeat(path_lengths).to_numpy()
    timestamp_series = paths_df["timestamp"].repeat(path_lengths).to_numpy()
    hashedIpAddress_series = paths_df["hashedIpAddress"].repeat(path_lengths).to_numpy()
    durationInSec_series = paths_df["durationInSec"].repeat(path_lengths).to_numpy()
    
    paths_pairs_df = pd.DataFrame(
        index=indices,
    )
    paths_pairs_df["from"] = from_series
    paths_pairs_df["to"] = to_series
    # paths_pairs_df["rating"] = rating_series
    paths_pairs_df["timestamp"] = timestamp_series
    paths_pairs_df["hashedIpAddress"] = hashedIpAddress_series
    paths_pairs_df["durationInSec"] = durationInSec_series
    
    return paths_pairs_df

### clustering ###

FEATURES_COLS_USED_FOR_CLUSTERING = [
    "durationInSec",
    #   'rating_x',
    "backtrack",
    "numberOfPath",
    # "link_position",
    "position_mean",
    "position_std",
    "path_length",
    #  'optimal_path_length',
    "coarse_mean_time",
    "semantic_similarity",
    "ratio",
    # "path_degree_slope_before",
    # "path_degree_slope_after",
    # "path_clustering_slope_before",
    # "path_clustering_slope_after",
    # "path_degree_centrality_slope_before",
    # "path_degree_centrality_slope_after",
    # "path_betweenness_slope_before",
    # "path_betweenness_slope_after",
    # "path_closeness_slope_before",
    # "path_closeness_slope_after",
    'path_closeness_abs_sum'
]

COLS_REPLACE_NAN_WITH_MEAN = [ # cols where nan values are replaced with mean of the column
    # "path_degree_slope_before",
    # "path_degree_slope_after",
    # "path_clustering_slope_before",
    # "path_clustering_slope_after",
    # "path_degree_centrality_slope_before",
    # "path_degree_centrality_slope_after",
    # "path_betweenness_slope_before",
    # "path_betweenness_slope_after",
    # "path_closeness_slope_before",
    # "path_closeness_slope_after",
    'path_closeness_abs_sum',
    "position_mean",
    "position_std",
]
COLS_LOG = [ # cols to apply log transformation
    "durationInSec",
    "numberOfPath",
    "path_length",
    "coarse_mean_time",
]


def replace_nan_with_mean(df, cols):
    for col in cols:
        df[col] = df[col].fillna(df[col].mean())
    return df

def replace_value_with_mean(df, cols, value):
    for col in cols:
        df[col] = df[col].replace(value, df[col].mean())
    return df

def normalize_features(df, cols_log=COLS_LOG, cols_replace_nan_with_mean=COLS_REPLACE_NAN_WITH_MEAN):
    df=df.copy(deep=True)
    # replace NaN with mean
    df = replace_nan_with_mean(df, cols_replace_nan_with_mean)
    # take the log of a set determined features to make them more normally distributed
    df[cols_log] = np.log(df[cols_log])
    # get the z-score of each features, to have them all on the same scale
    df = df.apply(lambda x: (x - x.mean()) / x.std())
    return df

## PLOT UTILS ##

def camel_to_snake(s):
    """Goes from camelCase to snake_case"""
    s = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s).lower()

def get_feature_names_labels():
    """Returns the feature names and labels for the clustering plots."""
    feat_labels = FEATURES_COLS_USED_FOR_CLUSTERING
    feat_labels = [camel_to_snake(x) for x in feat_labels]
    # remove all underscores
    feat_labels = [x.replace("_", " ") for x in feat_labels]
    # cast letter in beginning to uppercase
    feat_labels = [x[0].upper() + x[1:] for x in feat_labels]
    return feat_labels

def set_axis_style(axs, i, add_xlabel=True, add_ylabel=True):
    feat_labels = get_feature_names_labels()
    if add_xlabel:
        axs[i].set_xlabel("Cluster", fontsize=20)
    if add_ylabel:
        try:
            axs[i].set_ylabel(feat_labels[i], fontsize=20)
        except:
            pass
    axs[i].patch.set_facecolor("#d3d3d3")
    leg = axs[i].get_legend()
    leg.get_frame().set_facecolor("#d3d3d3")
    leg.set_title("Cluster")
    leg_title = leg.get_title()
    leg_title.set_color("black")
    # change all legend text to black
    [text.set_color("black") for text in leg.get_texts()]
    
def format_3d_plot(fig):
    fig.update_layout({"plot_bgcolor": "#14181e", "paper_bgcolor": "#14181e"})
    fig.update_layout(font_color="white")
    fig.update_layout(scene=dict(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False), zaxis=dict(showticklabels=False)))
    fig.update_layout(legend_title_text="Cluster")
    fig.update_layout(legend = dict(bgcolor = 'rgba(0,0,0,0)'))
    fig.update_layout(scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"))
    fig.update_layout(scene = dict(
        xaxis = dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                # gridcolor="rgba(0, 0, 0,0)", # gridcolor is for logo
                showbackground=True,
                zerolinecolor="white",),
        yaxis = dict(
            backgroundcolor="rgba(0, 0, 0,0)",
            # gridcolor="rgba(0, 0, 0,0)",
            showbackground=True,
            zerolinecolor="white"),
        zaxis = dict(
            backgroundcolor="rgba(0, 0, 0,0)",
            # gridcolor="rgba(0, 0, 0,0)",
            showbackground=True,
            zerolinecolor="white",),),
    )