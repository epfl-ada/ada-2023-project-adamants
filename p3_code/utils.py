"""Utilities for P3 data loading."""
from urllib.parse import unquote
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
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

def load_data(data_path=DATA_PATH):
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
    articles.name = articles.name.apply(unquote)
    # add a column for each article name, and fill the dataframe with shortest path distances
    shortest_path_df = pd.DataFrame(
        shortest_path_matrix, index=articles.name, columns=articles.name
    )
    links = links.map(lambda x: unquote(x))
    return shortest_path_df, links, articles

def load_paths(
    path_finished=PATHS_FINISHED, path_unfinished=PATHS_UNFINISHED
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

def load_paths_pairs(data_path=DATA_PATH):
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
    # paths_pairs_df = pd.DataFrame()
    # count=0
    # for i in tqdm(range(len(paths_df))):
    #     row = paths_df.iloc[i]
    #     path = row["path"]
    #     if len(path) == 1:
    #         print(f"Skipping path {i} with only one page")
    #         continue
    #     for j in range(len(path) - 1):
    #         count+=1
    #         df = pd.DataFrame(
    #             {"from": path[j], 
    #              "to": path[j + 1],
    #              "rating": row["rating"],
    #              "timestamp": row["timestamp"],
    #              "hashedIpAddress": row["hashedIpAddress"],
    #              "durationInSec": row["durationInSec"]},
    #             index=[count],
    #         )
    #         paths_pairs_df = pd.concat([paths_pairs_df, df], ignore_index=True)
    # if save:
    #     paths_pairs_df.to_csv("./data/finished_paths_pairs.csv")
    # return paths_pairs_df
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
        