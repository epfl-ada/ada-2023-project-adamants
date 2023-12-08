from urllib.parse import unquote
import pandas as pd
import numpy as np
from pathlib import Path

PATH = Path.cwd()
PATH = PATH.parent
DATA_PATH = PATH / "data/wikispeedia_paths-and-graph/"
PATHS_FINISHED = (DATA_PATH / "paths_finished.tsv").resolve()
PATHS_UNFINISHED = (DATA_PATH / "paths_unfinished.tsv").resolve()
shortest_path = (DATA_PATH / "shortest-path-distance-matrix.txt").resolve()
assert shortest_path.is_file()
shortest_path = np.loadtxt(shortest_path, dtype=str)

def convert_to_matrix(data):
    """Replaces each row (string) with the integer values of the string and replaces _ with NaN"""
    data = np.array([[int(i) if i != "_" else np.nan for i in row] for row in data])
    return data

shortest_path_matrix = convert_to_matrix(shortest_path)
articles_path = (DATA_PATH / "articles.tsv").resolve()
assert articles_path.is_file()
articles = pd.read_csv(
    articles_path,
    sep="\t",
    header=None,
    names=["name"],
    skiprows=11,
    skip_blank_lines=True,
)
articles.name = articles.name.apply(unquote)


def compare_paths(user_path, shortest_path_df):
    """Looks up for the optimal path and compares it to the user path."""
    start, end = user_path[0], user_path[-1]
    try:
        shortest_p = shortest_path_df.loc[start, end]
        if shortest_p == 0:
            shortest_p = 1  # FIXME
        elif np.isnan(shortest_p):
            return np.nan
    except KeyError:
        return np.nan
    user_len = len(user_path)
    if np.isnan(shortest_p):
        return np.nan
    return user_len / shortest_p

def average_ratio(dataframe):
    """Calculates the average ratio for each player"""
    dataframe = dataframe.groupby("hashedIpAddress").agg(
        {
            "ratio": "mean",
            "path": "count",
            "durationInSec": "mean",
            "average_time_on_page": "mean",
        }
    )
    dataframe = dataframe.rename(columns={"path": "count"})
    dataframe = dataframe.rename(columns={"durationInSec": "mean_duration"})
    return dataframe

def player_unfinished(dataframe):
    """Calculates the number of unfinished paths and giving up time average for each player"""
    dataframe = dataframe.groupby("hashedIpAddress").agg(
        {"path": "count", "durationInSec": "mean"}
    )
    dataframe = dataframe.rename(columns={"path": "count"})
    dataframe = dataframe.rename(columns={"durationInSec": "mean_duration"})
    return dataframe

def average_ratio(dataframe):
    """Calculates the average ratio for each player"""
    dataframe = dataframe.groupby("hashedIpAddress").agg(
        {
            "ratio": "mean",
            "path": "count",
            "durationInSec": "mean",
            "average_time_on_page": "mean",
        }
    )
    dataframe = dataframe.rename(columns={"path": "count"})
    dataframe = dataframe.rename(columns={"durationInSec": "mean_duration"})
    return dataframe



def add_paths_ratio(df,feature_name="paths_ratio",quiet=False):
    """Add a feature to the dataframe that show the ratio between the user paths and the shortest path
    --------------------
    Input:
        df: dataframe with a column "path" containing the path, shotest_path and hashedIpAddress
        feature_name: name of the feature to add
        quiet: if True, do not print a warning if the dataframe already contains a column with the same name
    Return:
        the dataframe with the new feature
    """
    df = df.copy()
    if feature_name in df.columns:
        if not quiet:
            print(f"Warning: the dataframe already contains a column named {feature_name}")
    else:
        shortest_path_df = pd.DataFrame(
        shortest_path_matrix, index=articles.name, columns=articles.name
        )
        df["ratio"] = df["path"].apply(lambda x: compare_paths(x, shortest_path_df))
    return df

def add_average_time_on_page(df,feature_name="average_time_on_page",quiet=False):
    """Add a feature to the dataframe that show the average time spent on each page
    --------------------
    Input:
        df: dataframe with a column "path" containing the path and hashedIpAddress
        feature_name: name of the feature to add
        quiet: if True, do not print a warning if the dataframe already contains a column with the same name
    Return:
        the dataframe with the new feature
    """
    df = df.copy()
    if feature_name in df.columns:
        if not quiet:
            print(f"Warning: the dataframe already contains a column named {feature_name}")
    else:
        df["path_len"] = df["path"].apply(len)
        df["average_time_on_page"] = (
        df["durationInSec"] / df["path_len"]
        )
        df.drop(columns=["path_len"], inplace=True)
    return df
