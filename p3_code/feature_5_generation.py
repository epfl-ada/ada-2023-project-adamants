from urllib.parse import unquote
import pandas as pd
import numpy as np
from pathlib import Path
from feature_26_generation import remove_backclicks_and_split
from tqdm import tqdm


def compare_paths(user_path, shortest_path_distance_matrix,articles):
    """Looks up for the optimal path and compares it to the user path."""
    start, end = user_path[0], user_path[-1]
    opt = 0 
    start_index = articles[articles["article name"] == start].index[0]
    end_index = articles[articles["article name"] == end].index[0]
    if shortest_path_distance_matrix[start_index][end_index] == "_":
        opt = -1  
    else:
        opt = shortest_path_distance_matrix[start_index][end_index]
    return opt


# def average_ratio(dataframe):
#     """Calculates the average ratio for each player"""
#     dataframe = dataframe.groupby("hashedIpAddress").agg(
#         {
#             "ratio": "mean",
#             "path": "count",
#             "durationInSec": "mean",
#             "average_time_on_page": "mean",
#         }
#     )
#     dataframe = dataframe.rename(columns={"path": "count"})
#     dataframe = dataframe.rename(columns={"durationInSec": "mean_duration"})
#     return dataframe

# def player_unfinished(dataframe):
#     """Calculates the number of unfinished paths and giving up time average for each player"""
#     dataframe = dataframe.groupby("hashedIpAddress").agg(
#         {"path": "count", "durationInSec": "mean"}
#     )
#     dataframe = dataframe.rename(columns={"path": "count"})
#     dataframe = dataframe.rename(columns={"durationInSec": "mean_duration"})
#     return dataframe

# def average_ratio(dataframe):
#     """Calculates the average ratio for each player"""
#     dataframe = dataframe.groupby("hashedIpAddress").agg(
#         {
#             "ratio": "mean",
#             "path": "count",
#             "durationInSec": "mean",
#             "average_time_on_page": "mean",
#         }
#     )
#     dataframe = dataframe.rename(columns={"path": "count"})
#     dataframe = dataframe.rename(columns={"durationInSec": "mean_duration"})
#     return dataframe



def add_paths_ratio(df,short_matrix,articles,feature_name="paths_ratio",quiet=False,finished=True):
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
        if finished == False:
            df = df[df["target"].isin(articles["article name"])]
        # Removing back clicks (<) and splitting paths
        df = remove_backclicks_and_split(df).copy(deep=True)   
        df["ratio"] = df["path"].apply(lambda x: compare_paths(x, short_matrix,articles))
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
