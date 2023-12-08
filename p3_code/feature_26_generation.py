import pandas as pd
import numpy as np
from tqdm import tqdm
import sys


DATA_FOLDER = "../data/"
PATHS_AND_GRAPH = DATA_FOLDER + "wikispeedia_paths-and-graph/"
PATHS_FINISHED = PATHS_AND_GRAPH + "paths_finished.tsv"
PATHS_UNFINISHED = PATHS_AND_GRAPH + "paths_unfinished.tsv"
SHORTEST_PATH_MATRIX = PATHS_AND_GRAPH + "shortest-path-distance-matrix.txt"
ARTICLES = PATHS_AND_GRAPH + "articles.tsv"

def load(name):
    """Loading different datasets""" 
    if name == 'paths_finished':
        names=["hashedIpAddress", "timestamp", "durationInSec", "path", "rating"]
        rows=16
        PATH = PATHS_FINISHED
    if name == 'paths_unfinished':
        names=["hashedIpAddress", "timestamp", "durationInSec", "path", "target", "type"]
        rows=17
        PATH = PATHS_UNFINISHED
    if name == 'shortest_path_matrix':
        names=["shortest path"]
        rows=17
        PATH = SHORTEST_PATH_MATRIX
    if name == 'articles':
        names=["article name"]
        rows=12,
        PATH = ARTICLES

    df = pd.read_csv(
        PATH,
        sep="\t",
        header=None,
        names=names,
        encoding="utf-8",
        skiprows=rows,
    ).copy(deep=True)

    if (name == 'paths_unfinished' or name == 'paths_finished'):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    
    # for shortest path matrix: splitting string into list
    if name == 'shortest_path_matrix':
        df = df["shortest path"].apply(lambda x: list(x))  
        
    return df


def load_articles():
    """Loading articles dataset"""
    articles = pd.read_csv(
        ARTICLES,
        sep="\t",
        header=None,
        encoding="utf-8",
        names=["article name"],
        skiprows=12,
    ).copy(deep=True)
    return articles


def add_link_position(df, finished):
    """Adds the next link position to path pairs as extra columns.
    
    Makes a deep copy of the data, the modified copy is returned."""
    pd.options.mode.chained_assignment = None  # to remove warning

    # Removing back clicks (<) and splitting paths
    df = remove_backclicks_and_split(df).copy(deep=True)
    pairs = successive_pairs(df)

    # Computing maximum path length
    path_length = df["path"].apply(lambda x: len(x))
    max_path_length = path_length.max()

    position_mean, position_std = compute_position_for_all_pairs(pairs, max_path_length)

    # add new features to dataframe
    position_mean = position_mean.reshape(len(position_mean),)
    df['position_mean'] = pd.Series(position_mean)
    position_std = position_std.reshape(len(position_std),)
    df['position_std'] = pd.Series(position_std)

    
    # return pd.Series(next_link_position_series), average_position
    return df
   

def successive_pairs(paths):
    """Grouping successive pairs from paths"""
    pairs = [
        [(x[i], x[i + 1]) for i in range(len(x) - 1)]
        for x in paths["path"].to_list()
    ]
    return pairs


def remove_backclicks_and_split(paths):
    """Removing back clicks (<) and splitting paths (;)"""
    for i in tqdm(range(len(paths)), file=sys.stdout):
        paths["path"].iloc[i] = paths["path"].iloc[i].split(";")
        for item in paths["path"].iloc[i].copy():
            if item == "<":
                paths["path"].iloc[i].remove(item)
    return paths    


def path_to_plaintext(article_name):
    """Returns path to plaintext article"""
    article_name_undsc = article_name.replace(" ", "_")
    ALL_PLAINTEXT = "../data/plaintext_articles/"
    path = ALL_PLAINTEXT + article_name_undsc + ".txt"
    return path


def find_word_position(successive_pair):
    """Function returning character count before clicked link of a sucessive pair
    
    that is, position of second word in seccessive pair in first word in pair's article"""
    target_words = successive_pair[1].replace("_", " ")
    article = path_to_plaintext(successive_pair[0])

    with open(article, encoding="utf8") as file:
        content = file.read()
        try:
            number_of_characters = len(content)
            return content.index(target_words)/number_of_characters
        except:  # Mistake because sometimes word in is html file but not in text file. 
                # To be treated later. Right now we don't consider these datapoints
            # print(f"The group of words '{successive_pair[1]}' was not found in the file '{successive_pair[0]}'.")
            return 0


def compute_position_for_all_pairs(successive_pairs, max_path_length):
    """Applies find_word_position wholes dataframe"""
    next_link_position = np.zeros(
        (len(successive_pairs), max_path_length - 1)
    )
    for i in tqdm(range(len(successive_pairs)), file=sys.stdout):
        for j in range(len(successive_pairs[i])):
            next_link_position[i, j] = find_word_position(successive_pairs[i][j])

    # make it into a series
    #position_series = []
    #for i in tqdm(range(len(next_link_position)), file=sys.stdout):
        #position_series = np.append(position_series,next_link_position[i,np.nonzero(next_link_position[i])][0])

    # or compute mean position for each path
    position_mean = np.zeros((len(next_link_position), 1))
    position_std = np.zeros((len(next_link_position), 1))
    for i in tqdm(range(len(next_link_position)), file=sys.stdout):
        position_mean[i] = next_link_position[i,np.nonzero(next_link_position[i])].mean()
        position_std[i] = next_link_position[i,np.nonzero(next_link_position[i])].std()

    return position_mean, position_std




def add_path_length(df, articles, shortest_path_distance_matrix, finished, feature_name="path_length"):
    """Adds the path_length and optimal_path_length to paths as extra columns.
    
    Makes a deep copy of the data, the modified copy is returned."""
    # Delete datapoints from 'paths_unfinished_copy' if 'target' isn't part of article list from 'articles' dataframe
    if finished == False:
        df = df[df["target"].isin(articles["article name"])]

    # Removing back clicks (<) and splitting paths
    df = remove_backclicks_and_split(df).copy(deep=True)                    
    
    # Comparing optimal path length between finished and unfinished
    optimal_path_lengths = optimal_path_length(df, finished, articles, shortest_path_distance_matrix)  # change to only take unique path datapoints ??
    print("Mean optimal path length:",  "%.2f" % optimal_path_lengths.mean())

    # add path_length feature to dataframe
    df[feature_name] = [len(x) for x in df['path']]
    df['optimal_path_length'] = optimal_path_lengths
    
    return df, optimal_path_lengths.mean()


def optimal_path_length(paths, finished, articles, shortest_path_distance_matrix):
    """Function computing optimal path length for each path"""
    opt = np.zeros((len(paths), 1))
    for i in tqdm(range(len(opt)), file=sys.stdout):
        if finished == True:
            start_article = paths["path"][i][0]
            end_article = paths["path"][i][-1]
        else:
            start_article = paths["path"].iloc[i][0]
            end_article = paths["target"].iloc[i]
        start_index = articles[articles["article name"] == start_article].index[0]
        end_index = articles[articles["article name"] == end_article].index[0]
        if shortest_path_distance_matrix[start_index][end_index] == "_":
            opt[i] = -1  
        else:
            opt[i] = shortest_path_distance_matrix[start_index][end_index]
    return opt

