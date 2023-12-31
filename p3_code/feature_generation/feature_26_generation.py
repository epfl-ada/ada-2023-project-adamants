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

def load(name, drop_timeouts=False):
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
        if name == 'paths_unfinished' and drop_timeouts == True:
            df = df[not "timeout" in df["type"] and df["path"].str.count(";") > 0]
            df.reset_index(inplace=True)
    
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
        j = 1
        while (j < len(paths["path"].iloc[i])):
            if paths["path"].iloc[i][j] == "<":
                del paths["path"].iloc[i][j]
                del paths["path"].iloc[i][j-1]
                j = j-2
            j += 1
    return paths    


def path_to_html(article_name):
    """Returns path to article's html file"""
    article_name_undsc = article_name.replace(" ", "_")
    ALL_HTML = "../data/wikispeedia_articles_html/wpcd/wp/"
    path = ALL_HTML + article_name[0].lower() + '/' + article_name_undsc + ".htm"
    return path


def add_25_behind_percentage(input_string):
    """Resolves html bug by adding 25 behind % in html links from html files"""
    modified_string = ""
    for char in input_string:
        if char == '%':
            modified_string += '%' + '25'
        else:
            modified_string += char
    modified_string = modified_string.replace(" ", "_")
    return modified_string


def find_word_position_html(successive_pair):
    """Function returning character count before clicked link of a sucessive pair
    
    that is, position of second word in seccessive pair in first word in pair's article"""
    target_words = successive_pair[1].replace("_", " ")
    article = path_to_html(successive_pair[0])

    with open(article, 'r', encoding='utf-8') as html_file:
        content = html_file.read()
        try:
            number_of_characters = len(content)
            return content.index(target_words)/number_of_characters

        except:  # For target words containing %
            new_target_words = add_25_behind_percentage(target_words)
            number_of_characters = len(content)
            return content.index(new_target_words)/number_of_characters
            


def compute_position_for_all_pairs(successive_pairs, max_path_length):
    """Applies find_word_position whole dataframe"""
    next_link_position = np.zeros(
        (len(successive_pairs), max_path_length - 1)
    )
    for i in tqdm(range(len(successive_pairs)), file=sys.stdout):
        for j in range(len(successive_pairs[i])):
            next_link_position[i, j] = find_word_position_html(successive_pairs[i][j])

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

