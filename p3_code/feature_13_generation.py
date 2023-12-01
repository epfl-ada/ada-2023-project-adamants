import pandas as pd
import warnings

def add_number_of_backtracks(df,feature_name="backtrack",quiet=False):
    """Add a feature to the dataframe that counts the number of backtracks in the path
    --------------------
    Input:
        df: dataframe with a column "path" containing the path
        feature_name: name of the feature to add
        quiet: if True, do not print a warning if the dataframe already contains a column with the same name
    Return:
        the dataframe with the new feature
    """
    df= df.copy(deep=True)
    if not "path" in df.columns:
        raise ValueError("The dataframe needs to contain the full path in the column 'path'")
    if feature_name in df.columns and not quiet:
        warnings.warn("The dataframe already contains a column named "+feature_name)
    df[feature_name] = df["path"].str.count("<")
    return df

def add_number_of_paths_previously_played(df1,df2,feature_name="numberOfPath",quiet=False):
    """Add a feature to the dataframe that counts the number of paths previously played
    --------------------
    Input:
        df1: dataframe with a column "path" containing the path
        df2: dataframe with a column "path" containing the path
        feature_name: name of the feature to add
        quiet: if True, do not print a warning if the dataframe already contains a column with the same name
    Return:
        the dataframes with the new feature
    """
    df1= df1.copy(deep=True)
    df2= df2.copy(deep=True)
    if feature_name in df1.columns and not quiet:
        warnings.warn("The df1 already contains a column named "+feature_name)
    if feature_name in df2.columns and not quiet:
        warnings.warn("The df2 already contains a column named "+feature_name)
    #add a unique identifier to each dataframe
    df1["id"]=df1.index.astype(str)+"_df1"
    df2["id"]=df2.index.astype(str)+"_df2"
    #merge the two dataframes
    df_merged = pd.concat([df1, df2], ignore_index=True)
    df_merged[feature_name] = (
        df_merged.sort_values(by=["timestamp"]).groupby("hashedIpAddress").cumcount() + 1
    )
    #split the dataframe
    df1 = df1.merge(df_merged[["id", feature_name]], on="id", how="left")
    df2 = df2.merge(df_merged[["id", feature_name]], on="id", how="left")
    #drop the id column
    df1.drop(columns=["id"], inplace=True)
    df2.drop(columns=["id"], inplace=True)

    return df1,df2