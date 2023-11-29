import pandas as pd
import numpy as np
import itertools

def add_time_per_edge(df):
	df_cp = df.copy()
	try:
		df_cp["path_length"] = df_cp["path"].apply(
		lambda x: len(x.split(";"))
		)
	except AttributeError: # In case the data has already been split into a list
		df_cp["path_length"] = df_cp["path"].apply(
		lambda x: len(x)
		)
	df_cp["coarse_mean_time"] = (
		df_cp["durationInSec"] / df_cp["path_length"]
	)

	return df_cp

def split_into_edges(df):
	return pd.DataFrame({'edge': list(
		set(itertools.chain(*[
			[(x.split(";")[i], x.split(";")[i + 1]) for i in range(len(x.split(";")) - 1)]
			for x in df["path"].to_list()])
		)
	)})


from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def add_BERTscore_metric(df_path, df_edge):
	df_cp = df_path.copy()
	tokenizer = AutoTokenizer.from_pretrained(
    	"dslim/bert-base-NER"
	)  # Named entity recognition-specific model!


	def compute_sim(s1, s2):
		enc1 = tokenizer.encode(s1, return_tensors="pt").reshape(1, -1)
		enc2 = tokenizer.encode(s2, return_tensors="pt").reshape(1, -1)

		trunc = min(enc1.size(dim=1), enc2.size(dim=1))

		enc1 = enc1[:, :trunc]
		enc2 = enc2[:, :trunc]

		return cosine_similarity(enc1, enc2).squeeze().item()


	df_cp['sucessive_pairs'] = [
		[(x.split(";")[i], x.split(";")[i + 1]) for i in range(len(x.split(";")) - 1)]
		for x in df_cp["path"].to_list()
	]
	df_cp['sucessive_pairs_encoded'] = [[compute_sim(*a) for a in x] for x in tqdm(df_cp['sucessive_pairs'])]

	df_cp['sucessive_pairs_encoded_mean'] = np.array(
		[np.mean(x) for x in df_cp['sucessive_pairs_encoded']]
	)  # Mean BERTscore per path
	df_cp = df_cp.loc[
		~np.isnan(df_cp['sucessive_pairs_encoded_mean'])
	]  # Remove NaNs

	from collections import defaultdict

	global_dict = defaultdict(list)

	for i in range(len(df_cp['sucessive_pairs'])):
		local_rating = df_cp["rating"].iloc[i]
		for key, value in zip(df_cp['sucessive_pairs'].iloc[i], df_cp['sucessive_pairs_encoded'].iloc[i]):
			global_dict[key].append((local_rating, value))

	global_dict = {key: np.array(value) for key, value in global_dict.items()}
	edge_score_df = pd.DataFrame(
		{
			"edge": global_dict.keys(),
			"mean_bert_score": [np.nanmean(a[:, 1]) for a in global_dict.values()],
			"mean_rating": [np.nanmean(a[:, 0]) for a in global_dict.values()],
		}
	)

	df_cp.rename(columns={'successive_pairs_mean': 'BERTscore'}, inplace=True)

	return df_cp, pd.merge(df_edge, edge_score_df, on='edge', how='outer')