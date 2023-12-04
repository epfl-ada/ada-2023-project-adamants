import pandas as pd
import numpy as np
import itertools
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from os import environ

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
from tqdm import tqdm, trange

def add_BERTscore_metric(df_path, df_edge, finished=True):
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

	global_dict = defaultdict(list)

	if finished:
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
	else:
		for i in range(len(df_cp['sucessive_pairs'])):
			for key, value in zip(df_cp['sucessive_pairs'].iloc[i], df_cp['sucessive_pairs_encoded'].iloc[i]):
				global_dict[key].append((-1, value))

		global_dict = {key: np.array(value) for key, value in global_dict.items()}
		edge_score_df = pd.DataFrame(
			{
				"edge": global_dict.keys(),
				"mean_bert_score": [np.nanmean(a[:, 1]) for a in global_dict.values()],
			}
		)

	df_cp.rename(columns={'sucessive_pairs_encoded_mean': 'BERTscore'}, inplace=True)

	return df_cp, pd.merge(df_edge, edge_score_df, on='edge', how='outer')

from torch import device
from torch.cuda import is_available

def add_sentence_similarity_metric(df_path, df_edge, finished=True):

	environ['TOKENIZERS_PARALLELISM'] = "true"
	dev = device('cuda' if is_available() else 'cpu')

	df_cp = df_path.copy()
	tokenizer = SentenceTransformer('all-mpnet-base-v2', device=dev)
	# tokenizer = SentenceTransformer('all-MiniLM-L6-v2', device=dev)


	def compute_sim(s1, s2):
		enc1 = all_terms_mapped_encodings[s1]
		enc2 = all_terms_mapped_encodings[s2]
		return util.dot_score(enc1, enc2).item()


	df_cp['sucessive_pairs'] = [
		[(x.split(";")[i], x.split(";")[i + 1]) for i in range(len(x.split(";")) - 1)]
		for x in df_cp["path"].to_list()
	]

	all_terms = list(set(itertools.chain(*df_cp['sucessive_pairs'].to_list()))) # This is a list of lists of tuples -> all pairs, for each path. We turn that into a list of pairs
	all_terms = list(set(itertools.chain(*all_terms))) # Grab this list of pairs, and turn it to unique terms
	all_encoded_terms = tokenizer.encode(all_terms, convert_to_numpy = False, convert_to_tensor = True, batch_size=64, normalize_embeddings = True, show_progress_bar = True, device = dev)
	all_terms_mapped_encodings = {all_terms[i]: all_encoded_terms[i] for i in range(len(all_terms))}

	encoded_result = [None] * len(df_cp['sucessive_pairs'])
	print('Starting loop')
	for i in trange(len(df_cp['sucessive_pairs']), desc='Computing dot products'):
		encoded_result[i] = [compute_sim(*a) for a in df_cp['sucessive_pairs'].iloc[i]]

	df_cp['sucessive_pairs_encoded'] = encoded_result

	df_cp['sucessive_pairs_encoded_mean'] = np.array(
		[np.mean(x) for x in df_cp['sucessive_pairs_encoded']]
	)  # Mean BERTscore per path
	df_cp = df_cp.loc[
		~np.isnan(df_cp['sucessive_pairs_encoded_mean'])
	]  # Remove NaNs

	global_dict = defaultdict(list)

	if finished:
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
	else:
		for i in range(len(df_cp['sucessive_pairs'])):
			for key, value in zip(df_cp['sucessive_pairs'].iloc[i], df_cp['sucessive_pairs_encoded'].iloc[i]):
				global_dict[key].append((-1, value))

		global_dict = {key: np.array(value) for key, value in global_dict.items()}
		edge_score_df = pd.DataFrame(
			{
				"edge": global_dict.keys(),
				"mean_bert_score": [np.nanmean(a[:, 1]) for a in global_dict.values()],
			}
		)

	df_cp.rename(columns={'sucessive_pairs_encoded_mean': 'BERTscore'}, inplace=True)

	return df_cp, pd.merge(df_edge, edge_score_df, on='edge', how='outer')