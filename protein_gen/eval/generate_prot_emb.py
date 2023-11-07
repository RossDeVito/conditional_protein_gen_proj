""" Generate protein embeddings using pretrained model and save to csv. """

import argparse
import os
import platform

import numpy as np
import pandas as pd
import tqdm
from tqdm import trange

import torch
from transformers import T5Tokenizer, T5EncoderModel


MODEL_NAME_TO_OUTPUT_PATH = {
	'arlm_small_top5': os.path.join(
		'..', 'autoregressive_LM', 'saved_output', 'full', 
		'arlm_small', 'top5', 'generated_proteins.csv'
	),
	'arlm_small_top10': os.path.join(
		'..', 'autoregressive_LM', 'saved_output', 'full', 
		'arlm_small', 'top10', 'generated_proteins.csv'
	),
	'arlm_small_top10_t1_5': os.path.join(
		'..', 'autoregressive_LM', 'saved_output', 'full', 
		'arlm_small', 'top10_t1_5', 'generated_proteins.csv'
	),
	'arlm_large_top5': os.path.join(
		'..', 'autoregressive_LM', 'saved_output', 'full',
		'arlm_large', 'top5', 'generated_proteins.csv'
	),
	'arlm_large_top5_t1_5': os.path.join(
		'..', 'autoregressive_LM', 'saved_output', 'full',
		'arlm_large', 'top5_t1_5', 'generated_proteins.csv'
	),
	'arlm_large_top10': os.path.join(
		'..', 'autoregressive_LM', 'saved_output', 'full',
		'arlm_large', 'top10', 'generated_proteins.csv'
	),
	'arlm_large_top10_tdo': os.path.join(
		'..', 'autoregressive_LM', 'saved_output', 'full',
		'arlm_large', 'top10_tdo', 'generated_proteins.csv'
	),
	'arlm_large_top10_v0': os.path.join(
		'..', 'autoregressive_LM', 'saved_output', 'full',
		'arlm_large', 'top10_v0', 'generated_proteins.csv'
	),
	'arlm_large_top10_uncond': os.path.join(
		'..', 'autoregressive_LM', 'saved_output', 'full',
		'arlm_large', 'top10_uncond', 'generated_proteins.csv'
	),
	'arlm_large_top10_t0_9': os.path.join(
		'..', 'autoregressive_LM', 'saved_output', 'full',
		'arlm_large', 'top10_t0_9', 'generated_proteins.csv'
	),
	'arlm_large_top10_t1_1': os.path.join(
		'..', 'autoregressive_LM', 'saved_output', 'full',
		'arlm_large', 'top10_t1_1', 'generated_proteins.csv'
	),
	'arlm_large_top15': os.path.join(
		'..', 'autoregressive_LM', 'saved_output', 'full',
		'arlm_large', 'top15', 'generated_proteins.csv'
	),
	'arlm_large_top15_t1_2': os.path.join(
		'..', 'autoregressive_LM', 'saved_output', 'full',
		'arlm_large', 'top15_t1_2', 'generated_proteins.csv'
	),
	'arlm_large_top15_t1_5': os.path.join(
		'..', 'autoregressive_LM', 'saved_output', 'full',
		'arlm_large', 'top15_t1_5', 'generated_proteins.csv'
	),
	'arlm_small_subset': os.path.join(
		'..', 'autoregressive_LM', 'saved_output', 'subset', 
		'arlm_small', 'top10', 'generated_proteins.csv'
	),
	'rand_10_subset_0': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'subset',
		'percent_sub_0.1', 'rand_sub_replicate_0.csv'
	),
	'rand_10_subset_1': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'subset',
		'percent_sub_0.1', 'rand_sub_replicate_1.csv'
	),
	'rand_10_subset_2': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'subset',
		'percent_sub_0.1', 'rand_sub_replicate_2.csv'
	),
	'rand_10_subset_3': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'subset',
		'percent_sub_0.1', 'rand_sub_replicate_3.csv'
	),
	'rand_10_subset_4': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'subset',
		'percent_sub_0.1', 'rand_sub_replicate_4.csv'
	),
	'rand_50_subset_0': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'subset',
		'percent_sub_0.5', 'rand_sub_replicate_0.csv'
	),
	'rand_50_subset_1': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'subset',
		'percent_sub_0.5', 'rand_sub_replicate_1.csv'
	),
	'rand_50_subset_2': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'subset',
		'percent_sub_0.5', 'rand_sub_replicate_2.csv'
	),
	'rand_50_subset_3': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'subset',
		'percent_sub_0.5', 'rand_sub_replicate_3.csv'
	),
	'rand_50_subset_4': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'subset',
		'percent_sub_0.5', 'rand_sub_replicate_4.csv'
	),
	# Full random replaces
	'rand_10_0': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'full',
		'percent_sub_0.1', 'rand_sub_replicate_0.csv'
	),
	'rand_10_1': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'full',
		'percent_sub_0.1', 'rand_sub_replicate_1.csv'
	),
	'rand_10_2': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'full',
		'percent_sub_0.1', 'rand_sub_replicate_2.csv'
	),
	'rand_10_3': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'full',
		'percent_sub_0.1', 'rand_sub_replicate_3.csv'
	),
	'rand_10_4': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'full',
		'percent_sub_0.1', 'rand_sub_replicate_4.csv'
	),
	'rand_10_5': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'full',
		'percent_sub_0.1', 'rand_sub_replicate_5.csv'
	),
	'rand_25_0': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'full',
		'percent_sub_0.25', 'rand_sub_replicate_0.csv'
	),
	'rand_25_1': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'full',
		'percent_sub_0.25', 'rand_sub_replicate_1.csv'
	),
	'rand_25_2': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'full',
		'percent_sub_0.25', 'rand_sub_replicate_2.csv'
	),
	'rand_25_3': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'full',
		'percent_sub_0.25', 'rand_sub_replicate_3.csv'
	),
	'rand_25_4': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'full',
		'percent_sub_0.25', 'rand_sub_replicate_4.csv'
	),
	'rand_25_5': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'full',
		'percent_sub_0.25', 'rand_sub_replicate_5.csv'
	),
	'rand_50_0': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'full',
		'percent_sub_0.5', 'rand_sub_replicate_0.csv'
	),
	'rand_50_1': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'full',
		'percent_sub_0.5', 'rand_sub_replicate_1.csv'
	),
	'rand_50_2': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'full',
		'percent_sub_0.5', 'rand_sub_replicate_2.csv'
	),
	'rand_50_3': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'full',
		'percent_sub_0.5', 'rand_sub_replicate_3.csv'
	),
	'rand_50_4': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'full',
		'percent_sub_0.5', 'rand_sub_replicate_4.csv'
	),
	'rand_50_5': os.path.join(
		'..', 'random_sub', 'saved_random_subs', 'full',
		'percent_sub_0.5', 'rand_sub_replicate_4.csv'
	),

	# D3PM
	'd3pm_small_s20_top1': os.path.join(
		'..', 'd3pm', 'saved_output', 'full',
		'd3pm_small', 's20_top1', 'generated_proteins.csv'
	),
	'd3pm_large_s20_top1': os.path.join(
		'..', 'd3pm', 'saved_output', 'full',
		'd3pm_large', 's20_top1', 'generated_proteins.csv'
	),
	'd3pm_large_s100_top1': os.path.join(
		'..', 'd3pm', 'saved_output', 'full',
		'd3pm_large', 's100_top1', 'generated_proteins.csv'
	),
	'd3pm_large_s20_top5': os.path.join(
		'..', 'd3pm', 'saved_output', 'full',
		'd3pm_large', 's20_top5', 'generated_proteins.csv'
	),
	'd3pm_large_s100_top5': os.path.join(
		'..', 'd3pm', 'saved_output', 'full',
		'd3pm_large', 's100_top5', 'generated_proteins.csv'
	),
	'd3pm_large_s20_top10': os.path.join(
		'..', 'd3pm', 'saved_output', 'full',
		'd3pm_large', 's20_top10', 'generated_proteins.csv'
	),

	# version b
	'd3pm_large_b_s100_top1': os.path.join(
		'..', 'd3pm', 'saved_output', 'full',
		'd3pm_large_b', 's100_top1', 'generated_proteins.csv'
	),
	'd3pm_large_b_s100_top5': os.path.join(
		'..', 'd3pm', 'saved_output', 'full',
		'd3pm_large_b', 's100_top5', 'generated_proteins.csv'
	),

}

EMB_TYPE_TO_MODEL_PARAMS = {
	'prot_t5_xl_half': {
		'tokenizer_kwargs': {
			'pretrained_model_name_or_path': 'Rostlab/prot_t5_xl_half_uniref50-enc',
			'do_lower_case': False,
		},
		'model_kwargs': {
			'pretrained_model_name_or_path': 'Rostlab/prot_t5_xl_half_uniref50-enc',
			'torch_dtype': torch.float16
		},
	},	
}
	


if __name__ == '__main__':
	__spec__ = None

	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model_name', type=str, required=True)
	parser.add_argument('-e', '--embedding_type', type=str, required=True)
	parser.add_argument('-b', '--batch_size', type=int, default=32)
	parser.add_argument('-t', '--do_ground_true', action='store_true')
	parser.add_argument('-n', '--num_samples', type=int, default=-1)

	args = parser.parse_args()
	print(args)

	# Load generated proteins
	generated_proteins = pd.read_csv(MODEL_NAME_TO_OUTPUT_PATH[args.model_name])

	# Load pretrained model and tokenizer for embedding generation
	tokenizer = T5Tokenizer.from_pretrained(
		**EMB_TYPE_TO_MODEL_PARAMS[args.embedding_type]['tokenizer_kwargs']
	)
	model = T5EncoderModel.from_pretrained(
		**EMB_TYPE_TO_MODEL_PARAMS[args.embedding_type]['model_kwargs']
	)

	if torch.backends.mps.is_available():
		device = torch.device("cpu")
		model.to(device)
	elif torch.cuda.is_available():
		device = torch.device("cuda")
		model.to(device)
	else:
		device = torch.device("cpu")

	model.eval().to(device)

	if device.type == 'cpu':
		print('Warning: Using CPU for embedding generation. This may take a long time.')
		model = model.float()

	# Format protein sequences for embedding generation
	if args.do_ground_true:
		protein_seqs = generated_proteins['orig_protein_string']
	else:
		protein_seqs = generated_proteins['gen_protein_string']

	# Add space between each amino acid
	protein_seqs = protein_seqs.apply(lambda x: ' '.join(x))

	# Embed protein sequences
	prot_mean_embs = []
	prot_masked_mean_embs = []

	tqdm_batches = trange(0, len(protein_seqs), args.batch_size)

	for i in tqdm_batches:
		tqdm_batches.write(str(i))
		tqdm_batches.refresh()
		seqs = protein_seqs[i:i+args.batch_size]
		tokens = tokenizer.batch_encode_plus(
			seqs, add_special_tokens=True, padding="longest"
		)

		input_ids = torch.tensor(tokens['input_ids']).to(device)
		attention_mask = torch.tensor(tokens['attention_mask']).to(device)

		with torch.no_grad():
			embedding_rpr = model(
				input_ids=input_ids, attention_mask=attention_mask
			)

		prot_mean_embs.extend(embedding_rpr.last_hidden_state.mean(1).tolist())

		# Mask out padding tokens
		masked_emb = embedding_rpr.last_hidden_state * attention_mask.unsqueeze(-1)
		prot_masked_mean_embs.extend(
			(masked_emb.sum(1) / attention_mask.sum(1).unsqueeze(-1)).tolist()
		)

		if args.num_samples > 0 and len(prot_masked_mean_embs) > args.num_samples:
			break

	# Save embeddings with numpy
	if args.do_ground_true:
		save_path = os.path.join(
			'embeddings', args.embedding_type, 'ground_true'
		)
	else:
		save_path = os.path.join(
			'embeddings', args.embedding_type, args.model_name
		)

	if not os.path.exists(save_path):
		os.makedirs(save_path)

	np.save(os.path.join(save_path, 'prot_mean_embs.npy'), prot_mean_embs)
	np.save(os.path.join(save_path, 'prot_masked_mean_embs.npy'), prot_masked_mean_embs)
