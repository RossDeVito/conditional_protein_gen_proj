""" Generate protein embeddings using pretrained model and save to csv. """

import argparse
import os
import platform

import numpy as np
import pandas as pd
from tqdm import trange

import torch
from transformers import T5Tokenizer, T5EncoderModel


MODEL_NAME_TO_OUTPUT_PATH = {
	'arlm_small_subset': os.path.join(
		'..', 'autoregressive_LM', 'saved_output', 'subset', 
		'arlm_small', 'top10', 'generated_proteins.csv'
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

	args = parser.parse_args()

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

	for i in trange(0, len(protein_seqs), args.batch_size):
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

	# Save embeddings with numpy
	if args.do_ground_true:
		save_path = os.path.join(
			'embeddings', args.embedding_type, 'ground_true'
		)
	else:
		save_path = os.path.join(
			'embeddings', args.embedding_type, args.model_name
		)
	if not os.path.exists(os.path.dirname(save_path)):
		os.makedirs(os.path.dirname(save_path))

	np.save(os.path.join(save_path, 'prot_mean_embs.npy'), prot_mean_embs)
	np.save(os.path.join(save_path, 'prot_masked_mean_embs.npy'), prot_masked_mean_embs)
