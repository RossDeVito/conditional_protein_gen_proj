"""
Load model and use to generate proteins with the attribute tags of the
test set.

All models (e.g. autoregressive LM, DPMM) will output generated proteins
in the same format so a common evaluation script can compare them head to head.
"""

import argparse
import os
import json
import platform

import pandas as pd
import torch
import pytorch_lightning as pl
from tqdm import tqdm

from protein_gen.data_modules import (
	ProteinDataModule, AutoRegressiveLMCollationFn
)
from protein_gen.autoregressive_LM import ARLM
from protein_gen.data_modules import AMINO_ACID_SYM_TO_IDX, AMINO_ACID_IDX_TO_SYM


MODEL_PATHS = {
	'arlm_small': os.path.join(
		'training_output', 'arlm_small', 'version_13',
		'checkpoints', 'epoch=133-best_val_loss.ckpt'
	),
}

SAMPLING_PARAMS = {
	'top5': {
		'top_k': 5,'temperature': 1.0
	},
	'top10': {
		'top_k': 10,'temperature': 1.0
	},
}


if __name__ == '__main__':
	__spec__ = None

	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--model_name', type=str, required=True)
	parser.add_argument('-p', '--sampling_params', type=str, required=True)
	parser.add_argument('-s', '--use_subset', action='store_true')
	parser.add_argument('-b', '--batch_size', type=int, default=32)

	args = parser.parse_args()

	# Load Pytorch Lightning checkpoint
	model = ARLM.load_from_checkpoint(MODEL_PATHS[args.model_name])
	model.eval()
	training_args = model.config['training_config'].copy()

	if torch.backends.mps.is_available():
		mps_device = torch.device("mps")
		model.to(mps_device)

	# Set test data path
	if args.use_subset:
		training_args['data_mod_kwargs']['test_data_fname'] = 'subset_test_data.csv'

	# Set batch size
	training_args['data_mod_kwargs']['batch_size'] = args.batch_size

	# Create data module
	data_module = ProteinDataModule(
		collate_fn=AutoRegressiveLMCollationFn,
		**training_args['data_mod_kwargs']
	)
	data_module.setup(stage='test')
	test_dataloader = data_module.test_dataloader()

	# Generate proteins for each sample in the test set
	sample_kwargs = SAMPLING_PARAMS[args.sampling_params]

	gen_protein_strings = []
	orig_protein_strings = []
	orig_protein_tags = []

	for b_idx, batch in enumerate(tqdm(test_dataloader)):
		# Generate proteins
		gen_tokens, gen_strings = model.sample(batch, **sample_kwargs)

		gen_protein_strings.extend(gen_strings)

		# Get original protein strings and tags
		x_seq = batch['x_seq'].cpu().numpy()
		seq_mask = batch['seq_mask'].cpu().numpy()

		for i in range(x_seq.shape[0]):
			orig_protein_strings.append(
				''.join(
					[
						AMINO_ACID_IDX_TO_SYM[idx]
						for idx in x_seq[i, 1:seq_mask[i].sum()]
					]
				)
			)

		orig_protein_tags.extend(batch['conditioning_tags'].detach().tolist())
		
		# if b_idx > 5:
		# 	break

	# Save generated proteins
	# Save dir concats model name, sampling params, and if subset is used
	save_dir = os.path.join(
		'saved_output',
		'subset' if args.use_subset else 'full',
		args.model_name,
		args.sampling_params,
	)

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# Make DateFrame and save as csv
	df = pd.DataFrame(
		{
			'orig_protein_string': orig_protein_strings,
			'orig_protein_tags': orig_protein_tags,
			'gen_protein_string': gen_protein_strings
		}
	)
	df.to_csv(os.path.join(save_dir, 'generated_proteins.csv'), index=False)



	

		
		
