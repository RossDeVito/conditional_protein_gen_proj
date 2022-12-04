"""
Load model and use to generate proteins with the attribute tags of the
test set.

All models (e.g. autoregressive LM, DPMM) will output generated proteins
in the same format so a common evaluation script can compare them head to head.
"""

import argparse
import os
import platform

import pandas as pd
import torch
import pytorch_lightning as pl
from tqdm import tqdm

from protein_gen.data_modules import (
	ProteinDataModule, D3PMCollationFn
)
from protein_gen.d3pm import D3PM
from protein_gen.data_modules import AMINO_ACID_SYM_TO_IDX, AMINO_ACID_IDX_TO_SYM


MODEL_PATHS = {
	'd3pm_small': os.path.join(
		'training_output', 'small', 'version_3',
		'checkpoints', 'epoch=87-best_val_loss.ckpt'
	),
	'd3pm_large': os.path.join(
		'training_output', 'large', 'version_2',
		'checkpoints', 'epoch=48-best_val_loss.ckpt'
	),
}

SAMPLING_PARAMS = {
	's20_top1': {
		'n_diffusion_steps': 20 , 'top_k': 1
	},
	's100_top1': {
		'n_diffusion_steps': 100 , 'top_k': 1
	},
	's20_top5': {
		'n_diffusion_steps': 20 , 'top_k': 5
	},
	's100_top5': {
		'n_diffusion_steps': 100 , 'top_k': 5
	},
	's20_top10': {
		'n_diffusion_steps': 20 , 'top_k': 10
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
	parser.add_argument('-u', '--uncondioned', action='store_true')
	parser.add_argument('-m', '--min_num_samples', type=int, default=-1)

	args = parser.parse_args()

	print(args)

	sample_kwargs = SAMPLING_PARAMS[args.sampling_params]

	# Load Pytorch Lightning checkpoint
	model = D3PM.load_from_checkpoint(MODEL_PATHS[args.model_name])
	model.eval()
	training_args = model.config['training_config'].copy()

	if torch.backends.mps.is_available():
		mps_device = torch.device("mps")
		model.to(mps_device)
	elif torch.cuda.is_available():
		gpu_device = torch.device("cuda")
		model.to(gpu_device)

	# Set test data path
	if args.use_subset:
		training_args['data_mod_kwargs']['test_data_fname'] = 'subset_test_data.csv'

	# Set batch size
	training_args['data_mod_kwargs']['batch_size'] = args.batch_size

	if args.uncondioned:
		training_args['data_mod_kwargs']['taxon_dropout_rate'] = 1.0
		training_args['data_mod_kwargs']['attribute_dropout_rate'] = 1.0
	elif 'tdo' in sample_kwargs.keys():
		tdo = sample_kwargs.pop('tdo')
		training_args['data_mod_kwargs']['taxon_dropout_rate'] = tdo
		training_args['data_mod_kwargs']['attribute_dropout_rate'] = tdo
	else:
		training_args['data_mod_kwargs']['taxon_dropout_rate'] = 0.0
		training_args['data_mod_kwargs']['attribute_dropout_rate'] = 0.0

	# Create data module
	data_module = ProteinDataModule(
		collate_fn=D3PMCollationFn,
		**training_args['data_mod_kwargs']
	)
	data_module.setup(stage='test')
	test_dataloader = data_module.test_dataloader()

	# Generate proteins for each sample in the test set
	gen_protein_strings = []
	orig_protein_strings = []
	orig_protein_tags = []

	tqdm_batches = tqdm(test_dataloader)

	for b_idx, batch in enumerate(tqdm_batches):
		tqdm_batches.refresh()
		tqdm_batches.write(str(b_idx))
		# Move all values in batch to device
		batch = {k: v.to(model.device) for k, v in batch.items()}

		# Generate proteins
		gen_tokens, gen_strings = model.sample(
			batch, max_length=batch['sequence'].shape[-1], **sample_kwargs
		)

		gen_protein_strings.extend(gen_strings)

		# Get original protein strings and tags
		x_seq = batch['sequence'].cpu().numpy()

		for i in range(x_seq.shape[0]):
			orig_protein_strings.append(
				''.join(
					[
						AMINO_ACID_IDX_TO_SYM[idx]
						for idx in x_seq[i] if idx not in (22, 0)
					]
				)
			)

		orig_protein_tags.extend(batch['conditioning_tags'].detach().tolist())
		
		if args.min_num_samples > 0 and len(orig_protein_strings) > args.min_num_samples:
			break

	tqdm_batches.refresh()

	# Save generated proteins
	# Save dir concats model name, sampling params, and if subset is used
	save_dir = os.path.join(
		'saved_output',
		'subset' if args.use_subset else 'full',
		args.model_name,
		args.sampling_params + ('_uncond' if args.uncondioned else ''),
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



	

		
		
