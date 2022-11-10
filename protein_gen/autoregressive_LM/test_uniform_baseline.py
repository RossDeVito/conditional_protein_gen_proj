""" Training script for autoregressive language model. """

import argparse
import os
import json
import platform

import torch
import pytorch_lightning as pl

from protein_gen.data_modules import (
	ProteinDataModule, AutoRegressiveLMCollationFn
)
from protein_gen.autoregressive_LM import ARLMConfig, UniformBaselineARLM


if __name__ == '__main__':
	__spec__ = None

	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--training_args', type=str, required=True)

	args = parser.parse_args()

	with open(args.training_args, 'r') as f:
		training_args = json.load(f)

	# Create data module
	data_module = ProteinDataModule(
		collate_fn=AutoRegressiveLMCollationFn,
		**training_args['data_mod_kwargs']
	)

	# Create model
	model_config = ARLMConfig(
		training_config=training_args,
		**training_args['model_config_kwargs']
	)

	model = UniformBaselineARLM(model_config.as_dict())
	
	save_dir = os.path.join(
		os.getcwd(), 
		training_args['training_output_dir'], 
		training_args['model_version'],
		'uniform_baseline'
	)
	os.makedirs(save_dir, exist_ok=True)

	trainer = pl.Trainer()

	# Get performance on test set
	data_module.setup(stage='test')
	test_results = trainer.test(
		model,
		dataloaders=data_module.test_dataloader()
	)[0]

	# Save test results
	with open(os.path.join(save_dir, 'test_res.json'), 'w') as f:
		json.dump(test_results, f)