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
from protein_gen.autoregressive_LM import ARLMConfig, EmpiricalBaselineARLM


def get_n_trainable_params(model):
	""" Get number of trainable parameters in model. """
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

	model = EmpiricalBaselineARLM(model_config.as_dict())

	# Create training callbacks
	callbacks = [
		pl.callbacks.ModelCheckpoint(
			monitor="val_loss",
			filename='{epoch}-best_val_loss'
		),
		pl.callbacks.ModelCheckpoint(
			save_last=True,
			filename='{epoch}-last'
		)
	]

	if "LearningRateMonitor" in training_args['callbacks']:
		callbacks.append(pl.callbacks.LearningRateMonitor())
	if "EarlyStopping" in training_args['callbacks']:
		callbacks.append(
			pl.callbacks.EarlyStopping(
				**training_args['callbacks']['EarlyStopping']
			)
		)
	if "StochasticWeightAveraging" in training_args['callbacks']:
		callbacks.append(
			pl.callbacks.StochasticWeightAveraging(
				**training_args['callbacks']['StochasticWeightAveraging']
			)
		)

	# Create trainer
	trainer_args = {
		"callbacks": callbacks,
		"log_every_n_steps": 1,
		**training_args['trainer_kwargs']
	}
	trainer_args['logger'] = pl.loggers.TensorBoardLogger(
		os.path.join(os.getcwd(), training_args['training_output_dir']), 
		training_args['model_version'],
		default_hp_metric=False
	)

	trainer_args['accelerator'] = 'cpu'
	
	trainer = pl.Trainer(**trainer_args)
	
	# Save training arguments
	if not os.path.exists(trainer.logger.log_dir):
		os.makedirs(trainer.logger.log_dir)
	with open(os.path.join(trainer.logger.log_dir, 'training_args.json'), 'w') as f:
		json.dump(training_args, f)

	# Train model
	trainer.fit(model, data_module)

	# Get performance on test set
	data_module.setup(stage='test')
	test_results = trainer.test(ckpt_path='best',
		dataloaders=data_module.test_dataloader()
	)[0]

	# Save test results
	with open(os.path.join(trainer.logger.log_dir, 'test_res.json'), 'w') as f:
		json.dump(test_results, f)