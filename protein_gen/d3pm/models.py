"""Discrete DDPMs for protein generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from protein_gen.data_modules import AMINO_ACID_SYM_TO_IDX, AMINO_ACID_IDX_TO_SYM


class D3PMConfig:
	"""Config wrapper for D3PM. Returns dict as arg when creating D3PM.

	Args:
		aa_vocab_size: Size of amino acid vocabulary, default is 23
			(20 standard amino acids + unknown + start/stop + padding).
		conditioning_tags_vocab_size: Size of conditioning tags vocabulary
			(padding + num. taxa + num. tags). The default is for 
			uniref50_small, (1 + 100 + 172) = 273.
		max_sequence_length: Maximum sequence length, default is 257
			(256 + start/stop).
		d_model: Size of the embedding, default is 512.
		emb_p_drop: Dropout probability for the embedding, default is 0.1.
		mask_p_floor: The chance of masking a token in training will be
			RandUniform(0, 1) + mask_p_floor. Goal is to avoid unmasked
			inputs and increase chance of fully masked inputs without
			using descrete steps. Default is 0.05.
		transformer_kwargs (dict or default None): Keyword arguments for
			torch.nn.Transformer, excluding d_model. If None, default
			kwargs are used.
		optimizer_kwargs (dict or default None): Keyword arguments for
			torch.optim.AdamW, excluding model parameters.
		training_config (dict or default None): Shortcut for saving full
			training configuration json along with model's hyperparams.
		reduce_lr_on_plateau_kwargs (dict or default None): If not None,
			keyword arguments for torch.optim.lr_scheduler.ReduceLROnPlateau.
	"""
	batch_first = True

	def __init__(
		self,
		aa_vocab_size=23,
		conditioning_tags_vocab_size=273,
		max_sequence_length=257,
		d_model=512,
		emb_p_drop=0.1,
		mask_p_floor=0.05,
		transformer_kwargs=None,
		optimizer_kwargs=None,
		reduce_lr_on_plateau_kwargs=None,
		training_config=None,
	):
		self.aa_vocab_size = aa_vocab_size
		self.conditioning_tags_vocab_size = conditioning_tags_vocab_size
		self.max_sequence_length = max_sequence_length
		self.d_model = d_model
		self.emb_p_drop = emb_p_drop
		self.mask_p_floor = mask_p_floor
		self.training_config = training_config

		self.transformer_kwargs = transformer_kwargs
		if self.transformer_kwargs is None:
			self.transformer_kwargs = {'d_model': d_model}
		else:
			self.transformer_kwargs['d_model'] = d_model
		self.transformer_kwargs['batch_first'] = self.batch_first

		self.optimizer_kwargs = optimizer_kwargs
		if self.optimizer_kwargs is None:
			self.optimizer_kwargs = {}

		self.reduce_lr_on_plateau_kwargs = reduce_lr_on_plateau_kwargs
		
	def as_dict(self):
		return vars(self)


class D3PM(pl.LightningModule):
	"""Discrete DDPM for protein generation.
	
	Args:
		config (dict): Configuration for D3PM.
	"""

	def __init__(self, config):
		super().__init__()
		self.config = config
		self.save_hyperparameters()

		# Create model
		# Embedding layers
		self.conditioning_tags_emb = nn.Embedding(
			self.config['conditioning_tags_vocab_size'],
			self.config['d_model'],
			padding_idx=0,
		)
		self.aa_emb = nn.Embedding(
			self.config['aa_vocab_size'] + 1, # +1 for input masking
			self.config['d_model'],
			padding_idx=0,
		)
		self.positional_embedding = nn.Parameter(
			torch.zeros(1, self.config['max_sequence_length'], self.config['d_model'])
		)
		self.emb_dropout = nn.Dropout(self.config['emb_p_drop'])

		# Transformer
		self.transformer = nn.Transformer(**self.config['transformer_kwargs'])

		# Decoder head
		self.output_layer_norm = nn.LayerNorm(self.config['d_model'])
		self.output_layer = nn.Linear(
			self.config['d_model'], self.config['aa_vocab_size']
		)

	def forward(self, x):
		"""Forward pass.
		
		x is dict with items:
			"masked_sequence": (masked) input sequence
			"conditioning_tags": conditioning tags
			"tag_mask": mask for conditioning tags

		I'm going to not mask the input sequences in terms of attention to pad
		tokens like in the AR models b/c diffusion instead of AR.

		Args:
			x (dict): Input dictionary.
		"""

		# Embed inputs
		cond_tag_emb = self.conditioning_tags_emb(x['conditioning_tags'])
		aa_seq_emb = self.aa_emb(x['masked_sequence'])
		pos_emb = self.positional_embedding[:, :x['masked_sequence'].shape[1], :]

		seq_emb = self.emb_dropout(aa_seq_emb + pos_emb)

		# Transformer
		transformer_output = self.transformer(
			src=self.emb_dropout(cond_tag_emb),
			src_key_padding_mask=~x["tag_mask"],
			memory_key_padding_mask=~x["tag_mask"],
			tgt=seq_emb,
		)

		# Decoder head
		transformer_output = self.output_layer_norm(transformer_output)
		transformer_output = self.output_layer(transformer_output)

		return transformer_output

	def shared_step(self, batch):
		"""Shared step for training and validation."""

		logits = self(batch)

		# Calculate loss
		loss = F.cross_entropy(
			logits.view(-1, logits.shape[-1]),
			batch["sequence"].view(-1),
			ignore_index=0
		)

		return logits, loss

	def add_masked_seq(self, batch):
		""" Add randomly masked version of sequence to batch dict.
		
		Will randomly mask tokens by:
			- Generate a random uniform val 0-1 for each seq in batch
			- Add self.config['mask_p_floor'] to rand val and use as
				probability of each token being masked
			- Set masked tokens to token idx self.config['aa_vocab_size']
		"""
		# Randomly draw masking prob for each seq in batch
		mask_p = torch.rand(batch['sequence'].shape[0], 1)
		mask_p = torch.clamp(mask_p + self.config['mask_p_floor'], 0, 1)
		mask_p = mask_p.to(batch['sequence'].device)

		# Randomly mask tokens
		mask_p = mask_p.expand(batch['sequence'].shape)
		mask = torch.bernoulli(mask_p).bool()

		masked_seqs = batch['sequence'].clone()
		masked_seqs[mask] = self.config['aa_vocab_size']
		batch['masked_sequence'] = masked_seqs

		return batch

	def add_masked_seq_set_p(self, batch, p):
		""" Randomly mask sequence tokens, but with a set probability.

		Used when sampling to generate proteins.
		
		Args:
			batch (dict): Batch dict.
			p (float): Probability of masking each token.
		"""
		# Randomly mask tokens
		mask_p = torch.full(batch['sequence'].shape, p)
		mask_p = mask_p.to(batch['sequence'].device)
		mask = torch.bernoulli(mask_p).bool()

		masked_seqs = batch['sequence'].clone()
		masked_seqs[mask] = self.config['aa_vocab_size']
		batch['masked_sequence'] = masked_seqs

		return batch

	def training_step(self, batch, batch_idx):
		""" Training step.
		
		Will randomly mask tokens by:
			- Generate a random uniform val 0-1 for each seq in batch
			- Add self.config['mask_p_floor'] to rand val and use as
				probability of each token being masked
			- Set masked tokens to token idx self.config['aa_vocab_size']
		"""

		# Randomly mask input seq
		batch = self.add_masked_seq(batch)

		# Forward pass
		logits, loss = self.shared_step(batch)
		self.log(
			"train_loss", loss, on_step=True, on_epoch=True, prog_bar=True
		)
		return loss

	def validation_step(self, batch, batch_idx):
		batch = self.add_masked_seq(batch)
		logits, loss = self.shared_step(batch)
		self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
		return loss

	def test_step(self, batch, batch_idx):
		batch = self.add_masked_seq(batch)
		logits, loss = self.shared_step(batch)
		self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
		return loss

	def configure_optimizers(self):
		# create the optimizer
		no_decay = ["bias", "LayerNorm.weight"]
		params_decay = [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)]
		params_nodecay = [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]
		optim_groups = [
			{
				"params": params_decay, 
				"weight_decay": self.config['optimizer_kwargs']['weight_decay']
			},
			{"params": params_nodecay, "weight_decay": 0.0},
		]
		optimizer = torch.optim.AdamW(
			optim_groups, 
			**{k:v for k,v in self.config['optimizer_kwargs'].items() if k != 'weight_decay'}
		)

		if self.config['reduce_lr_on_plateau_kwargs'] is not None:
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
				optimizer, **self.config['reduce_lr_on_plateau_kwargs']
			)
			return {
				'optimizer': optimizer,
				'lr_scheduler': {
					'scheduler': scheduler,
					'monitor': 'val_loss',
				}
			}
		else:
			return optimizer

	@torch.no_grad()
	def sample(
		self,
		x,
		max_length,
		temperature=1.0, 
		top_k=1,
		return_strings=True,
		n_diffusion_steps=20,
		use_mask_p_floor=False,
	):
		""" Sample from model via diffusion with n_diffusion_steps steps.

		Diffusion will begin with an all masked sequence of length
		self.config['max_sequence_length'].
		
		Args:
			x (dict): Input dictionary with keys:
				"conditioning_tags": Conditioning tags (encoder input) of shape
					(batch_size, max_num_conditioning_tags).
				"tag_mask": Mask for conditioning tags of shape
					(batch_size, max_num_conditioning_tags).
			temperature (float): Temperature for sampling.
			top_k (int, default 1): Number of top k tokens to sample from.
			return_string (bool, default True): Whether to return the sampled
				sequence as a string in addition to the token ids.
			n_diffusion_steps (int): Number of diffusion steps, each of which
				have linearly less masking of the sequences from the previous
				forward pass.
			use_mask_p_floor (bool): If True, diffusion will have minimum mask
				prob of mask_p_floor, otherwise 0 (default).
		"""

		# Get device
		device = next(self.parameters()).device
	
		# Move conditioning tags to device
		conditioning_tags = x["conditioning_tags"].to(device)
		tag_mask = x["tag_mask"].to(device)

		# Create all masked sequence
		seqs = torch.full(
			(conditioning_tags.shape[0], max_length),
			fill_value=self.config['aa_vocab_size'],
			dtype=torch.long,
			device=device
		)

		# compute masking probabilities for n_diffusion_steps steps
		min_mask_prob = self.config['mask_p_floor'] if use_mask_p_floor else 0
		mask_ps = torch.linspace(
			1, min_mask_prob, steps=n_diffusion_steps, device=device
		)

		# Loop over diffusion steps
		batch = {
			"sequence": seqs,
			"conditioning_tags": conditioning_tags,
			"tag_mask": tag_mask,
		}

		for i in range(n_diffusion_steps):
			# Add noise to seq
			batch = self.add_masked_seq_set_p(batch, p=mask_ps[i].item())

			# Forward pass
			logits = self.forward(batch)

			# Sample from logits
			logits = logits / temperature

			if top_k > 1:
				top_k_logits, top_k_indices = torch.topk(logits, top_k)
				distribution = torch.distributions.categorical.Categorical(logits=top_k_logits)
				tokens = distribution.sample()
			else:
				tokens = torch.argmax(logits, dim=-1)

			# Update batch
			batch['sequence'] = tokens

		# Convert to strings using tokens and AMINO_ACID_IDX_TO_SYM.
		# Will stop when hits start/stop token
		if return_strings:
			seq_strings = []
			for i in range(batch['sequence'].shape[0]):
				aa_list = []
				for aa_idx in batch['sequence'][i]:
					aa_string = AMINO_ACID_IDX_TO_SYM[aa_idx.item()]
					if aa_string == 'start/stop':
						break
					elif aa_string == 'pad':
						continue
					aa_list.append(aa_string)
				seq_strings.append(''.join(aa_list))

			return batch['sequence'], seq_strings
		else:
			return batch['sequence']


if __name__ == '__main__':
	import os
	from protein_gen.data_modules import (
		ProteinDataset, D3PMCollationFn
	)

	# test
	training_args = {
	"training_output_dir": "training_output",
	"model_version": "v0",
	"data_mod_kwargs": {
		"data_dir": "../../data/uniref50_small",
		"train_data_fname": "train_data.csv",
		"test_data_fname": "test_data.csv",
		"val_data_fname": "val_data.csv",
		"batch_size": 128,
		"num_workers": 4,
		"taxon_dropout_rate": 0.2,
		"attribute_dropout_rate": 0.2
	},
	"model_config_kwargs": {
		"d_model": 128,
		"transformer_kwargs": {
			"nhead": 8,
			"num_encoder_layers": 1,
			"num_decoder_layers": 5,
			"dim_feedforward": 192,
			"dropout": 0.1,
			"activation": "gelu",
			"norm_first": True
		},
		"optimizer_kwargs": {
			"betas": [0.9, 0.95],
			"lr": 1e-4,
			"weight_decay": 1e-2
		},
		"reduce_lr_on_plateau_kwargs": {
			"patience": 5,
			"factor": 0.5,
			"cooldown": 1,
			"verbose": True
		}
	},
	"callbacks": {
		"LearningRateMonitor": {},
		"EarlyStopping": {
			"monitor": "val_loss",
			"patience": 15
		},
		"StochasticWeightAveraging": {
			"swa_lrs": 1e-3
		}
	},
	"trainer_kwargs": {
		"max_epochs": -1,
		"accumulate_grad_batches": 10,
		"gradient_clip_val": 0.5,
		"precision": 16
	}
}

	# Create model
	config = D3PMConfig(
		training_config=training_args,
		**training_args['model_config_kwargs']
	)
	model = D3PM(config.as_dict())

	# Load data
	data_path = os.path.join(
		'..', '..', 'data', 'uniref50_small'
	)

	dataset = ProteinDataset(
		data_path, 
		'subset_train_data.csv', 
		taxon_dropout_rate=0.0,
		attribute_dropout_rate=0.0,
	)

	# Test dataset
	batch = [dataset[0], dataset[12], dataset[13], dataset[101]]
	batch = D3PMCollationFn(batch)

	# Forward with mask
	batch = model.add_masked_seq_set_p(batch, p=0.5)

	logits = model(batch)

	# Test sampling
	samples, seq_strings = model.sample(
		batch,
		n_diffusion_steps=10,
		temperature=1.0,
		top_k=1,
		use_mask_p_floor=False,
		return_strings=True
	)