"""
Autoregressive transformer language model for conditional protein generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class ARLMConfig:
	""" Base configuration class for autoregressive language models. 
	
	Args:
		aa_vocab_size: Size of amino acid vocabulary, default is 23
			(20 standard amino acids + unknown + start/stop + padding).
		conditioning_tags_vocab_size: Size of conditioning tags vocabulary
			(padding + num. taxa + num. tags). The default is for 
			uniref50_small, (1 + 100 + 172) = 273.
		max_sequence_length: Maximum sequence length, default is 257
			(256 + start/stop).
		d_model: Size of the embedding, default is 512.
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


class ARLM(pl.LightningModule):
	""" Autoregressive transformer language model.

	Some stuff from https://github.com/williamFalcon/minGPT
	
	Args:
		config: ARLMConfig instance.
	"""

	def __init__(self, config):
		super().__init__()
		self.config = config
		self.max_sequence_length = config.max_sequence_length
		self.save_hyperparameters(self.config.as_dict())
		
		# Create model
		# Embedding layers
		self.conditioning_tags_embedding = nn.Embedding(
			self.config.conditioning_tags_vocab_size,
			self.config.d_model,
			padding_idx=0
		)
		self.aa_embedding = nn.Embedding(
			self.config.aa_vocab_size,
			self.config.d_model,
			padding_idx=0
		)
		self.positional_embedding = nn.Parameter(
			torch.zeros(1, self.max_sequence_length, self.config.d_model)
		)
		self.emb_dropout = nn.Dropout(self.config.emb_p_drop)
		# Transformer
		self.transformer = nn.Transformer(**self.config.transformer_kwargs)
		# Decoder head
		self.output_layer_norm = nn.LayerNorm(self.config.d_model)
		self.linear = nn.Linear(self.config.d_model, self.config.aa_vocab_size)

	def forward(self, x):
		""" Forward pass of model.

		"x_seq": x_seq,
		"y_seq": y_seq,
		"seq_mask": seq_mask,
		"conditioning_tags": batched_samples["conditioning_tags"],
		"tag_mask": tag_mask,

		Args:
			x (dict): Input dictionary with keys:
				"x_seq": Decoder target sequence of shape (batch_size, 
					max_sequence_length).
				"seq_mask": Mask for decoder target sequence of shape
					(batch_size, max_sequence_length).
				"conditioning_tags": Conditioning tags (encoder input) of shape
					(batch_size, max_num_conditioning_tags).
				"tag_mask": Mask for conditioning tags of shape
					(batch_size, max_num_conditioning_tags).
		"""
		
		# Embed inputs
		conditioning_tag_emb = self.conditioning_tags_embedding(x["conditioning_tags"])
		aa_seq_emb = self.aa_embedding(x["x_seq"])
		pos_emb = self.positional_embedding[:, :x["x_seq"].shape[1], :]

		seq_emb = self.emb_dropout(aa_seq_emb + pos_emb)

		# Transformer
		transformer_output = self.transformer(
			src=self.emb_dropout(conditioning_tag_emb),
			tgt=seq_emb,
			src_key_padding_mask=~x["tag_mask"],
			tgt_key_padding_mask=~x["seq_mask"],
			memory_key_padding_mask=~x["tag_mask"],
			tgt_mask=~torch.tril(
				torch.ones(x["x_seq"].shape[1], x["x_seq"].shape[1])
			).bool().type_as(x["tag_mask"])
		)

		# Decoder head
		transformer_output = self.output_layer_norm(transformer_output)
		transformer_output = self.linear(transformer_output)

		return transformer_output

	def shared_step(self, batch):
		""" Shared step for training and validation. """
		logits = self(batch)

		# Calculate loss
		loss = F.cross_entropy(
			logits.view(-1, logits.shape[-1]),
			batch["y_seq"].view(-1),
			ignore_index=0
		)

		return logits, loss

	def training_step(self, batch, batch_idx):
		logits, loss = self.shared_step(batch)
		self.log(
			"train_loss", loss, on_step=True, on_epoch=True, prog_bar=True
		)
		self.log(
			"train_perplexity", torch.exp(loss), on_step=True, on_epoch=True, prog_bar=True
		)
		return loss

	def validation_step(self, batch, batch_idx):
		logits, loss = self.shared_step(batch)
		self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
		ppl = torch.exp(loss)
		self.log(
			"val_perplexity", ppl, on_step=False, on_epoch=True, prog_bar=True
		)
		return {"val_loss": loss, "val_perplexity": ppl}

	def test_step(self, batch, batch_idx):
		logits, loss = self.shared_step(batch)
		self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
		ppl = torch.exp(loss)
		self.log(
			"test_perplexity", ppl, on_step=False, on_epoch=True, prog_bar=True
		)
		return {"test_loss": loss, "test_perplexity": ppl}

	def configure_optimizers(self):
		# create the optimizer
		no_decay = ["bias", "LayerNorm.weight"]
		params_decay = [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)]
		params_nodecay = [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]
		optim_groups = [
			{
				"params": params_decay, 
				"weight_decay": self.config.optimizer_kwargs.pop('weight_decay')
			},
			{"params": params_nodecay, "weight_decay": 0.0},
		]
		optimizer = torch.optim.AdamW(
			optim_groups, 
			**self.config.optimizer_kwargs
		)

		if self.config.reduce_lr_on_plateau_kwargs is not None:
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
				optimizer, **self.config.reduce_lr_on_plateau_kwargs
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



if __name__ == "__main__":
	__spec__ = None

	from protein_gen.data_modules import ProteinDataModule, AutoRegressiveLMCollationFn

	# Load data
	data_module = ProteinDataModule(
		'../../data/uniref50_small',
		collate_fn=AutoRegressiveLMCollationFn,
		train_data_fname='subset_train_data.csv',
		val_data_fname='subset_val_data.csv',
		test_data_fname='subset_test_data.csv',
		batch_size=64,
		num_workers=4,
	)
	data_module.setup()

	batch = next(iter(data_module.train_dataloader()))

	# Create model
	arlm_config = ARLMConfig()
	model = ARLM(arlm_config)

	# Forward pass
	output = model(batch)