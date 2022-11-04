"""Dataset class for protein data."""
import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


""" AA counts in uniref50_small train split
L    17573382
A    15113074
G    11900521
V    11781465
S    11133398
I    10190101
R    10074974
E     9649921
T     8819863
D     8376274
K     8083029
P     8072415
F     7206278
N     5945184
Q     5699055
Y     4965135
M     3970114
H     3669758
W     2322306
C     2289727
X       57665
B          65
Z          51
U          29
O           1

Will map all unusual translations and ambiguous residues to X
"""

AMINO_ACID_SYM_TO_IDX = {
	"pad": 0,	# Padding
	"start/stop": 22,
	"A": 1,		# 20 standard amino acid residues
	"C": 2,
	"D": 3,
	"E": 4,
	"F": 5,
	"G": 6,
	"H": 7,
	"I": 8,
	"K": 9,
	"L": 10,
	"M": 11,
	"N": 12,
	"P": 13,
	"Q": 14,
	"R": 15,
	"S": 16,
	"T": 17,
	"V": 18,
	"W": 19,
	"Y": 20,
	"X": 21,	# Unknown token - Below also mapped to X
	"B": 21,	# Aspartic acid or asparagine
	"Z": 21,	# Glutamic acid or glutamine
	"U": 21,	# Selenocysteine
	"O": 21,	# Pyrrolysine
}


class ProteinDataset(Dataset):
	""""
	Dataset class for protein data, including conditioning tags and
	sequence data.

	Handles subsampling and shuffling of conditioning tags.

	TODO: Larger subsets of uniref50 may need dask incorporated here to handle
	larger than memory dataset.

	Args:
		data_dir: Path to data directory that includes data_fname, 
			included_tags.csv, and included_taxa.csv.
		data_fname: Name of data CSV file.
		taxon_dropout_rate (default 0.0): Probability of sample's taxon
			being excluded as a conditioning tag.
		attribute_dropout_rate (default 0.0): Probability of each attribute
			tag being excluded from conditioning set.
	"""

	def __init__(
		self,
		data_dir,
		data_fname,
		taxon_dropout_rate=0.0,
		attribute_dropout_rate=0.0,
	):
		self.data_file = os.path.join(data_dir, data_fname)
		self.taxon_dropout_rate = taxon_dropout_rate
		self.attribute_dropout_rate = attribute_dropout_rate
		
		# Load data
		self.data = pd.read_csv(self.data_file, index_col=0)

		# Create value to integer mapping for conditioning tags
		# Taxa
		self.taxa_idx_to_val = pd.read_csv(
			os.path.join(data_dir, "included_taxa.csv"), header=None
		)[0].to_dict()
		self.taxa_val_to_idx = {v: k + 1 for k, v in self.taxa_idx_to_val.items()}
		self.taxa_idx_to_val = {v: k for k, v in self.taxa_val_to_idx.items()}

		# GO Attributes
		self.attributes_idx_to_val = pd.read_csv(
			os.path.join(data_dir, "included_tags.csv"), header=None
		)[0].to_dict()
		# modify keys to start after last taxon index
		last_taxon_idx = max(self.taxa_val_to_idx.values())
		self.attributes_idx_to_val = {
			k + last_taxon_idx: v for k, v in self.attributes_idx_to_val.items()
		}
		self.attributes_val_to_idx = {
			v: k for k, v in self.attributes_idx_to_val.items()
		}

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		"""Returns a sample from the dataset.

		Args:
			idx: Index of sample to return.

		Returns:
			Sample as a dict containing:
				sequence: Sequence of amino acids as a tensor of indices.
				conditioning_tags: Conditioning tags as a tensor of indices.
		"""
		# Get sample
		sample = self.data.iloc[idx]

		# Get sequence
		sequence = torch.tensor(
			[AMINO_ACID_SYM_TO_IDX[sym] for sym in sample["sequence"]]
		)

		# Get conditioning tags
		conditioning_tags = []
		# Taxon
		if torch.rand(1) > self.taxon_dropout_rate:
			conditioning_tags.append(
				self.taxa_val_to_idx[sample["common_taxon"]]
			)
		# Attributes
		for tag in sample["tags"].split(" "):
			if tag != '' and torch.rand(1) > self.attribute_dropout_rate:
				conditioning_tags.append(self.attributes_val_to_idx[tag])

		if conditioning_tags == []:
			conditioning_tags = [self.taxa_val_to_idx['root']]

		# Shuffle conditioning tags
		conditioning_tags = torch.tensor(conditioning_tags)
		conditioning_tags = conditioning_tags[torch.randperm(len(conditioning_tags))]

		return {
			"sequence": sequence,
			"sequence_length": sample['sequence_length'],
			"conditioning_tags": conditioning_tags,
		}


def SeperateInputColationFn(batch):
	"""Collation function for protein dataset that separates sequence and
	conditioning tags into separate tensors.

	Args:
		batch: Batch of samples from dataset.

	Returns:
		Batch as a dict containing:
			sequence: Sequence of amino acids as a tensor of indices.
			sequence_lengths: Lengths of sequences as a tensor of indices.
			conditioning_tags: Conditioning tags as a tensor of indices.
	"""
	sequence = torch.nn.utils.rnn.pad_sequence(
		[sample["sequence"] for sample in batch], 
		batch_first=True,
		padding_value=0
	).long()

	# Get sequence lengths
	sequence_lengths = torch.tensor(
		[sample["sequence_length"] for sample in batch]
	).long()

	# Get conditioning tags
	conditioning_tags = torch.nn.utils.rnn.pad_sequence(
		[sample["conditioning_tags"] for sample in batch], 
		batch_first=True,
		padding_value=0
	).long()

	return {
		"sequence": sequence,
		"sequence_lengths": sequence_lengths,
		"conditioning_tags": conditioning_tags,
	}


def AutoRegressiveLMCollationFn(batch):
	batched_samples = SeperateInputColationFn(batch)

	# Create input seq by prepending the start/stop token to the sequence
	startstop_token = AMINO_ACID_SYM_TO_IDX["start/stop"]

	x_seq = torch.cat(
		[
			torch.full(
				(batched_samples["sequence"].shape[0], 1), 
				AMINO_ACID_SYM_TO_IDX["start/stop"]
			), 
			batched_samples["sequence"]
		],
		dim=1
	).long()

	# Create target seq by appending the end token to the sequence
	y_seq = torch.cat(
		[
			batched_samples["sequence"],
			torch.zeros(
				(batched_samples["sequence"].shape[0], 1)
			)
		],
		dim=1
	).long()
	y_seq[
		torch.arange(batched_samples['sequence_lengths'].shape[0]),
		batched_samples['sequence_lengths']
	] = AMINO_ACID_SYM_TO_IDX["start/stop"]

	# Create mask for y_seq loss
	seq_lengths = batched_samples["sequence_lengths"] + 1
	# seq_mask = torch.arange(y_seq.shape[1]).expand(len(seq_lengths), y_seq.shape[1]) < seq_lengths.unsqueeze(1)
	seq_mask = y_seq.bool()

	# Create mask for conditioning tags
	tag_mask = batched_samples["conditioning_tags"].bool()

	return {
		"x_seq": x_seq,
		"y_seq": y_seq,
		"seq_mask": seq_mask,
		"conditioning_tags": batched_samples["conditioning_tags"],
		"tag_mask": tag_mask,
	}


class ProteinDataModule(pl.LightningDataModule):
	"""
	Data module that handles train/val/test datasets during training
	and evaluation.
	
	Args:
		data_dir: Path to directory containing data.
		collate_fn: Collate function to use (e.g. SeperateInputColationFn).
		train_data_fname (default 'train_data.csv'): Training data CSV in
			data_dir.
		val_data_fname (default 'val_data.csv'): Validation data CSV in
			data_dir.
		test_data_fname (default 'test_data.csv'): Test data CSV in
			data_dir.
		batch_size (default 64): Batch size to use.
		num_workers (default 4): Number of workers to use for data loading.
		**kwargs: Additional keyword arguments passed to ProteinDataset.
	"""

	def __init__(
		self,
		data_dir,
		collate_fn,
		train_data_fname="train_data.csv",
		val_data_fname="val_data.csv",
		test_data_fname="test_data.csv",
		batch_size=64,
		num_workers=4,
		**kwargs
	):
		super().__init__()
		self.data_dir = data_dir
		self.collate_fn = collate_fn
		self.train_data_fname = train_data_fname
		self.val_data_fname = val_data_fname
		self.test_data_fname = test_data_fname
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.kwargs = kwargs

	def setup(self, stage=None):
		"""Sets up train/val/test datasets."""
		if stage == "fit" or stage is None:
			self.train_dataset = ProteinDataset(
				self.data_dir,
				self.train_data_fname,
				**self.kwargs
			)
			self.val_dataset = ProteinDataset(
				self.data_dir,
				self.val_data_fname,
				**self.kwargs
			)
		
		if stage == "test" or stage is None:
			self.test_dataset = ProteinDataset(
				self.data_dir,
				self.test_data_fname,
				**self.kwargs
			)

	def train_dataloader(self):
		"""Returns training dataloader."""
		return DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=self.num_workers,
			collate_fn=self.collate_fn
		)

	def val_dataloader(self):
		"""Returns validation dataloader."""
		return DataLoader(
			self.val_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			collate_fn=self.collate_fn
		)

	def test_dataloader(self):
		"""Returns test dataloader."""
		return DataLoader(
			self.test_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			collate_fn=self.collate_fn
		)


if __name__ == '__main__':
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

	collated_batch_sep = SeperateInputColationFn(batch)

	collated_batch = AutoRegressiveLMCollationFn(batch)
