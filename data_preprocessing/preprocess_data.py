"""Subset to just samples wanted for training and preprocess."""

import os

import pandas as pd
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


# Dataset options
DATASET_VERSION_PARAMS = {
	'small': {
		'min_len': 50,
		'max_len': 256,
		'min_taxon_count': 1000,
		'min_tag_count': 1000,
		'save_dir': os.path.join('..', 'data', 'uniref50_small'),
		'drop_wo_tag': True,	# drop samples that have no tag when tag filtering applied
		'val_frac': 0.1,
		'test_frac': 0.1
	}
}


if __name__ == '__main__':
	__spec__ = None

	pbar = ProgressBar()
	pbar.register()
	print(f'CPU count:\t{dask.multiprocessing.CPU_COUNT}')

	# Options
	data_csv_dir = os.path.join('..', 'data', 'uniref50_csvs')
	ds_version = 'small'
	ds_params = DATASET_VERSION_PARAMS[ds_version]

	# Load all samples
	prot_data = dd.read_csv(os.path.join(data_csv_dir, '*.csv'))

	# Filter by length
	prot_data = prot_data[
		prot_data.sequence_length.between(ds_params['min_len'], ds_params['max_len'])
	]

	# Filter out samples without any tags
	prot_data = prot_data[prot_data.property_tags != '[]'].compute()
	prot_data = prot_data.reset_index(drop=True)

	# Find taxa that meet min count threshold, modify common_taxon to 'root'
	# if does not make min count, and save taxa that meet min.
	taxa_counts = prot_data.common_taxon.value_counts()
	included_taxa = taxa_counts[taxa_counts >= ds_params['min_taxon_count']]
	prot_data.loc[~prot_data.common_taxon.isin(included_taxa.index), 'common_taxon'] = 'root'

	if not os.path.exists(ds_params['save_dir']):
		os.makedirs(ds_params['save_dir'])
	included_taxa.reset_index().iloc[:,0].to_csv(
		os.path.join(ds_params['save_dir'], 'included_taxa.csv'),
		index=False,
		header=False
	)

	# Find tags with min number of examples and update tags for
	# examples to only include these
	flattened_tags = prot_data.property_tags.str.extractall(r"'(.*?)'")
	tag_counts = flattened_tags[0].value_counts()
	included_tags = tag_counts[tag_counts >= ds_params['min_tag_count']]
	flattened_tags = flattened_tags[flattened_tags[0].isin(included_tags.index)]

	included_tags.reset_index().to_csv(
		os.path.join(ds_params['save_dir'], 'included_tags.csv'),
		header=False,
		index=False
	)

	filtered_tags = flattened_tags.reset_index().drop(columns=['match']
		).groupby('level_0').agg(lambda x: ' '.join(x)).rename(columns={0: 'tags'})

	# Join tags
	prot_data = prot_data.join(filtered_tags).fillna({'tags': ''})

	# Drop samples without any tags if 'drop_wo_tag'
	if ds_params['drop_wo_tag'] is True:
		prot_data = prot_data[prot_data.tags != '']

	# Shuffle and Split
	prot_data = prot_data.sample(frac=1.0, random_state=147).reset_index(drop=True)

	n_samples = len(prot_data)
	end_train = int(n_samples * (1 - ds_params['val_frac'] - ds_params['test_frac']))
	end_val = int(n_samples * (1 - ds_params['test_frac']))

	train_data = prot_data.iloc[:end_train]
	val_data = prot_data.iloc[end_train:end_val]
	test_data = prot_data.iloc[end_val:]

	# Save
	train_data.to_csv(os.path.join(ds_params['save_dir'], 'train_data.csv'))
	val_data.to_csv(os.path.join(ds_params['save_dir'], 'val_data.csv'))
	test_data.to_csv(os.path.join(ds_params['save_dir'], 'test_data.csv'))

	# Save Subsets for dev
	train_data.iloc[:10000].to_csv(os.path.join(ds_params['save_dir'], 'subset_train_data.csv'))
	val_data.iloc[:5000].to_csv(os.path.join(ds_params['save_dir'], 'subset_val_data.csv'))
	test_data.iloc[:5000].to_csv(os.path.join(ds_params['save_dir'], 'subset_test_data.csv'))
