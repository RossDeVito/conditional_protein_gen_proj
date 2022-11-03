import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar



if __name__ == '__main__':
	__spec__ = None

	data_csv_dir = os.path.join('..', 'data', 'uniref50_csvs')

	print(f'CPU count:\t{dask.multiprocessing.CPU_COUNT}')

	pbar = ProgressBar()
	pbar.register()

	# Load csv dir as dask dataframe
	prot_data = dd.read_csv(os.path.join(data_csv_dir, '*.csv'))

	# # Look at protein lengths
	# prot_lens = prot_data.sequence_length.compute()

	# prots_1k = prot_lens[prot_lens <= 1000]

	# sns.histplot(prot_lens)
	# plt.savefig('protein_len_dist.png')
	# plt.close()
	# sns.histplot(prots_1k)
	# plt.savefig('protein_len_dist_lt1k.png')
	# plt.close()

	# Reduce to only sequences at most 1k in length
	prot_data = prot_data[prot_data.sequence_length.between(50, 256)].compute()

	# Get unique taxon
	taxon_counts = prot_data.common_taxon.value_counts().compute()

	# Get unique property tags
	tag_counts = prot_data.property_tags.str.extractall(r"'(.*?)'")[0].value_counts().compute()

	# Only those with property tags
	prots_w_tags = prot_data[prot_data.property_tags != '[]'].compute()
	