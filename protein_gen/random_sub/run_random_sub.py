"""
Randomly substitute X% of the amino acids in a protein sequence with a random 
different amino acid.
"""

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from protein_gen.data_modules import ProteinDataset, AMINO_ACID_SYM_TO_IDX

INVALID_SUBS = ['pad', 'start/stop', 'B', 'Z', 'U', 'O']


if __name__ == '__main__':
	save_dir = 'saved_random_subs'

	# Parse command line arguments
	parser = argparse.ArgumentParser()

	parser.add_argument('-d', '--data_dir', type=str, required=True)
	parser.add_argument('-s', '--use_subset', action='store_true')
	parser.add_argument('-p', '--percent_sub', type=float, default=0.1)
	parser.add_argument('-n', '--num_replicates', type=int, default=1)
	
	args = parser.parse_args()

	# Load data
	dataset = ProteinDataset(
		data_dir=args.data_dir,
		data_fname='subset_test_data.csv' if args.use_subset else 'test_data.csv',
	)

	# Randomly substitute X% of the amino acids in each protein sequence
	valid_subs = [
		aa for aa in AMINO_ACID_SYM_TO_IDX.keys() if aa not in INVALID_SUBS
	]
	save_dir = os.path.join(
		save_dir, 
		'subset' if args.use_subset else 'full',
		f'percent_sub_{args.percent_sub}'
	)

	for r in tqdm(range(args.num_replicates)):
		seqs = dataset.data.sequence.tolist()

		# Make substitutions
		for i, seq in tqdm(enumerate(seqs), total=len(seqs)):
			seq = list(seq)
			num_subs = int(len(seq) * args.percent_sub)
			sub_indices = np.random.choice(len(seq), num_subs, replace=False)
			for sub_idx in sub_indices:
				# Randomly replace aa at idx with a different aa
				aa = seq[sub_idx]
				new_aa = np.random.choice(valid_subs)
				while new_aa == aa:
					new_aa = np.random.choice(valid_subs)
				seq[sub_idx] = new_aa
			seqs[i] = ''.join(seq)

		# Make dataframe to save results
		rand_seqs = pd.DataFrame({
			'orig_protein_string': dataset.data.sequence.tolist(),
			'gen_protein_string': seqs,
		})

		# Save data
		out_fname = f'rand_sub_replicate_{r}.csv'
		out_fpath = os.path.join(save_dir, out_fname)
		os.makedirs(os.path.dirname(out_fpath), exist_ok=True)
		
		rand_seqs.to_csv(out_fpath, index=False)

	


