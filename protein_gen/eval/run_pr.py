""" Run k-NN precision-recall evaluation. """

import argparse
import os
import json

import numpy as np
import torch

from precision_recall import IPR


if __name__ == '__main__':
	# Parse arguments.
	parser = argparse.ArgumentParser()
	parser.add_argument('-e', '--emb_dir', type=str, default='prot_t5_xl_half')
	parser.add_argument('-m', '--model_dir', type=str, required=True)
	parser.add_argument('-mm', '--masked_mean', action='store_true')
	parser.add_argument('-s', '--save_ground_truth_manifold', action='store_true')
	parser.add_argument('-k', '--k', type=int, default=3)

	args = parser.parse_args()

	# Set up IPR
	ipr = IPR(args.k)
	ipr_manifold_dir = os.path.join(
		'ipr_manifolds', args.emb_dir, 
		'masked_mean' if args.masked_mean else 'mean',
		'subset' if 'subset' in args.model_dir else 'full'
	)

	# If --save_ground_truth_manifold is set, save ground truth manifold to
	# disk to then use with models with the same emb_dir and masked_mean args.
	if args.save_ground_truth_manifold:
		print('Saving ground truth manifold to disk')
		# Load ground truth embeddings.
		gt_embs = np.load(os.path.join(
			'embeddings', args.emb_dir,
			'ground_true_subset' if 'subset' in args.model_dir else 'ground_true', 
			'prot_masked_mean_embs.npy' if args.masked_mean else 'prot_mean_embs.npy'
		))

		# Save ground truth manifold.
		ipr.compute_manifold_ref(gt_embs)
		
		os.makedirs(ipr_manifold_dir, exist_ok=True)
		ipr.save_ref(
			os.path.join(ipr_manifold_dir, 'k{}_gt_manifold'.format(args.k))
		)
	
	# Otherwise, load ground truth manifold and run k-NN precision-recall
	else:
		# Load generated protein embeddings.
		emb_path = os.path.join(
			'embeddings', args.emb_dir, args.model_dir,
			'prot_masked_mean_embs.npy' if args.masked_mean else 'prot_mean_embs.npy'
		)
		prot_embs = np.load(emb_path)

		# compute metric
		ipr.compute_manifold_ref(
			os.path.join(ipr_manifold_dir, 'k{}_gt_manifold.npz'.format(args.k))
		)
		metric = ipr.precision_and_recall(prot_embs)

		print('k-NN precision-recall metric: {}'.format(metric))

		# Save metric as JSON
		is_masked_mean = 'masked_mean' if args.masked_mean else 'mean'
		save_path = os.path.join(
			'ipr_output', args.emb_dir, is_masked_mean, str(args.k),
			'{}.json'.format(args.model_dir)
		)
		os.makedirs(os.path.dirname(save_path), exist_ok=True)

		with open(save_path, 'w') as f:
			json.dump({
				'precision': metric.precision,
				'recall': metric.recall,
				'model_name': args.model_dir,
				'k': args.k,
				'masked_mean': args.masked_mean,
				'emb_model': args.emb_dir,
			}, f)
	
