""" 
Use k-nearest neighbors in embedding space to evaluate impact of conditioning.
"""

import argparse
import os
import json

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from joblib import dump, load


if __name__ == '__main__':
	# Parse arguments.
	parser = argparse.ArgumentParser()
	parser.add_argument('-e', '--emb_dir', type=str, default='prot_t5_xl_half')
	parser.add_argument('-m', '--model_dir', type=str, required=True)
	# -g will train and store knn model trained on true embeddings
	parser.add_argument('-g', '--ground_truth', action='store_true') 
	# parser.add_argument('-mm', '--masked_mean', action='store_true')
	parser.add_argument('-k', '--k', type=int, default=5)
	parser.add_argument('-w', '--knn_weight_func', type=str, default='uniform')
	# if not -f, will do cellular component comparison
	parser.add_argument('-f', '--function_cls', action='store_true')
	
	args = parser.parse_args()

	# Load indices of embeddings of classes
	with open(os.path.join('knn_output', 'tag_idx.json'), 'r') as f:
		tag_idx = json.load(f)

	if args.function_cls:
		print('Running KNN for molecular function')
		task_name = 'function'
		cls0 = 'function_DNA_binding'
		cls1 = 'function_ATP_binding'
	else:
		print('Running KNN for cellular component')
		task_name = 'component'
		cls0 = 'component_membrane'
		cls1 = 'component_cytoplasm'

	cls0_idx = tag_idx[cls0]
	cls1_idx = tag_idx[cls1]

	# Load data and make into labeled dataset
	emb_path = os.path.join(
		'embeddings', args.emb_dir, args.model_dir,
		'prot_masked_mean_embs.npy'
	)
	prot_embs = np.load(emb_path)

	cls0_emb = prot_embs[cls0_idx]
	cls1_emb = prot_embs[cls1_idx]

	X = np.vstack([cls0_emb, cls1_emb])
	y = np.concatenate(
		[np.zeros(cls0_emb.shape[0]), np.ones(cls1_emb.shape[0])]
	)

	# Define paths for laoding and saving model
	params_dir = f'{task_name}_k{args.k}_w{args.knn_weight_func}'
	gt_model_save_dir = os.path.join(
		'knn_output', 'ground_truth_knn_models', args.emb_dir,
		'ground_true', params_dir,
	)
	gt_model_save_path = os.path.join(gt_model_save_dir, 'gt_model.joblib')

	# If -g flag, train and save ground truth model
	if args.ground_truth:
		# Train model
		knn_model = KNeighborsClassifier(
			n_neighbors=args.k, weights=args.knn_weight_func, n_jobs=-1
		)
		knn_model.fit(X, y)

		# Save model
		if not os.path.exists(gt_model_save_dir):
			os.makedirs(gt_model_save_dir)
		dump(knn_model, gt_model_save_path) 
	else:
		# Load model
		knn_model = load(gt_model_save_path)

		# Evaluate and save output probs
		pred_probs = knn_model.predict_proba(X)

		cls_report = metrics.classification_report(
			y_true=y, y_pred=pred_probs.argmax(1), output_dict=True
		)
		ap_score_macro = metrics.average_precision_score(
			y_true=np.eye(2)[y.astype(int)], y_score=pred_probs, average='macro'
		)
		ap_score_by_class = metrics.average_precision_score(
			y_true=np.eye(2)[y.astype(int)], y_score=pred_probs, average=None
		)

		cls_report['macro avg']['avg_prec'] = ap_score_macro
		cls_report['0.0']['avg_prec'] = ap_score_by_class[0]
		cls_report['1.0']['avg_prec'] = ap_score_by_class[1]

		# Save results as JSON
		res_save_path = os.path.join(
			'knn_output', args.emb_dir, args.model_dir, 
		)
		if not os.path.exists(res_save_path):
			os.makedirs(res_save_path)

		res_fname = params_dir + '_eval.json'
		with open(os.path.join(res_save_path, res_fname), 'w') as f:
			json.dump(cls_report, f)

		# Find and save top example idx

