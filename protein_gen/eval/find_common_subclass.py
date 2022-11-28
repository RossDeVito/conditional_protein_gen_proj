""" Common sets of conditioning tags. """

import os
import json
from ast import literal_eval
from functools import partial

import numpy as np
import pandas as pd

from protein_gen.data_modules import ProteinDataset


def to_set_and_remove_zero(val_list):
	vals = set(val_list)
	vals.discard(0)
	return vals


if __name__ =='__main__':
	res_path = os.path.join(
		'..', 'autoregressive_LM', 'saved_output', 'full', 'arlm_large',
		'top10_v1', 'generated_proteins.csv'
	)

	tags = pd.read_csv(
		res_path, converters={"orig_protein_tags": literal_eval}
	).orig_protein_tags

	tag_sets = tags.map(to_set_and_remove_zero)

	# Get tag counts
	flat_tags = np.concatenate(tags.tolist()).ravel()
	t, c = np.unique(flat_tags, return_counts=True)
	sort_idx = np.argsort(-c)
	sorted_tags = t[sort_idx]
	tag_counts = c[sort_idx]

	data_mod = ProteinDataset(
		data_dir='../../data/uniref50_small',
		data_fname='test_data.csv'
	)

	# Find indices with the following tags and save as JSON
	# Tag			Key	Count	Meaning
	# GO:0016021	100	10569	Cellular component: membrane
	# GO:0005737	103	 1268	Cellular component: cytoplasm
	# GO:0003677	101	 1896	Molecular function: DNA binding
	# GO:0005524	102	 1430	Molecular function: ATP binding
	#
	# Links:
	#	http://www.informatics.jax.org/vocab/gene_ontology/GO:0016020
	#	http://www.informatics.jax.org/vocab/gene_ontology/GO:0005737
	#	http://www.informatics.jax.org/vocab/gene_ontology/GO:0003677
	#	http://www.informatics.jax.org/vocab/gene_ontology/GO:0005524

	tags_to_save = {
		'component_membrane': 100,
		'component_cytoplasm': 103,
		'function_DNA_binding': 101,
		'function_ATP_binding': 102
	}

	for t, t_id in tags_to_save.items():
		tags_to_save[t] = np.where(tag_sets.apply(lambda x: t_id in x))[0].tolist()

	# Save
	with open(os.path.join('knn_output', 'tag_idx.json'), 'w') as f:
		json.dump(tags_to_save, f)