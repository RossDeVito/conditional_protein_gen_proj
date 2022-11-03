import os
import gzip

import pandas as pd
from tqdm import tqdm
from tqdm_logger import TqdmLogger
import xmltodict

import dask
import dask.bag as db
from dask.diagnostics import ProgressBar


if __name__ == '__main__':
	# data_path = os.path.join('..', 'data', 'uniref50.xml')
	# prot_data = db.read_text(data_path)

	data_path = os.path.join('..', 'data', 'uniref50.xml')
	save_dir = os.path.join('..', 'data', 'uniref50_csvs')
	log_file = 'xml_to_csvs_log.log'
	save_every_n = 100000

	# Read in 
	prot_entries = []
	n_prots = 0
	n_save_files = 0

	tqdm_stream = TqdmLogger(log_file)
	tqdm_stream.reset()
	pbar = tqdm(total=11862245, file=tqdm_stream, miniters=1000)

	with open(data_path, 'r') as f:
		line_buffer = []

		for line_num, line in enumerate(f):
			if line[:6] == '<entry':
				line_buffer = []
			line_buffer.append(line)
			if line[:7] == '</entry':
				n_prots += 1
				pbar.update(1)
				xml_dict = xmltodict.parse(''.join(line_buffer))['entry']
				line_buffer = []

				properties = dict(i.values() for i in xml_dict['property'])
				properties.pop('common taxon ID')
				prot_entries.append({
					'cluster_id': xml_dict['@id'],
					'cluster_name': xml_dict['name'],
					'cluster_size': properties.pop('member count'),
					'common_taxon': properties.pop('common taxon'),
					'property_tags': list(properties.values()),
					'last_line_num': line_num,
					'sequence': xml_dict['representativeMember']['sequence']['#text'],
					'sequence_length': xml_dict['representativeMember']['sequence']['@length']
				})

				if len(prot_entries) >= save_every_n:
					save_df = pd.DataFrame(prot_entries)
					save_df.to_csv(
						os.path.join(save_dir, f'{n_save_files}.csv'),
						index=False
					)
					n_save_files += 1
					prot_entries = []
					pbar.refresh()

	pbar.close()

	if len(prot_entries) > 0:
		save_df = pd.DataFrame(prot_entries)
		save_df.to_csv(
			os.path.join(save_dir, f'{n_save_files}.csv'),
			index=False
		)
		n_save_files += 1