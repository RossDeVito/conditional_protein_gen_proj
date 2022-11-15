import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats


# def get_cov(size=20, length=50):    
# 	""" From cosmiccoding.com.au/tutorials/gaussian_processes """
# 	x = torch.arange(size)
# 	cov = torch.exp(-(1 / length) * (x - np.atleast_2d(x).T)**2)
# 	return 


if __name__ == '__main__':
	seq_len = 100
	dist_var = 10.0

	idx = torch.arange(seq_len).float()

	# Make exponentiated quadratic covariance matrix
	l2_dist = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2

	# exp(-||x - y||**2 / (2 * length_scale**2))
	cov = torch.exp(-l2_dist / (2 * dist_var))
	# cov = cov / cov.sum(dim=1, keepdim=True)

	# Sample from multivariate normal
	mn = stats.multivariate_normal(
		mean=np.zeros(seq_len), cov=cov.numpy(), allow_singular=True
	)


	# Plot covariance matrix
	sns.heatmap(cov)
	plt.show()

	# Create Multivariate Normal distribution to sample from for masking
	mvn = torch.distributions.MultivariateNormal(
		torch.zeros(seq_len),
		covariance_matrix=cov
	)
	# mvn = torch.distributions.MultivariateNormal(
	# 	torch.zeros(seq_len),
	# 	covariance_matrix=get_cov(seq_len, dist_var)
	# )

	# # Sample from MVN
	# samples = mvn.sample((10,))

	# # Take top N samples
	# top_n = 5
	# top_n_samples = torch.topk(samples, top_n, dim=1).indices

	# Make demo seq to mask
	seq = torch.tensor(
		[0] * 6
		+ [1] * 7
		+ [2] * 8
		+ [3] * 9
	).float()
	# stack into batch
	batch = torch.stack([seq] * 6)
	
	# Random sample noise then guassian blur
	noise = torch.randn_like(batch)
	noise = torch.nn.functional.gaussian_blur1d(noise, (dist_var,))




