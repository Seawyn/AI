import numpy as np

def standardize(data):
	# new_val = (val - mean) / deviation
	means = data.mean(axis=0)
	st_devs = data.std(axis=0, ddof=1)
	standardized = np.zeros(data.shape)
	for i in range(standardized.shape[0]):
		for j in range(standardized.shape[1]):
			standardized[i, j] = (data[i, j] - means[j]) / st_devs[j]
	return standardized

def get_covariance(x, y):
	mean_x = x.mean()
	mean_y = y.mean()

	total_sum = 0
	for i in range(len(x)):
		total_sum += (x[i] - mean_x) * (y[i] - mean_y)

	return total_sum / len(x)

def cov_matrix(data):
	cov_mtr = np.zeros((data.shape[1], data.shape[1]))
	for i in range(cov_mtr.shape[0]):
		for j in range(cov_mtr.shape[1]):
			cov_mtr[i, j] = get_covariance(data[:, i], data[:, j])
	return cov_mtr

def pca(data, n_components):
	if n_components > data.shape[1]:
		raise ValueError('Number of components must be equal or below number of features') 

	st = standardize(data)
	cov_mtr = cov_matrix(st)
	eig_vals, eig_vecs = np.linalg.eig(cov_mtr)
	tr = eig_vecs[:, :n_components]

	return st.dot(tr)
