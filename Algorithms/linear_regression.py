import copy
import matplotlib.pyplot as plt
import numpy as np

def get_predictions(params, vals):
	return params[0] + params[1] * np.arange(len(vals))

def cost_func(params, vals):
	pred = get_predictions(params, vals)
	# (1 / 2m) * sum(pred - obt) ** 2
	cost = (pred - vals) ** 2
	cost = sum(cost) / (2 * len(vals))
	return cost

def gradient_descent(vals, l_r=0.01, init='random', n_iter=100):
	# Params is a two dimensional weight vector (bias and weight)
	params = np.zeros(2)
	if init == 'random':
		params = np.random.random_sample(2) * 5

	m = len(vals)

	last_params = copy.deepcopy(params)
	for i in range(n_iter):
		cur_cost = cost_func(params, vals)
		
		resids = get_predictions(params, vals) - vals
		params[0] -= l_r * (2 / m) * sum(resids)
		params[1] -= l_r * (2 / m) * sum(resids * np.arange(m))

		if abs(max(params - last_params)) < 1e-8:
			break

		last_params = copy.deepcopy(params)

	return params

def test_regression():
	test = np.arange(50) - (np.random.random_sample(50) + 0.5) * 2
	params = gradient_descent(test, l_r=0.001, n_iter=1000)
	pred = get_predictions(params, test)
	plt.scatter(np.arange(len(test)), test)
	plt.plot(np.arange(len(test)), pred)
	plt.show()
