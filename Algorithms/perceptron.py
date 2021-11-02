import copy
import numpy as np
import pandas
import warnings

class perceptron:
	def __init__(self, learning_rate=0.5, thr=0):
		self.lr = learning_rate
		self.w = np.array([])
		self.b = 0
		self.thr = thr

	# Signal output function maps value to either 1 or -1 according to threshold
	def signal(self, val):
		if val < self.thr:
			return -1
		else:
			return 1

	# Receives entry and returns predicted class
	def eval(self, entry):
		res = self.w.dot(entry)
		return self.signal(res)

	# Receives training data in a two-dimensional pandas dataframe
	def fit(self, data, outcome, initial='random', max_epochs=100, batch_size=1):
		data = data.values

		# Initialize weights and bias
		n_weights = data.shape[1] + 1
		if initial == 'random':
			self.w = np.random.rand(n_weights)
			self.b = np.random.rand(1)[0]
		elif initial == 'zeros':
			self.w = np.zeros(n_weights)
			self.b = 0
		elif initial == 'ones':
			self.w = np.ones(n_weights)
			self.b = 1
		else:
			raise ValueError('Unknown initial weights')

		if data.shape[0] % batch_size != 0:
			raise ValueError('Number of entries must be divisible by batch size')

		# Update parameters
		for i in range(max_epochs):
			# Used for convergence detection
			last_weights = copy.deepcopy(self.w)
			data_cursor = 0
			while data_cursor < data.shape[0]:
				# Used for batch sizes
				current_weights = copy.deepcopy(self.w)
				for j in range(batch_size):
					entry = data[data_cursor, :]
					entry = np.append(self.b, entry)
					pred = self.eval(entry)
					exp = outcome[data_cursor]
					current_weights = current_weights + self.lr * (exp - pred) * entry
					data_cursor += 1
				self.w = copy.deepcopy(current_weights)
			
			# Evaluate current epoch
			total_correct = 0
			for entr in range(data.shape[0]):
				entry = data[entr, :]
				entry = np.append(self.b, entry)
				pred = self.eval(entry)
				exp = outcome[entr]
				if pred == exp:
					total_correct += 1
			print('Iteration', i, 'Recall:', round(total_correct / data.shape[0], 5))

			if total_correct == data.shape[0]:
				print('Perceptron is optimal')
				break

			elif np.array_equal(self.w, last_weights):
				print('Perceptron converged at epoch', i)
				break

			elif i == max_epochs - 1:
				warnings.warn('Perceptron did not converge')
