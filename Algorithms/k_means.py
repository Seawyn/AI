import copy
import matplotlib.pyplot as plt
import numpy as np
import random

# Calculates the Euclidean Distance between a cluster and a point
def get_distance(p, c):
	res = (p - c) ** 2
	res = sum(res) ** 0.5
	return res


# Randomly initiate new num_c clusters within the scope of the data
def get_init_clusters(points, num_c, init_method='random'):
	new_clusters = []
	
	if init_method == 'random':
		# Get minimum and maximum coordinates for random cluster creation
		min_coords = points.min(axis=0)
		max_coords = points.max(axis=0)

		for i in range(num_c):
			curr_c = []
			for j in range(min_coords.shape[0]):
				curr_c.append(random.uniform(min_coords[j], max_coords[j]))
			new_clusters.append(curr_c)

	elif init_method == 'forgy':
		# Select num_c random points that will act as initial clusters
		c_centers = random.sample([i for i in range(len(points))], num_c)
		for c in c_centers:
			new_clusters.append(points[c, :].tolist())

	else:
		# Randomly assign a cluster to each point and estimate initial clusters accordingly
		assignments = {i: [] for i in range(num_c)}
		for p in range(len(points)):
			cl = random.randint(0, num_c - 1)
			assignments[cl].append(p)
		for c in range(num_c):
			new_c = points[assignments[c], :].mean(axis=0)
			new_clusters.append(new_c.tolist())

	return new_clusters


# Checks for changes in clusters from one iteration to another
# Used for convergence detection
def cluster_changes(clusters, new_clusters, num_c):
	for c in range(num_c):
		if clusters[c]['points'] != new_clusters[c]['points']:
			return True
	return False


# Classify an entry based on the nearest cluster
def classify(clusters, entry):
	min_cluster = 0
	min_dist = get_distance(entry, clusters[0]['center'])
	for c in clusters.keys():
		curr_dist = get_distance(entry, clusters[c]['center'])
		if curr_dist < min_dist:
			min_cluster = c
			min_dist = curr_dist
	return min_cluster


# K-Means Algorithm
def k_means(points, num_c, max_iter=100, init_method='random'):
	init_clusters = get_init_clusters(points, num_c, init_method=init_method)
	clusters = {i: {'center': init_clusters[i], 'points': []} for i in range(num_c)}
	for i in range(max_iter):
		new_clusters = {i: {'center': [], 'points': []} for i in range(num_c)}
		for j in range(points.shape[0]):
			# Estimate nearest cluster
			min_cluster = classify(clusters, points[j, :])
			new_clusters[min_cluster]['points'].append(j)

		# Estimate new cluster centers
		for c in range(num_c):
			if new_clusters[c]['points'] == []:
				new_clusters[c]['center'] = clusters[c]['center']
			else:
				new_clusters[c]['center'] = points[new_clusters[c]['points'], :].mean(axis=0)

		# Check if clusters changed
		if not cluster_changes(clusters, new_clusters, num_c):
			break
		else:
			clusters = copy.deepcopy(new_clusters)
	return new_clusters


# Simple testing routine
def test_k_means():
	initial_centers = np.array([[1, 1], [-2, -1], [2, -3]])
	points = None
	for i in range(250):
		offset = np.zeros(2)
		offset[0] = random.uniform(-2, 2)
		offset[1] = random.uniform(-2, 2)
		cursor = random.randint(0, len(initial_centers) - 1)
		new_p = initial_centers[cursor, :] - offset
		if points is None:
			points = copy.deepcopy(new_p)
		else:
			points = np.vstack([points, new_p])

	# Obtain clusters
	clusters = k_means(points, 3, init_method='forgy')

	# Classify each point
	classifications = []
	for j in range(points.shape[0]):
		classifications.append(classify(clusters, points[j, :]))

	# Extract cluster centers
	c_centers = []
	for key in clusters.keys():
		c_centers.append(clusters[key]['center'])
	c_centers = np.array(c_centers)

	# Plot results
	plt.scatter(points[:, 0], points[:, 1], c=classifications)
	plt.scatter(c_centers[:, 0], c_centers[:, 1], marker='x')
	plt.show()
