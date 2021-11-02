import math
import numpy as np
import pandas
import statistics

# Get value distribution
def get_val_dist(vals):
	val_dist = {}
	for val in vals:
		try:
			val_dist[val] += 1
		except KeyError:
			val_dist[val] = 1
	return val_dist

# Calculate the entropy of a set
def entropy(vals):
	val_dist = get_val_dist(vals)
	val_dist = list(val_dist.values())
	res = 0
	for val in val_dist:
		p_i = val / sum(val_dist)
		res -= p_i * math.log(p_i, 2)
	return res

# Calculate the multiple attribute entropy
def mult_entropy(df, var_1, var_2):
	res = 0
	val_dist = get_val_dist(df[var_1].values)
	for val in val_dist.keys():
		p_i = val_dist[val] / sum(list(val_dist.values()))
		c_vals = df.loc[df[var_1] == val][var_2].values
		res += p_i * entropy(c_vals)
	return res

# Calculate the information gain
def info_gain(df, var_1, var_2):
	return entropy(df[var_2].values) - mult_entropy(df, var_1, var_2)

def subdivide_data(df, var, val):
	return df.loc[df[var] == val]

def decision_tree(df, class_var):
	d_vars = list(df.columns)
	tree = {}

	# Estimate the root
	max_info_gain = info_gain(df, class_var, d_vars[0])
	root = d_vars[0]
	for var in d_vars:
		if var == class_var:
			continue
		c_info_gain = info_gain(df, class_var, var)
		if c_info_gain > max_info_gain:
			max_info_gain = c_info_gain
			root = var

	tree['node'] = root
	tree['is_leaf'] = False
	tree['children'] = {}
	for val in list(set(df[root].values)):
		c_data = subdivide_data(df, root, val)
		c_data = c_data.drop(columns=[root])
		
		# If there are no remaining variables besides class variable or there is only one class
		if len(c_data.columns) == 1 or len(set(c_data[class_var].values)) == 1:
			tree['children'][val] = {'is_leaf': True, 'class': statistics.mode(c_data[class_var].values)}

		else:
			tree['children'][val] = decision_tree(c_data, class_var)
	
	return tree
