import numpy as np


def positional_encoding(context_size, embedding_size):
	position_value = np.concatenate([[
						pos / (1e4**(np.arange(embedding_size)/embedding_size)) 
						for pos in range(context_size)]]
						, axis=1)

	for i in range(embedding_size):
		if i%2 == 0:
			position_value[:,i] = np.sin(position_value[:,i]) * 1e-3
		else:
			position_value[:,i] = np.cos(position_value[:,i]) * 1e-3

	return position_value


def softmax2D(X):
	# is working only with 2D matrices along the first dimenstion
	X  = np.exp(X)
	X /= X.sum(axis=1, keepdims=True)
	return X
