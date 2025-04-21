import numpy as np


def positional_encoding(context_size, embedding_size, n=1e4):
	position_value = np.zeros((context_size, embedding_size))
	for pos in range(context_size):
		for i in range(int(embedding_size/2)):
			embedding_item = pos / np.power(n, 2*i/embedding_size)
			position_value[pos, 2*i] 	 = np.sin(embedding_item) 
			position_value[pos, 2*i + 1] = np.cos(embedding_item) 

	return position_value


def softmax(X):
	# is working only with 2D matrices along the first dimenstion
	X  = np.exp(X)
	if len(X.shape) == 1:
		X /= X.sum()
	else:
		X /= X.sum(axis=1, keepdims=True)
	return X
