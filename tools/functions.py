import numpy as np


# Mini-batch update:
# Batch dim is required ( [batch_size; context_size, embedding_size] )
def positional_encoding(batch_size, context_size, embedding_size, n=1e4):
	position_value = np.zeros((context_size, embedding_size))
	for pos in range(context_size):
		for i in range(int(embedding_size/2)):
			embedding_item = pos / np.power(n, 2*i/embedding_size)
			position_value[pos, 2*i] 	 = np.sin(embedding_item) 
			position_value[pos, 2*i + 1] = np.cos(embedding_item) 

	return position_value


def softmax(X):
	if len(X.shape) == 1:
		X -= np.max(X)
		X  = np.exp(X)
		X /= X.sum()
	else:
		X -= np.max(X, axis=-1, keepdims=True)
		X  = np.exp(X)
		X /= X.sum(axis=-1, keepdims=True)
		
	return X
