'''
Layers for NN models (forward and backward passes).
Written by Danil Napad (https://github.com/SLotAbr).
BSD License
'''
# from multiprocessing import Process
import numpy as np
import pickle
from tools.functions import softmax
from tools.optimizers import AdaM as AdaM


class token_embedding:
	def __init__(self, vocabulary_size, d_model, context_size, optim_param):
		# self.TE_table = np.random.randn(vocabulary_size, d_model) * 1e-3
		scale = 2 / (vocabulary_size + d_model)
		self.TE_table = np.random.normal( # Gaussian distribution
			0, scale, size = (vocabulary_size, d_model)
		)
		self.vocabulary_size = vocabulary_size
		self.d_model = d_model
		self.context_size = context_size
		self.input_field = 0
		self.optim = AdaM(optim_param)

	def __call__(self, index_list):
		# form X matrix from tokens indexes
		# We should use 2D array for further concatenation
		self.input_indexes = index_list
		context = [[self.TE_table[j] for j in index_list]]

		return np.concatenate(context, axis=1)

	def update_weights(self, dX, dTE_linear):
		# dTE_linear - the second part of TE derivative
		# TE derivative have 2 parts - so, we'll get it by external source
		dTE = np.zeros((self.vocabulary_size, self.d_model))
		for i in range(self.context_size):
			dTE[self.input_indexes[i]] += dX[i]

		dTE += dTE_linear
		self.TE_table = self.optim.weights_update(self.TE_table, dTE)

	def linear(self, x):
		'''
		using token_embeddings as linear layer with bias=0
		we'll use it for finding out output token probabilities
		:x.shape = [context_size; d_model]
		:output.shape = [context_size; vocabulary_size]
		'''
		self.input_field = x
		return x@self.TE_table.T

	def linear_backward(self, dl):
		# returns derivatives for input signal and TE_table
		return dl@self.TE_table, (self.input_field.T@dl).T

	def save_weights(self, path):
		with open(path, 'wb') as f:
			pickle.dump([self.TE_table, self.optim], f)

	def restore_weights(self, path):
		with open(path, 'rb') as f:
			self.TE_table, self.optim = pickle.load(f)


class Linear:
	def __init__(self, hidden_units, number_of_neurons, optim_param):
		# self.W = np.random.randn(hidden_units, number_of_neurons) * 1e-3
		scale = 2 / (hidden_units + number_of_neurons)
		self.W = np.random.normal( # Gaussian distribution
			0, scale, size = (hidden_units, number_of_neurons)
		)
		self.b = np.zeros(number_of_neurons)
		self.input_field = 0 # Memory for backpropagation
		self.w_optim = AdaM(optim_param)
		self.b_optim = AdaM(optim_param)

	def __call__(self, x):
		self.input_field = x
		return (x @ self.W + self.b) #np.dot(x, w) + b

	def backward(self, dl):
		dw = self.input_field.T @ dl
		db = dl.sum(axis=0)

		# Updating weights
		self.W = self.w_optim.weights_update(self.W, dw)
		self.b = self.b_optim.weights_update(self.b, db)

		# returns dl for previous layers
		return dl @ self.W.T

	def save_weights(self, path):
		with open(path, 'wb') as f:
			pickle.dump([self.W, 
						self.b, 
						self.w_optim, 
						self.b_optim], f)

	def restore_weights(self, path):
		with open(path, 'rb') as f:
			self.W, self.b, self.w_optim, self.b_optim = pickle.load(f)


class ReLU:
	def __call__(self, x):
		result = np.maximum(0, x)
		self.mask = result>0
		return result

	def backward(self, dl):
		return dl * self.mask


class LayerNormalization:
	def __init__(self, context_size):
		self.context_size = context_size

	def __call__(self, x, phase='train'):
		'''
		I'll delete if-else construction and replace it more
		eficient version for evaluation phase later.
		There is the same construction in MH_attention_mechanism (__call__ field)
		'''
		if phase == 'train':
			context_size = self.context_size
		else:
			context_size = x.shape[0]

		x_mean = x.mean(axis=1, keepdims=True)
		self.x_var = x.var(axis=1, keepdims=True)
		return (x-x_mean) / np.sqrt(self.x_var+1e-12)

	def backward(self, dl):
		l_mean = dl.mean(axis=1, keepdims=True)
		return (dl-l_mean) / np.sqrt(self.x_var+1e-12)


class MH_attention_mechanism:
	def __init__(self, context_size, d_model, H):
		self.d_k = 1 / np.sqrt(d_model/H)
		self.context_size = context_size
		self.H = H
		# A matrix with 'True' values above the main diagonal
		# We'll use it later to replace elements in dot product of Q and K
		self.mask = (np.tril(np.ones((context_size, context_size)))==0)
		self.backward_mask = np.tril(np.ones((context_size, context_size)))
		
	def __call__(self, x, phase='train'):
		self.Q, self.K, self.V = np.split(x, 3, axis=1)
		self.Q = np.split(self.Q, self.H, axis=1)
		self.K = np.split(self.K, self.H, axis=1)
		self.V = np.split(self.V, self.H, axis=1)

		# When we generate text ('eval phase'), context_size always different
		if phase == 'train':
			context_size = self.context_size
		else:
			context_size = x.shape[0]

		# Replace it by pre-init fields for faster implementation?
		C = 	 [None for h in range(self.H)]
		self.S = [None for h in range(self.H)]
		Z = 	 [None for h in range(self.H)]

		# https://docs.python.org/3/library/multiprocessing.html
		for h in range(self.H):
			# Attention formula
			C[h] = self.Q[h] @ self.K[h].T * self.d_k

			if phase == 'train':
				C[h][self.mask]=-1e12
			else:
				# We've got different context_size during evaluation
				mask = (np.tril(np.ones((context_size, context_size)))==0)
				C[h][mask] = -1e12

			self.S[h] = softmax(C[h])
			# print('softmax\'s state:\n', self.S[h])
			Z[h] = self.S[h]@self.V[h]
			# print('Z\'s state:\n', Z[h])

		return np.concatenate(Z, axis=1)

	def backward(self, dl):
		dZ = np.split(dl, self.H, axis=1)
		dQ = [None for h in range(self.H)]
		dK = [None for h in range(self.H)]
		dV = [None for h in range(self.H)]

		for h in range(self.H):
			dV[h] = self.S[h].T @ dZ[h]

			dZ[h] = dZ[h]@self.V[h].T
			# We should multiplicate it later in chain-rule, 
			# but there isn't a mistake to do this now
			dZ[h] = dZ[h] * self.backward_mask
			dZ[h] = dZ[h]@ (self.S[h]*(1-self.S[h]))

			dK[h] = (self.Q[h].T@dZ[h] * self.d_k).T
			dQ[h] = dZ[h]@self.K[h] * self.d_k

		dQ = np.concatenate(dQ, axis=1)
		dK = np.concatenate(dK, axis=1)
		dV = np.concatenate(dV, axis=1)

		return np.concatenate([dQ, dK, dV], axis=1)
