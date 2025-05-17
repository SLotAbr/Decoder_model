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
	def __init__(self, 
		batch_size, 
		vocabulary_size, 
		d_model, 
		context_size, 
		optim_param
	):
		scale = 2 / (vocabulary_size + d_model)
		self.TE_table = np.random.normal( # Gaussian distribution
			0, scale, size = (vocabulary_size, d_model)
		)
		self.batch_size = batch_size
		self.vocabulary_size = vocabulary_size
		self.d_model = d_model
		self.context_size = context_size
		# self.input_field = 0
		self.optim = AdaM(optim_param, batch_size)

	def __call__(self, index_list):
		# [context_indexes] -> [Context; Vocabulary]
		# [Batch; context_indexes] -> [Batch; Context; Vocabulary]
		self.input_indexes = index_list
		return self.TE_table[index_list]

	def update_weights(self, dX, dTE_linear):
		# The deravative have 2 parts: from the model tail 
		# (accumulates signal for the chosen vocabulary items) 
		# and its head (linear projection)

		dTE = np.zeros(
			(self.batch_size, self.vocabulary_size, self.d_model)
		)
		for b in range(self.batch_size):
			for c in range(self.context_size):
				dTE[b][self.input_indexes[b][c]] += dX[b][c]

		dTE += dTE_linear
		
		# The optimizer receives the sum of batch derivatives and 
		# normalizes them later
		self.TE_table = self.optim.weights_update(
			self.TE_table,
			# [B; V; d_model] -> [V; d_model]
			dTE.sum(axis=0)
		)

	def linear(self, x):
		'''
		Projects hidden dimension to vocabulary_size. When paired with 
		softmax, it allows to get output token probabilities

		[Batch; Context; d_model] -> [Batch; Context; Vocabulary]
		[Context; d_model] -> [Context; Vocabulary]
		'''
		self.input_field = x
		return x@self.TE_table.T

	def linear_backward(self, dl):
		dx = dl@self.TE_table
		# ( input_field.T@dl ).T
		dw = np.moveaxis(
				np.moveaxis(self.input_field, -1, -2) @ dl,
				-1, -2
			)
		return dx, dw

	def save_weights(self, path):
		with open(path, 'wb') as f:
			pickle.dump([self.TE_table, self.optim], f)

	def restore_weights(self, path):
		with open(path, 'rb') as f:
			self.TE_table, self.optim = pickle.load(f)


class Linear:
	def __init__(self, 
		hidden_units, 
		number_of_neurons, 
		optim_param, 
		batch_size, 
		weight_decay
	):
		scale = 2 / (hidden_units + number_of_neurons)
		self.W = np.random.normal( # Gaussian distribution
			0, scale, size = (hidden_units, number_of_neurons)
		)
		self.b = np.zeros(number_of_neurons)
		# self.input_field = 0 # Memory for backpropagation
		self.w_optim = AdaM(optim_param, batch_size, weight_decay)
		self.b_optim = AdaM(optim_param, batch_size)

	def __call__(self, x):
		self.input_field = x
		return (x @ self.W + self.b) #np.dot(x, w) + b

	def backward(self, dl):
		# input_field.T @ dl
		dw = np.moveaxis(self.input_field, -1, -2) @ dl
		db = dl.sum(axis=1)

		# [Batch; *W.shape] -> [*W.shape]
		# [Batch; *b.shape] -> [*b.shape]
		self.W = self.w_optim.weights_update(self.W, dw.sum(axis=0))
		self.b = self.b_optim.weights_update(self.b, db.sum(axis=0))

		return dl @ self.W.T # dl/dx

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


class Dropout:
	def __init__(self, input_shape, drop_prob):
		assert 0 <= drop_prob < 1
		# (B, C, d_model) or (B, C, C)
		self.input_shape = input_shape
		self.drop_prob = drop_prob
		self.retention_prob = 1 - drop_prob

	def __call__(self, x, phase="train"):
		if phase == "train":
			self.mask = np.random.choice(
				[0, 1], 
				size = self.input_shape, 
				p = [self.drop_prob, self.retention_prob],
				# replace = True
			)
			x *= self.mask
		else: # phase == "eval"
			x *= self.retention_prob
		return x

	def backward(self, dl):
		return dl * self.mask


class LayerNormalization:
	# The initialization is useless without additional variables such
	# as betta and gamma
	def __init__(self, context_size):
		self.context_size = context_size

	def __call__(self, x, phase='train'):
		# if phase == 'train':
		# 	context_size = self.context_size
		# else: # phase == 'eval'
		# 	context_size = x.shape[0]

		x_mean = x.mean(axis=-1, keepdims=True)
		self.x_var = x.var(axis=-1, keepdims=True)
		return (x-x_mean) / np.sqrt(self.x_var+1e-12)

	def backward(self, dl):
		l_mean = dl.mean(axis=-1, keepdims=True)
		return (dl-l_mean) / np.sqrt(self.x_var+1e-12)


class MH_attention_mechanism:
	"""
	Causally masked multihead self-attention
	"""
	def __init__(self, 
		batch_size, 
		context_size, 
		d_model, 
		H, 
		drop_prob = None
	):
		self.d_k = 1 / np.sqrt(d_model/H)
		# self.context_size = context_size
		self.H = H
		self.backward_mask = np.tril(
			np.ones((batch_size, context_size, context_size))
		)
		self.mask = self.backward_mask == 0
		if drop_prob:
			self.att_dropout = []
			for h in range(H):
				self.att_dropout.append(Dropout(
					(batch_size, context_size, context_size), drop_prob
				))
	
	def __call__(self, x, phase='train'):
		self.Q, self.K, self.V = np.split(x, 3, axis=-1)
		self.Q = np.split(self.Q, self.H, axis=-1)
		self.K = np.split(self.K, self.H, axis=-1)
		self.V = np.split(self.V, self.H, axis=-1)

		context_size = x.shape[-2] # <= train context size

		if phase == 'train':
			mask = self.mask
		else: # phase == 'eval'
			# there are no batches for evaluation
			mask = self.mask[0][:context_size,:context_size]

		# soft weights
		self.S = [None for h in range(self.H)]
		scores = [None for h in range(self.H)]

		# https://docs.python.org/3/library/multiprocessing.html
		for h in range(self.H):
			# Q @ K.T / K.shape[-1]
			self.S[h] = self.Q[h] @ np.moveaxis(self.K[h], -1, -2) * self.d_k

			self.S[h][mask] = -1e12
			self.S[h] = softmax(self.S[h])
			self.S[h] = self.att_dropout[h](self.S[h], phase)

			scores[h] = self.S[h] @ self.V[h]

		return np.concatenate(scores, axis=-1)

	def backward(self, dl):
		dScores = np.split(dl, self.H, axis=-1)
		dQ = [None for h in range(self.H)]
		dK = [None for h in range(self.H)]
		dV = [None for h in range(self.H)]

		for h in range(self.H):
			# S.T @ dl
			dV[h] = np.moveaxis(self.S[h], -1, -2) @ dScores[h]

			# dl @ V.T
			dScores[h] = dScores[h] @ np.moveaxis(self.V[h], -1, -2)
			dScores[h] = dScores[h] * self.backward_mask
			dScores[h] = dScores[h] @ (self.S[h]*(1-self.S[h]))
			dScores[h] = self.att_dropout[h].backward(dScores[h])

			# ( Q.T @ dl / Q.shape[-1] ).T
			dK[h] = np.moveaxis(
				np.moveaxis(self.Q[h], -1, -2) @ dScores[h] * self.d_k,
				-1, -2
			)
			dQ[h] = dScores[h] @ self.K[h] * self.d_k

		dQ = np.concatenate(dQ, axis=-1)
		dK = np.concatenate(dK, axis=-1)
		dV = np.concatenate(dV, axis=-1)

		return np.concatenate([dQ, dK, dV], axis=-1)
