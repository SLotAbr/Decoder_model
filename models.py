'''
Decoder model for language modelling. 
Written by Danil Napad (https://github.com/SLotAbr).
BSD License
'''
import numpy as np
import tools.nn as nn
from tools.functions import positional_encoding, softmax
from tools.optimizers import AdaM


class Decoder_model:
	def __init__(self, 
		batch_size, 
		context_size, 
		vocabulary_size, 
		d_model, 
		drop_prob, 
		H, 
		N, 
		optim_param,
		weight_decay
	):
		self.TE = nn.token_embedding(
			batch_size, vocabulary_size, d_model, context_size, optim_param
		)
		self.PE = positional_encoding(context_size, d_model)
		self.embedding_dropout = nn.Dropout(
			(batch_size, context_size, d_model), drop_prob
		)

		self.Attention_LayerNorm = [None for n in range(N)]
		self.W_attention = [None for n in range(N)]
		self.MH_attention = [None for n in range(N)]
		self.W_heads_projection = [None for n in range(N)]
		self.Attention_dropout = [None for n in range(N)]

		self.FC_LayerNorm = [None for n in range(N)]
		self.W_FC1 = [None for n in range(N)]
		self.activation = [None for n in range(N)]
		self.W_FC2 = [None for n in range(N)]
		self.FFN_dropout = [None for n in range(N)]

		for n in range(N):
			self.Attention_LayerNorm[n] = nn.LayerNormalization(context_size)
			self.W_attention[n] = nn.Linear(
				d_model, d_model*3, optim_param, batch_size, weight_decay
			)
			self.MH_attention[n] = nn.MH_attention_mechanism(
				batch_size, context_size, d_model, H, drop_prob
			)
			self.W_heads_projection[n] = nn.Linear(
				d_model, d_model, optim_param, batch_size, weight_decay
			)
			self.Attention_dropout[n] = nn.Dropout(
				(batch_size, context_size, d_model), drop_prob
			)

			self.FC_LayerNorm[n] = nn.LayerNormalization(context_size)
			self.W_FC1[n] = nn.Linear(
				d_model, d_model*4, optim_param, batch_size, weight_decay
			)
			self.activation[n] = nn.ReLU()
			self.W_FC2[n] = nn.Linear(
				d_model*4, d_model, optim_param, batch_size, weight_decay
			)
			self.FFN_dropout[n] = nn.Dropout(
				(batch_size, context_size, d_model), drop_prob
			)

		self.final_LayerNorm = nn.LayerNormalization(context_size)
		self.Output_token_probabilities = None
		self.batch_size = batch_size
		self.context_size = context_size
		self.N = N
		self.vocabulary_size = vocabulary_size

	def save_parameters(self, folder):
		# The folder must end on '/'!
		self.TE.save_weights(folder+'TE_param.pkl')

		for n in range(self.N):
			self.W_attention[n].save_weights(
				folder+'W_attention_param{}.pkl'.format(str(n))
			)
			self.W_heads_projection[n].save_weights\
						(folder+'W_heads_projection_param{}.pkl'.format(str(n)))
			self.W_FC1[n].save_weights(folder+'W_FC1_param{}.pkl'.format(str(n)))
			self.W_FC2[n].save_weights(folder+'W_FC2_param{}.pkl'.format(str(n)))

	def restore_parameters(self, folder):
		self.TE.restore_weights(folder+'TE_param.pkl')

		for n in range(self.N):
			self.W_attention[n].restore_weights(
				folder+'W_attention_param{}.pkl'.format(str(n))
			)
			self.W_heads_projection[n].restore_weights\
						(folder+'W_heads_projection_param{}.pkl'.format(str(n)))
			self.W_FC1[n].restore_weights(folder+'W_FC1_param{}.pkl'.format(str(n)))
			self.W_FC2[n].restore_weights(folder+'W_FC2_param{}.pkl'.format(str(n)))

	def change_lr(self, lr):
		self.TE.optim.lr = lr

		for n in range(self.N):
			self.W_attention[n].w_optim.lr = lr
			self.W_attention[n].b_optim.lr = lr
			self.W_heads_projection[n].w_optim.lr = lr
			self.W_heads_projection[n].b_optim.lr = lr
			self.W_FC1[n].w_optim.lr = lr
			self.W_FC1[n].b_optim.lr = lr
			self.W_FC2[n].w_optim.lr = lr
			self.W_FC2[n].b_optim.lr = lr

	def forward(self, index_list, target_list=None, phase='train'):
		"""
		index_list : 2D list for training and 1D list for evaluation
		target_list: 2D list, is used during train phase only
		"""
		assert len(index_list) <= self.context_size
		self.target_list = target_list

		# [B; C] -> [B; C; d_model]
		# 	 [C] -> [C; d_model]
		X = self.TE(index_list)
		if phase == 'train':
			X += self.PE
		else: # phase == 'eval'
			X += self.PE[:len(index_list)]
		X = self.embedding_dropout(X, phase)

		for n in range(self.N):
			X_sublayer = self.Attention_LayerNorm[n](X, phase)
			X_sublayer = self.W_attention[n](X_sublayer)
			X_sublayer = self.MH_attention[n](X_sublayer, phase) # <- dropout?
			X_sublayer = self.W_heads_projection[n](X_sublayer)
			X_sublayer = self.Attention_dropout[n](X_sublayer, phase)
			X += X_sublayer

			X_sublayer = self.FC_LayerNorm[n](X, phase)
			X_sublayer = self.W_FC1[n](X_sublayer)
			X_sublayer = self.activation[n](X_sublayer)
			X_sublayer = self.W_FC2[n](X_sublayer)
			X_sublayer = self.FFN_dropout[n](X_sublayer, phase)
			X += X_sublayer

		X = self.final_LayerNorm(X)
		X = self.TE.linear(X)
		self.Output_token_probabilities = softmax(X)

		if phase == 'train':
			loss_value=0
			for b in range(self.batch_size):
				for c in range(self.context_size):
					loss_value -= np.log(
						# B, C, V
						self.Output_token_probabilities[b][c][target_list[b][c]]
					)
			loss_value /= self.context_size
			loss_value /= self.batch_size
			# loss_value += lambda * sum(weight ** 2)
			return loss_value
		else: # phase == 'eval'
			## top-k token probabilities
			k = 10
			# C, V
			ixs = np.argpartition(self.Output_token_probabilities[-1], -k)[-k:]
			probs = softmax(self.Output_token_probabilities[-1][ixs])
			return np.random.choice(ixs, p=probs)

	def backward(self):
		dl = self.Output_token_probabilities
		for b in range(self.batch_size):
			for c in range(self.context_size):
				# B, C, V
				dl[b][c][self.target_list[b][c]] -= 1

		dl, TE_grad = self.TE.linear_backward(dl)
		dl = self.final_LayerNorm.backward(dl)

		for n in reversed(range(self.N)):
			dl = self.FFN_dropout[n].backward(dl)
			dl = self.W_FC2[n].backward(dl)
			dl = self.activation[n].backward(dl)
			dl = self.W_FC1[n].backward(dl)
			dl = self.FC_LayerNorm[n].backward(dl)
			dl += 1 # residual backprop: dx/dx

			dl = self.Attention_dropout[n].backward(dl)
			dl = self.W_heads_projection[n].backward(dl)
			dl = self.MH_attention[n].backward(dl)
			dl = self.W_attention[n].backward(dl)
			dl = self.Attention_LayerNorm[n].backward(dl)
			dl += 1 # residual backprop: dx/dx

		dl = self.embedding_dropout.backward(dl)
		self.TE.update_weights(dl, TE_grad)
