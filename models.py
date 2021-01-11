'''
One decoder-block model for language modelling. 
Written by Danil Napad (https://github.com/SLotAbr).
BSD License
'''
import numpy as np
import tools.nn as nn
from tools.functions import positional_encoding, string_softmax
from tools.optimizers import AdaM


class Decoder_model:
	def __init__(self, context_size, vocabulary_size, d_model, H, optim_param):
		self.TE = nn.token_embedding(vocabulary_size, d_model, context_size, optim_param)
		self.PE = positional_encoding(context_size, d_model)

		self.W_attention = nn.linear(d_model, d_model*3, optim_param)
		self.MH_attention = nn.MH_attention_mechanism(context_size, d_model, H)
		self.W_heads_projection = nn.linear(d_model, d_model, optim_param)
		self.Attention_LayerNorm = nn.LayerNormalization(context_size)

		self.W_FC1 = nn.linear(d_model, d_model*4, optim_param)
		self.activation = nn.ReLU()
		self.W_FC2 = nn.linear(d_model*4, d_model, optim_param)
		self.FC_LayerNorm = nn.LayerNormalization(context_size)

		self.Output_token_probabilities = 0
		self.context_size = context_size
		self.vocabulary_size = vocabulary_size
		self.residual_backprop = np.ones((context_size, d_model))

	def save_parameters(self, folder):
		# The folder must ends on '/'!
		self.TE.save_weights(folder+'TE_param.pkl')
		self.W_attention.save_weights(folder+'W_attention_param.pkl')
		self.W_heads_projection.save_weights(folder+'W_heads_projection_param.pkl')
		self.W_FC1.save_weights(folder+'W_FC1_param.pkl')
		self.W_FC2.save_weights(folder+'W_FC2_param.pkl')

	def restore_parameters(self, folder):
		self.TE.restore_weights(folder+'TE_param.pkl')
		self.W_attention.restore_weights(folder+'W_attention_param.pkl')
		self.W_heads_projection.restore_weights(folder+'W_heads_projection_param.pkl')
		self.W_FC1.restore_weights(folder+'W_FC1_param.pkl')
		self.W_FC2.restore_weights(folder+'W_FC2_param.pkl')

	def change_lr(self, lr):
		self.TE.optim.lr = lr
		self.W_attention.w_optim.lr = lr
		self.W_attention.b_optim.lr = lr
		self.W_heads_projection.w_optim.lr = lr
		self.W_heads_projection.b_optim.lr = lr
		self.W_FC1.w_optim.lr = lr
		self.W_FC1.b_optim.lr = lr
		self.W_FC2.w_optim.lr = lr
		self.W_FC2.b_optim.lr = lr


	def forward(self, index_list, target_list, phase='train'):
		# index_list and target_list must be 1D array
		# We use target_list only during train phase - so, 
		# 	we can give empty target_list during eval phase
		assert len(index_list)<=self.context_size,\
			"The current realization does not support sequences bigger than train context_size"
		# I remove it later, when the eficcient evaluation phase will complete
		self.target_list = target_list

		# Output matrix have the same shape during training
		X = self.TE(index_list)
		if phase=='train':
			X+= self.PE
			context_size = self.context_size
		else:
			X+= self.PE[:len(index_list)]
			context_size = X.shape[0]

		X_sublayer = self.W_attention(X)
		X_sublayer = self.MH_attention(X_sublayer, phase)
		X_sublayer = self.W_heads_projection(X_sublayer)
		X = self.Attention_LayerNorm(X + X_sublayer, phase)

		X_sublayer = self.W_FC1(X)
		X_sublayer = self.activation(X_sublayer)
		X_sublayer = self.W_FC2(X_sublayer)
		X = self.FC_LayerNorm(X + X_sublayer, phase)

		X = self.TE.linear(X)
		self.Output_token_probabilities = string_softmax(X, context_size)

		if phase=='train':
			loss_value=0
			for i in range(len(target_list)):
				loss_value -= np.log(self.Output_token_probabilities[i][target_list[i]])
			return loss_value
		else:
			# return the indexes with highest probability for tokens in vocabulary
			# return np.argmax(self.Output_token_probabilities[-1], axis=1)
			return np.random.choice(range(self.vocabulary_size), \
				p=self.Output_token_probabilities[-1].ravel())

	def backward(self):
		dl = self.Output_token_probabilities
		for i in range(len(self.target_list)):
			dl[i][self.target_list[i]] -= 1

		dl, TE_grad = self.TE.linear_backward(dl)

		dl = self.FC_LayerNorm.backward(dl)
		dl = self.W_FC2.backward(dl)
		dl = self.activation.backward(dl)
		dl = self.W_FC1.backward(dl)
		dl += self.residual_backprop

		dl = self.Attention_LayerNorm.backward(dl)
		dl = self.W_heads_projection.backward(dl)
		dl = self.MH_attention.backward(dl)
		dl = self.W_attention.backward(dl)
		dl += self.residual_backprop

		self.TE.update_weights(dl, TE_grad)
		
	def evaluation(self):
		pass
