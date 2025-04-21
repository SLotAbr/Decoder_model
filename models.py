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
	def __init__(self, context_size, vocabulary_size, d_model, H, N, optim_param):
		self.TE = nn.token_embedding(vocabulary_size, d_model, context_size, optim_param)
		self.PE = positional_encoding(context_size, d_model)

		self.W_attention = [None for n in range(N)]
		self.MH_attention = [None for n in range(N)]
		self.W_heads_projection = [None for n in range(N)]
		self.Attention_LayerNorm = [None for n in range(N)]

		self.W_FC1 = [None for n in range(N)]
		self.activation = [None for n in range(N)]
		self.W_FC2 = [None for n in range(N)]
		self.FC_LayerNorm = [None for n in range(N)]
		for n in range(N):
			self.W_attention[n] = nn.linear(d_model, d_model*3, optim_param)
			self.MH_attention[n] = nn.MH_attention_mechanism(context_size, d_model, H)
			self.W_heads_projection[n] = nn.linear(d_model, d_model, optim_param)
			self.Attention_LayerNorm[n] = nn.LayerNormalization(context_size)

			self.W_FC1[n] = nn.linear(d_model, d_model*4, optim_param)
			self.activation[n] = nn.ReLU()
			self.W_FC2[n] = nn.linear(d_model*4, d_model, optim_param)
			self.FC_LayerNorm[n] = nn.LayerNormalization(context_size)

		self.Output_token_probabilities = None
		self.context_size = context_size
		self.N = N
		self.vocabulary_size = vocabulary_size
		self.residual_backprop = np.ones((context_size, d_model))

	def save_parameters(self, folder):
		# The folder must ends on '/'!
		self.TE.save_weights(folder+'TE_param.pkl')

		for n in range(self.N):
			self.W_attention[n].save_weights(folder+'W_attention_param{}.pkl'.format(str(n)))
			self.W_heads_projection[n].save_weights\
						(folder+'W_heads_projection_param{}.pkl'.format(str(n)))
			self.W_FC1[n].save_weights(folder+'W_FC1_param{}.pkl'.format(str(n)))
			self.W_FC2[n].save_weights(folder+'W_FC2_param{}.pkl'.format(str(n)))

	def restore_parameters(self, folder):
		self.TE.restore_weights(folder+'TE_param.pkl')

		for n in range(self.N):
			self.W_attention[n].restore_weights(folder+'W_attention_param{}.pkl'.format(str(n)))
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


	def forward(self, index_list, target_list, phase='train'):
		# index_list and target_list must be 1D array
		# We use target_list only during train phase - so, 
		# 	we can give empty target_list during eval phase
		assert len(index_list) <= self.context_size,\
			"This implementation does not support sequences bigger than train context_size"
		# I remove it later, when the eficcient evaluation phase will complete
		self.target_list = target_list

		# Output matrix have the same shape during training
		X = self.TE(index_list)
		if phase=='train':
			X += self.PE
			context_size = self.context_size
		else:
			X += self.PE[:len(index_list)]
			context_size = X.shape[0]

		for n in range(self.N):
			X_sublayer = self.W_attention[n](X)
			X_sublayer = self.MH_attention[n](X_sublayer, phase)
			X_sublayer = self.W_heads_projection[n](X_sublayer)
			X = self.Attention_LayerNorm[n](X + X_sublayer, phase)

			X_sublayer = self.W_FC1[n](X)
			X_sublayer = self.activation[n](X_sublayer)
			X_sublayer = self.W_FC2[n](X_sublayer)
			X = self.FC_LayerNorm[n](X + X_sublayer, phase)

		X = self.TE.linear(X)
		self.Output_token_probabilities = softmax(X)

		if phase =='train':
			loss_value=0
			for i in range(len(target_list)):
				loss_value -= np.log(self.Output_token_probabilities[i][target_list[i]])
			return loss_value
		else: # phase == 'eval'
			## top-1 token probability
			# return np.argmax(self.Output_token_probabilities[-1]) #, axis=1)
			# return np.random.choice(range(self.vocabulary_size), \
			# 	p=self.Output_token_probabilities[-1].ravel())
			## top-k token probabilities
			k = 5
			ixs = np.argpartition(self.Output_token_probabilities[-1], -k)[-k:]
			probs = softmax(self.Output_token_probabilities[-1][ixs])
			return np.random.choice(ixs, p=probs)

	def backward(self):
		dl = self.Output_token_probabilities
		for i in range(len(self.target_list)):
			dl[i][self.target_list[i]] -= 1

		dl, TE_grad = self.TE.linear_backward(dl)

		for n in reversed(range(self.N)):
			dl = self.FC_LayerNorm[n].backward(dl)
			dl = self.W_FC2[n].backward(dl)
			dl = self.activation[n].backward(dl)
			dl = self.W_FC1[n].backward(dl)
			dl += self.residual_backprop

			dl = self.Attention_LayerNorm[n].backward(dl)
			dl = self.W_heads_projection[n].backward(dl)
			dl = self.MH_attention[n].backward(dl)
			dl = self.W_attention[n].backward(dl)
			dl += self.residual_backprop

		self.TE.update_weights(dl, TE_grad)
	
	# ???
	def evaluation(self):
		pass
