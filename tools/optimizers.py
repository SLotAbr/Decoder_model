import numpy as np


class AdaM:
	def __init__(self, optim_param, batch_size, weight_decay=None):
		self.lr = optim_param[0]
		self.b1 = optim_param[1]
		self.b2 = optim_param[2]
		self.batch_size = batch_size
		self.weight_decay = weight_decay
		self.m = 0
		self.v = 0

	def weights_update(self, w, dw, clip_threshold=1.0):
		self.m = (1-self.b1)*dw + self.b1*self.m
		self.v = (1-self.b2)*(dw**2) + self.b2*self.v
		# print('m:\n',self.m)
		# print('v:\n',self.v)
		grad = self.m/np.sqrt(self.v + 1e-9)
		grad_norm = np.sqrt(np.sum(np.pow(grad, 2)))
		# print(grad_norm)
		if grad_norm > clip_threshold:
			grad /= grad_norm

		if self.weight_decay:
			return (1 - self.lr*self.weight_decay) * w - (self.lr / self.batch_size * grad)
		else:
			return w - self.lr / self.batch_size * grad
