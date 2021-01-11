import numpy as np


class AdaM:
	def __init__(self, optim_param):
		self.lr = optim_param[0]
		self.b1 = optim_param[1]
		self.b2 = optim_param[2]
		self.m = 0
		self.v = 0

	def weights_update(self, w, dw):
		self.m = (1-self.b1)*dw + self.b1*self.m
		self.v = (1-self.b2)*(dw**2) + self.b2*self.v
		# print('m:\n',self.m)
		# print('v:\n',self.v)
		return w - self.lr* self.m/np.sqrt(self.v + 1e-9)