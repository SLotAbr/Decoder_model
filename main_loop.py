import numpy as np
import os
import pickle
from models import Decoder_model


save_folder = 'parameters/'
source = open('input.txt', 'r').read()
alphabet = list(set(source))
source_lenght, vocabulary_size = len(source), len(alphabet)
letter_transform = { letter:i for i, letter in enumerate(alphabet) }
indexes_transform = { i:letter for i, letter in enumerate(alphabet) }
# if some files will be found in "save_folder" folder,
# they will be used as the last checkpoint
if len(os.listdir(path=save_folder))!=0:
	with open(save_folder+'model_decription.pkl', 'rb') as f:
		context_size, d_model, H, optim_param = pickle.load(f)
	with open(save_folder+'iteration_param.pkl', 'rb') as f:
		s, step_num, loss = pickle.load(f)

	model = Decoder_model(context_size, vocabulary_size, d_model, H, optim_param)
	model.restore_parameters(save_folder)
else:
	s, step_num, loss = 0, 0, 0
	context_size, d_model, H = 128, 128, 2
	optim_param = [1e-6, 0.9, 0.98] # lr, b1, b2
	with open(save_folder+'model_decription.pkl', 'wb') as f:
		pickle.dump([context_size, d_model, H, optim_param], f)
	model = Decoder_model(context_size, vocabulary_size, d_model, H, optim_param)

threshold = 45000
checkpoint = step_num
checkpoint_loss = 1e12
print('preparation\'s complete!')

while True:
	if s+context_size+1 >= source_lenght or step_num == 0: s=0

	index_list = [letter_transform[letter] for letter in source[s:s+context_size]]
	target_list = [letter_transform[letter] for letter in source[s+1:s+context_size+1]]

	loss_value = model.forward(index_list,target_list, phase='train')
	loss = loss * 0.999 + loss_value * 0.001
	model.backward()

	if step_num%1000==0:
		model.save_parameters(save_folder)
		with open(save_folder+'iteration_param.pkl', 'wb') as f:
			pickle.dump([s, step_num, loss], f)

		index_list=[np.random.randint(0, vocabulary_size)]
		target_list=[]
		for l in range(127):
			index_list.append(model.forward(index_list,target_list, phase='eval'))
		
		text_example = ''.join(indexes_transform[i] for i in index_list)
		print('--------\n %s \n--------' % (text_example, ))
		print('iter %d, loss: %f, lr: %g' % (step_num, loss, lr))
	
	if loss < checkpoint_loss:
		checkpoint = step_num
		checkpoint_loss = loss
	elif (step_num - checkpoint) >= threshold:
		lr/= 1e1
		model.change_lr(lr)

	step_num += 1
	s += context_size
