import numpy as np
import os
import pickle
from models import Decoder_model


save_folder = 'parameters/'
if not os.path.exists(save_folder):
	os.mkdir(save_folder)

# if some files will be found in "save_folder" folder,
# they will be used as the last checkpoint
if len(os.listdir(path=save_folder))!=0:
	with open(save_folder+'text_decription.pkl', 'rb') as f:
		source, alphabet, \
		source_lenght, vocabulary_size, \
		letter_transform, indexes_transform = pickle.load(f)
	with open(save_folder+'model_decription.pkl', 'rb') as f:
		context_size, d_model, H, N, optim_param = pickle.load(f)
	with open(save_folder+'iteration_param.pkl', 'rb') as f:
		s, step_num, loss, lr = pickle.load(f)

	model = Decoder_model(context_size, vocabulary_size, d_model, H, N, optim_param)
	model.restore_parameters(save_folder)
else:
	s, step_num, loss = 0, 0, 0
	context_size, d_model, H, N = 32, 64, 2, 3
	lr = 1e-6
	optim_param = [lr, 0.9, 0.98] # lr, b1, b2

	source = open('input.txt', 'r').read()
	alphabet = list(set(source))
	source_lenght, vocabulary_size = len(source), len(alphabet)
	letter_transform = { letter:i for i, letter in enumerate(alphabet) }
	indexes_transform = { i:letter for i, letter in enumerate(alphabet) }
	print('vocabulary_size: ', vocabulary_size)

	with open(save_folder+'text_decription.pkl', 'wb') as f:
		pickle.dump([source, alphabet,
					 source_lenght, 
					 vocabulary_size, 
					 letter_transform, 
					 indexes_transform], f)
	with open(save_folder+'model_decription.pkl', 'wb') as f:
		pickle.dump([context_size, d_model, H, N, optim_param], f)
	model = Decoder_model(context_size, vocabulary_size, d_model, H, N, optim_param)

threshold = 30000
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
			pickle.dump([s, step_num, loss, lr], f)

		index_list=[np.random.randint(0, vocabulary_size)]
		target_list=[]
		for l in range(31):
			index_list.append(model.forward(index_list,target_list, phase='eval'))
		
		text_example = ' '.join(indexes_transform[i] for i in index_list)
		print('--------\n %s \n--------' % (text_example, ))
		print('iter %d, loss: %f, lr: %g' % (step_num, loss, lr))
	
	if step_num > 10000: # We'll change loss only after 1e4 iteration
		if loss <= checkpoint_loss:
			checkpoint = step_num
			checkpoint_loss = loss
		elif (step_num - checkpoint) > threshold:
			lr/= 10
			model.change_lr(lr)
			print('lr has been reduced. New value: ', lr)
			checkpoint = step_num
			checkpoint_loss = loss

	step_num += 1
	s += context_size
