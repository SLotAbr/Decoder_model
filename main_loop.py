import numpy as np
import os
import pickle
from models import Decoder_model


save_folder = 'parameters/'
if not os.path.exists(save_folder):
	os.mkdir(save_folder)
# TODO: remove it after code review
checkpoint_files = os.listdir(path=save_folder)
if len(checkpoint_files) != 0:
    for file in checkpoint_files:
        os.remove(save_folder + file)

# Load train data from checkpoint
if len(os.listdir(path=save_folder)) != 0:
	# with open(save_folder+'text_decription.pkl', 'rb') as f:
	# 	source, alphabet, \
	# 	source_lenght, vocabulary_size, \
	# 	letter_transform, indexes_transform = pickle.load(f)
	# with open(save_folder+'model_decription.pkl', 'rb') as f:
	# 	context_size, d_model, H, N, optim_param, weight_decay = pickle.load(f)
	# with open(save_folder+'iteration_param.pkl', 'rb') as f:
	# 	data_lenght, step_num, train_i, loss, lr = pickle.load(f)

	# model = Decoder_model(
	# 	context_size, vocabulary_size, d_model, H, N, optim_param, weight_decay
	# )
	# model.restore_parameters(save_folder)
	pass
else:
	epoch, batch, loss = -1, 0, 0
	context_size, d_model, H, N = 64, 64, 2, 4 # 32, 64, 2, 3
	batch_size = 16
	drop_prob = 0.1
	weight_decay = 0.1

	source = open('input.txt', 'r').read()
	alphabet = list(set(source))
	source_lenght, vocabulary_size = len(source), len(alphabet)
	letter_transform = { letter:i for i, letter in enumerate(alphabet) }
	indexes_transform = { i:letter for i, letter in enumerate(alphabet) }

	train_index, train_target = [], []
	for s in range(0, source_lenght-context_size-1, context_size):
		train_index.append([
			letter_transform[letter] 
				for letter in source[s:s+context_size]
		])
		train_target.append([
			letter_transform[letter] 
				for letter in source[s+1:s+context_size+1]
		])
	# data_lenght == data_samples
	data_samples = len(train_index)
	batch_num = data_samples // batch_size

	lr_max = (d_model ** (-0.5)) * (data_samples ** (-0.5))
	# T_warmup = 100000
	# lr = lr_max / T_warmup
	lr = lr_max
	optim_param = [lr, 0.9, 0.95] # lr, b1, b2

	# with open(save_folder+'text_decription.pkl', 'wb') as f:
	# 	pickle.dump([source, alphabet,
	# 				 source_lenght, 
	# 				 vocabulary_size, 
	# 				 letter_transform, 
	# 				 indexes_transform], f)
	# with open(save_folder+'model_decription.pkl', 'wb') as f:
	# 	pickle.dump(
	# 		[context_size, d_model, H, N, optim_param, weight_decay], f
	# 	)

	model = Decoder_model(
		batch_size, 
		context_size, 
		vocabulary_size, 
		d_model, 
		drop_prob,
		H, N, 
		optim_param, 
		weight_decay
	)

print("#"*72)
print("#  Train data stats")
print(
	"#  Data samples: %d, batch_size: %d, batches: %d" % \
		(data_samples, batch_size, batch_num)
)
print("#  Vocabulary size: %d, overall tokens: %d" % \
	(vocabulary_size, data_samples * context_size)
)
print("#"*72)

while True:
	if batch%batch_num == 0:
		train_batch_index, train_batch_target = [], []
		batch_sequence = np.random.choice( # shuffled data sample indexes
			data_samples, 
			size=data_samples, 
			replace=False
		)
		for bi in range(0, data_samples-batch_size+1, batch_size):
			batch_idx = batch_sequence[bi:bi+batch_size]
			train_batch_index.append(
				[train_index[x] for x in batch_idx]
			)
			train_batch_target.append(
				[train_target[x] for x in batch_idx]
			)
		epoch += 1
		batch = 0

	loss_value = model.forward(
		train_batch_index[batch],
		train_batch_target[batch], 
		phase='train'
	)
	loss = loss * 0.999 + loss_value * 0.001
	model.backward()

	if batch%1000 == 0:
		# model.save_parameters(save_folder)
		# with open(save_folder+'iteration_param.pkl', 'wb') as f:
		# 	pickle.dump(
		# 		[data_lenght, step_num, train_i, loss, lr], f
		# 	)

		eval_sample = ''.join(
			indexes_transform[i] for i in model.evaluation()
		)
		print('--------\n %s \n--------' % (eval_sample, ))
		print(
			'epoch: %d, iter: %d/%d, loss: %f, lr: %g' \
						% (epoch, batch, batch_num, loss, lr)
		)

		# print('max and min weight values for a random linear: ', 
		# 	round(np.max(model.W_FC1[2].W), 4),
		# 	round(np.min(model.W_FC1[2].W), 4)
		# )

	# if step_num <= lr_decay_threshold:
	# 	lr += lr_step
	# elif lr <= target_lr/100:
	# 	pass
	# else:
	# 	lr -= lr_step
	# model.change_lr(lr)

	batch  += 1
