# evaluator.py
# Madison Clark-Turner
# 10/18/2017

from __future__ import print_function

import tensorflow as tf 
import numpy as np

# model structure
from dqn_model import *

# file io
from input_pipeline import *
import os
from os.path import isfile, join

# helper methods
import sys
from datetime import datetime
import re

NUM_EPOCHS = 80000
NUM_ITER = 4

verbose = True
action_accuracy = True
reaction_accuracy = True

if __name__ == '__main__':
	#################################
	# Input params
	#################################

	ts = datetime.now()
	print("time start: ", ts)
	graphbuild = [0]*TOTAL_PARAMS
	if(len(sys.argv) > 1):
		graphbuild[int(sys.argv[1])] = 1
	else:
		graphbuild = [1]*TOTAL_PARAMS
	graphbuild = [1,1,1]
	num_params = np.sum(graphbuild)

	#################################
	# Generate Model
	#################################

	dqn = DQNModel(graphbuild=graphbuild, batch_size=BATCH_SIZE)

	#################################
	# Train Model
	#################################

	# generate list of all tfrecords for evaluation
	path = dqn.params.test_dir
	filenames = [path+f for f in os.listdir(path) if isfile(join(path, f))]

	coord = tf.train.Coordinator()

	'''
	sequence length - slen
	sequence length prime- slen_pr
	image raw - i
	points raw - p
	audio raw - a
	previous action - pl
	action - l
	image raw prime - i_pr
	points raw prime - p_pr
	audio raw prime - a_pr
	file identifier - n_id
	'''
	# read records from files into tensors	
	slen, slen_pr, i, p, a, pl, l, i_pr, p_pr, a_pr, n_id = input_pipeline(filenames)
	l = tf.squeeze(l, [1])

	dqn.sess.run(tf.local_variables_initializer())
	threads = tf.train.start_queue_runners(coord=coord, sess=dqn.sess)

	#######################
	## TESTING
	#######################
	
	print("BEGIN TESTING")
	good, failed = [], []
	stored_names = []
			
	total_acc = 0
	for iteration in range(NUM_ITER):
		print("iteration: ", str(iteration)+'/'+str(NUM_ITER))

		n_seq, n_seq2, img_data, pnt_data, aud_data, num_prompts, label_data, img_data2, \
		  pnt_data2, aud_data2, names = dqn.sess.run([slen, slen_pr, i, p, a, pl, l, \
		  i_pr, p_pr, a_pr, n_id])

		partitions_1 = np.zeros((BATCH_SIZE, np.max(n_seq)))
		
		for x in range(BATCH_SIZE):
			if(np.max(n_seq) > 0):
				v = n_seq[x]-1
				if v < 0:
					v = 0
				partitions_1[x][v] = 1

		# evaluate tfrecord
		q_vals, is_correct = dqn.sess.run([dqn.generate_q_values, dqn.correct_pred], feed_dict={
			dqn.seq_length_ph: n_seq, 
			dqn.img_ph: img_data, 
			dqn.pnt_ph: pnt_data, 
			dqn.aud_ph: aud_data, 
			dqn.y_ph: label_data
			,dqn.partitions_ph: partitions_1
			,dqn.train_ph: False
			,dqn.prompts_ph: num_prompts
			})

		if(verbose):
			print("name:", names)
			print("pred:", q_vals)
			print("labl:", is_correct)
		
		for t in range(len(is_correct)):
			tup = [names[t], num_prompts[0], label_data[t], q_vals[t]]

			if(names[t] not in stored_names):
				if(is_correct[t]):
					good.append(tup)
				else:
					failed.append(tup)
				
			stored_names.append(names[t])
	
	if(verbose):
		# print which files were correct and which files were wrong
		print("Name\t\tp_t\tLabel\tPred:")
		
		print("good:")
		for p in good:
			print(str(p[0])+'\t\t'+str(p[1])+'\t'+str(p[2])+'\t'+str(np.argmax(p[3])))

		print("failed:")
		for p in failed:
			print(str(p[0])+'\t\t'+str(p[1])+'\t'+str(p[2])+'\t'+str(np.argmax(p[3])))
		
	if(action_accuracy):
		# print the accuracies for the system and each of the actions
		bins = [[0,0],[0,0],[0,0]]

		for p in good:
			action = np.argmax(p[3])
			bins[action][0]+=1

		for p in failed:
			action = np.argmax(p[3])
			bins[action][1]+=1
		print()
		print("TOTAL Accuracy:", (len(good))/float(len(good)+ len(failed)))

		if(sum(bins[0])):
			print("PMT Accuracy: ", bins[0][0]/float(bins[0][0]+ bins[0][1]), 
					str(bins[0][0])+'/'+str(bins[0][0]+ bins[0][1]))
		else:
			print("No PMT observed")

		if(sum(bins[1])):
			print("REW Accuracy: ", bins[1][0]/float(bins[1][0]+ bins[1][1]), 
					str(bins[1][0])+'/'+str(bins[1][0]+ bins[1][1]))
		else:
			print("No REW observed")

		if(sum(bins[2])):
			print("END Accuracy: ", bins[2][0]/float(bins[2][0]+ bins[2][1]), 
					str(bins[2][0])+'/'+str(bins[2][0]+ bins[2][1]))
		else:
			print("No END observed")

	if(reaction_accuracy):
		# print the accuracies for each of the reactions
		bins = {'z':[0,0], 'g':[0,0],'a':[0,0], 'zg':[0,0], 'za':[0,0], 'ga':[0,0],'zg':[0,0], 'zga':[0,0],'none': [0,0]}
				
		for p in good:
			for k in bins:
				act = np.argmax(p[3])

				name_has_reaction = re.match(r".*"+k+"\d.*", p[0][5:])

				if(name_has_reaction and (act >=1 )):
					bins[k][0] +=1

		for p in failed:
			for k in bins:
				act = np.argmax(p[3])

				name_has_reaction = re.match(r".*"+k+"\d.*", p[0][5:])

				if(name_has_reaction and (act >=1 )):
					bins[k][1] +=1
		print()
		for k in bins:
			if(sum(bins[k])):
				print(k+" Accuracy: ", bins[k][0]/float(bins[k][0]+bins[k][1]))	
			else:
				print(k+" not observed")
