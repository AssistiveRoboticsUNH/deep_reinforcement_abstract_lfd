# might need to put variables into the model somehow
# it would be nice if I can train and execute on a
# model in a separate folder

from __future__ import print_function

from dqn_model_omega import *
from input_pipeline import *

import tensorflow as tf 
import numpy as np

import sys
import os
from os.path import isfile, join
from datetime import datetime
import math, random

import cv2, random

NUM_EPOCHS = 80000
GAMMA = 0.9
ALPHA = 1e-4
NUM_ITER = 50#2500
FOLDS = 1
NUM_REMOVED = 1

TEST_ITER = 10

# alpha or 1e-3 was unsuccesful when not using BN
# see if I can play with bn_lstm parameters to get a better result

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
	# Read contents of TFRecord file
	#################################
	path = "../tfrecords/tfrecords_final_eval/"
	filenames = [f for f in os.listdir(path) if isfile(join(path, f))]
	filenames = [path +x for x in filenames if (x.find("g") >= 0 and x.find("ga") < 0) ]
	filenames.sort()

	testing = filenames

	#################################
	# Generate Model
	#################################
	chkpt = "./omega_2/model.ckpt"
	dqn = DQNModel(graphbuild, batch_size=BATCH_SIZE, learning_rate=ALPHA, filename=chkpt, log_dir="LOG_DIR")
	

	#################################
	# Train Model
	#################################
	
	coord = tf.train.Coordinator()

	slen_t, slen_pr_t, i_t, p_t, a_t, pl_t, l_t, i_pr_t, p_pr_t, a_pr_t, name_t = input_pipeline(testing)
	l_t = tf.squeeze(l_t, [1])

	dqn.sess.run(tf.local_variables_initializer())#initializes batch queue
	threads = tf.train.start_queue_runners(coord=coord, sess=dqn.sess)

	#######################
	## TESTING
	#######################
	
	print("BEGIN TESTING")
	failed, good, total = [], [], NUM_ITER*BATCH_SIZE
	failed_names = []
		
	total_acc = 0
	for iteration in range(NUM_ITER):
		print("iteration: ", iteration)
		n_seq, n_seq2, img_data, pnt_data, aud_data, num_prompts, label_data, img_data2, pnt_data2, aud_data2, names = dqn.sess.run([slen_t, slen_pr_t, i_t, p_t, a_t, pl_t, l_t, i_pr_t, p_pr_t, a_pr_t, name_t])

		print(n_seq, img_data.shape)
		'''
		if(np.max(n_seq) > 129):
			n_seq = np.clip(n_seq, 1, 129)#[129]
			img_data = img_data[:,:129,:]
			pnt_data = pnt_data[:,:129,:]
			aud_data = aud_data[:,:129,:]
		print(n_seq, img_data.shape)
		'''
		partitions_1 = np.zeros((BATCH_SIZE, np.max(n_seq)))
		
		for x in range(BATCH_SIZE):
			if(np.max(n_seq) > 0):
				v = n_seq[x]-1
				if v < 0:
					v = 0
				partitions_1[x][v] = 1

		pred, equ = dqn.sess.run([dqn.pred, dqn.correct_pred], feed_dict={
			dqn.seq_length_ph: n_seq, 
			dqn.img_ph: img_data, 
			dqn.pnt_ph: pnt_data, 
			dqn.aud_ph: aud_data, 
			dqn.y_ph: label_data
			,dqn.partitions_ph: partitions_1
			,dqn.train_ph: False
			,dqn.prompts_ph: num_prompts
			})

		print("name:", names)
		print("pred:", pred)
		print("labl:", equ)

		
		for t in range(len(equ)):
			tup = [names[t], pred[t], label_data[t], num_prompts[0]]
			print(names[t], names[t] not in failed_names)
			if not equ[t] and (names[t] not in failed_names):
				failed.append(tup)
				failed_names.append(names[t])
			elif equ[t] and (names[t] not in failed_names):
				good.append(tup)
				failed_names.append(names[t])
		
		#print("outupt: ", type(units))
		#pnt_data = pnt_data.reshape((1, 40, 80, 80, 3))[0,11,...].astype(np.uint8)
		#print("pnt_data: ", pnt_data.shape, pnt_data.dtype)
		#cv2.imshow("pnts",pnt_data)
		#dqn.plotNNFilter(units)

	
	
	print("failed:")
	for p in failed:
		print(str(p[0]), str(p[3]), str(p[1]), str(np.argmax(p[2])))
		
		#print (str(p[1])+'\t\t'+str(p[2]) + '\t'+str(np.argmax(p[1]))+ '\t'+str(np.argmax(p[2])))

	print("good:")
	for p in good:
		print(str(p[0]), str(p[3]), str(p[1]), str(np.argmax(p[2])))
		
		#print (str(p[1])+'\t\t'+str(p[2]) + '\t'+str(np.argmax(p[1]))+ '\t'+str(np.argmax(p[2])))

	bins = [[0,0],[0,0],[0,0]]
	for p in good:
		bins[np.argmax(p[2])][0]+=1
	for p in failed:
		bins[np.argmax(p[2])][1]+=1

	print("bins", bins)
	#print("cat_2"[:-2].find("_2"))

	print("Prompt Accuracy:", (len(good))/float(len(good)+ len(failed)))

	print("Prompt Accuracy: ", bins[0][0]/float(bins[0][0]+ bins[0][1]))
	print("Reward Accuracy: ", bins[1][0]/float(bins[1][0]+ bins[1][1]))
	print("Abort Accuracy: ", bins[2][0]/float(bins[2][0]+ bins[2][1]))

	bins = {'z':[0,0], 'g':[0,0],'a':[0,0], 'zg':[0,0], 'ga':[0,0],'zg':[0,0], 'zga':[0,0],'none': [0,0]}
	for p in good:
		for k in bins:
			act = np.argmax(p[2])
			#print(k, act, p[0], p[0].find(k))
			if(p[0][5:].find(k) >= 0 and (act >=1 )):
				bins[k][0] +=1
	for p in failed:
		for k in bins:
			act = np.argmax(p[2])
			if(p[0][5:].find(k) >= 0 and (act >= 1)):
				bins[k][1] +=1

	print("bins", bins)
	for k in bins:
		print(k+" Accuracy: ", bins[k][0]/float(bins[k][0]+bins[k][1]))
	

	print(len(good)+len(failed), len(failed_names))
	print(failed_names)
	
