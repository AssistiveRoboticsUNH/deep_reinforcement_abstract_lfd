# model_trainer_omega.py
# Madison Clark-Turner
# 10/13/2017

import numpy as np
import tensorflow as tf 
from tensorflow.python.client import timeline

# model structure
from dqn_model import *

# file io
from basic_tfrecord_rw import parse_sequence_example
from input_pipeline import *
import os
from os.path import isfile, join

# helper methods
import sys
from datetime import datetime

GAMMA = 0.9
ALPHA = 1e-6
NUM_ITER = 30000
FOLDS = 1
NUM_REMOVED = 1

TEST_ITER = 50

if __name__ == '__main__':

	ts = datetime.now()
	print("time start: ", str(ts))
	#################################
	# Command-Line Parameters
	#################################
	graphbuild = [0]*TOTAL_PARAMS
	if(len(sys.argv) > 1):
		if (len(sys.argv[1]) == 1 and int(sys.argv[1]) < 3):
			graphbuild[int(sys.argv[1])] = 1
		else:
			print("Usage: python model_trainer_omega.py <args>")
			print("\t0 - only build network with RGB information")
			print("\t1 - only build network with Optical Flow information")
			print("\t2 - only build network with Audio information")
			print("\t(nothing) - build network with all information")
	else:
		graphbuild = [1]*TOTAL_PARAMS

	if(sum(graphbuild) < 3):
		print("#########################")
		print("BUILDING PARTIAL MODEL")
		print("#########################")

	num_params = np.sum(graphbuild)

	#################################
	# Read contents of TFRecord file
	#################################

	# define directory to read files from
	path = "../tfrecords/tfrecords_balanced/"
	
	# generate list of filenames
	filenames = [f for f in os.listdir(path) if isfile(join(path, f))]
	filenames = [path +x for x in filenames ]
	filenames.sort()
		
	#################################
	# Generate Model
	#################################

	# if building model from a checkpoint define location here. Otherwise use empty string ""
	dqn_chkpnt = ""
	dqn = DQNModel(graphbuild, batch_size=BATCH_SIZE, learning_rate=ALPHA, filename=dqn_chkpnt,\
				log_dir="LOG_DIR")

	# if building from checkpoint need to setup dqn_hat variables
	if(dqn_chkpnt != ""):
		dqn.assignVariables()

	#################################
	# Train Model
	#################################
	
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
	
	#initialize all variables
	dqn.sess.run(tf.local_variables_initializer())
	dqn.sess.graph.finalize()
	threads = tf.train.start_queue_runners(coord=coord, sess=dqn.sess)

	print("Num epochs: "+str(NUM_EPOCHS)+", Batch Size: "+str(BATCH_SIZE)+", Num Files: "+\
					str(len(filenames))+", Num iterations: "+str(NUM_ITER))

	for iteration in range(NUM_ITER):
		ts_it = datetime.now()

		#---------------------------------------
		# read a bacth of tfrecords into np arrays
		#---------------------------------------
		n_seq, n_seq2, img_data, pnt_data, aud_data, num_prompts, label_data, img_data2 \
					, pnt_data2, aud_data2, name = dqn.sess.run([slen, slen_pr, i, p, a, pl, l\
					, i_pr, p_pr, a_pr, n_id])
		
		#---------------------------------------
		# generate partitions; used for extracting relevant data from the LSTM layer
		#---------------------------------------
		partitions_1 = np.zeros((BATCH_SIZE, np.max(n_seq)))
		partitions_2 = np.zeros((BATCH_SIZE, np.max(n_seq2)))

		for x in range(BATCH_SIZE):
			
			if(np.max(n_seq) > 0):
				v = n_seq[x]-1
				if v < 0:
					v = 0
				partitions_1[x][v] = 1
			
			if(np.max(n_seq2) > 0):
				v = n_seq2[x]-1
				if v < 0:
					v = 0
				partitions_2[x][v] = 1
		
		#---------------------------------------
		# generate y_i for not terminal states
		#---------------------------------------
		newy = 0
		if(np.max(n_seq2) > 1):
			# if at least on of the input files in the batch is not terminal then we need 
			# to shape and pass the subsequent observation into the network in order to
			# generate a q-value from Q^hat
			img_data2 = set_shape(img_data2, img_dtype)
			pnt_data2 = set_shape(pnt_data2, pnt_dtype)
			aud_data2 = set_shape(aud_data2, aud_dtype)

			vals = {
				dqn.seq_length_ph: n_seq2, 
				dqn.img_ph: img_data2, 
				dqn.pnt_ph: pnt_data2, 
				dqn.aud_ph: aud_data2
				,dqn.partitions_ph: partitions_2
				,dqn.train_ph: False
				,dqn.prompts_ph: np.sign(n_seq2) }

			# get the maxium q-value from q^hat
			newy = dqn.sess.run(dqn.max_q_hat_value, feed_dict=vals)
			# assign the max q-value to the appropriate action
			newy *= np.sign(n_seq2)
		else:
			newy = np.zeros(BATCH_SIZE)
		
		# set up array for y_i and populate appropriately
		r = np.array(label_data)
		
		# reward given for executing the prompt action
		# the reward for abort and reward actions is 1.0
		r[:,0] = r[:,0]*.2 
	
		for j in range(r.shape[0]):
			for v in range(r.shape[1]):
				if r[j][v] != 0:
					if(v < 2):
						r[j][v]+= newy[j] * GAMMA
		
		#---------------------------------------
		# Optimize Network
		#---------------------------------------
		vals = {
			dqn.seq_length_ph: n_seq, 
			dqn.img_ph: img_data,
			dqn.pnt_ph: pnt_data, 
			dqn.aud_ph: aud_data, 
			dqn.y_ph: r,
			dqn.partitions_ph: partitions_1,
			dqn.train_ph: True,
			dqn.prompts_ph: num_prompts
			}

		prep_t = datetime.now() - ts_it
		switch_s = datetime.now()

		# Set vaiables in the DQN to be those of Q and not Q^hat
		dqn.restore_q_hat_vars(dqn.variables_img_main, dqn.variables_img)
		switch_t = datetime.now() - switch_s

		# OPTIMIZE
		run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		run_metadata=tf.RunMetadata()

		opt_s = datetime.now()

		# optimize network
		_ = dqn.sess.run([dqn.optimizer], feed_dict=vals, options=run_options, 
					run_metadata=run_metadata)
		
		# write summary information to file (disabled)
		#summary, _ = dqn.sess.run([dqn.merged_summary, dqn.optimizer], feed_dict=vals, options=run_options, run_metadata=run_metadata)
		#dqn.train_writer.add_run_metadata(run_metadata, 'step%03d' % iteration)

		opt_t = datetime.now() - opt_s
		
		# Store vaiables of Q in temporary data structure
		dqn.restore_q_hat_vars(dqn.variables_img, dqn.variables_img_main)

		#---------------------------------------
		# Gnerate Cross Entropy (disabled for now as it slows the optimization process)
		#---------------------------------------
		'''
		ce_s = datetime.now()
		vals[dqn.train_ph] = False
		ce = dqn.sess.run(dqn.cross_entropy, feed_dict=vals)
		ce_t = datetime.now() - ce_s
		'''

		#---------------------------------------
		# Print Metrics
		#---------------------------------------
		if(iteration%1 == 0):
			# print timing information
			print(iteration, "total_time:", datetime.now()-ts_it, "prep_time:",prep_t, "switch_time:",switch_t, "optimization_time:",opt_t)
		
		if(iteration%100 == 0):
			# evaluate system accuracy on train dataset
			pred = dqn.sess.run(dqn.generate_q_values, feed_dict=vals)
			print("pred: ", pred)
			print("label: ", label_data)
			print("--------")

			acc = dqn.sess.run(dqn.accuracy, feed_dict=vals)
			print("acc of train: ", acc)
		
		#---------------------------------------
		# Delayed System Updates
		#---------------------------------------
		if(iteration % 100 == 0):
			# update variables in Q^hat to be the same as in Q
			dqn.assignVariables()
			
			if(iteration % 1000 == 0):
				# save the model to checkpoint file
				#overwrite the saved model until 10,000 iterations have passed
				dir_name = "omega_"+str(iteration / 10000)
				if not os.path.exists(dir_name):
					os.makedirs(dir_name)
				dqn.saveModel(save_dir=dir_name)
			
	#######################
	## FINISH
	#######################
	
	# save final model to chekpoint file
	dir_name = "omega_final"
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)
	dqn.saveModel(save_dir=dir_name)

	te = datetime.now()
	print("time end: ", te)
	print("elapsed: ", te-ts)
	
