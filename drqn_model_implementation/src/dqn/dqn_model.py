# dqn_model.py
# Madison Clark-Turner
# 12/11/2017

# helper function for printing tensor size
from __future__ import print_function

import tensorflow as tf 
import numpy as np
import os

# incpetion network (updated 10/25/2017)
import inception_resnet_v2 as inception_resnet_v2
slim = tf.contrib.slim

# contains information relating to input data size
from constants import *

# contains various meta parameters
from read_params import Params

# network layer information for P_CNN
layer_elements = [-1, 16, 32, 128, 3]

output_sizes = [32,16,4]
filter_sizes = [4,4,8]
stride_sizes = [2,2,4]
padding_size = [1,1,2]

# network layer information for A_CNN
aud_layer_elements = [-1, 16, 32, 128, 3]

aud_output_sizes = [(32,6),(16,4),(4,4)]
aud_filter_sizes = [(8,3),(4,3),(8,3)]
aud_stride_sizes = [(4,1),(2,1),(4,1)]
aud_padding_size = [(2,0),(1,0),(2,1)]

DEVICE = '/gpu:0'

TOTAL_PARAMS = 3

'''
DQN Model generates q-values for a given input observation
'''
class DQNModel:
	#######################
	## Constructor
	#######################
	'''
	graphbuild - bool array with len = TOTAL_PARAMS, indicates which CNNs should be active 
			(all by default)
	batch_size - int
			(1 by default)
	filename - string, meta parameter file #file with saved model parameters 
			(no model listed by default)
	name - string, model name 
			("dqn" by default)
	learning_rate - float, speed at which the model trains
			(1e-5 by default)
	'''
	def __init__(self, graphbuild = [1]*TOTAL_PARAMS, batch_size=1, filename=PARAMS_FILE, 
					name="dqn", learning_rate=1e-5, inception_ckpt="", model_ckpt=""):
		self.graphbuild = graphbuild 
		self.__batch_size = batch_size
		self.__name = name
		self.__alpha = learning_rate
		self.params = Params(filename)
		if(len(inception_ckpt) == 0):
			inception_ckpt = os.path.join(self.params.irnv2_checkpoint_dir, self.params.irnv2_checkpoint)
		if(len(model_ckpt) == 0):
			model_ckpt = self.params.restore_file
		
		#---------------------------------------
		# Model variables
		#---------------------------------------
		def weight_variable(name, shape):
			initial = tf.truncated_normal(shape, stddev=0.1)
			return tf.Variable(initial, name=name)
			
		def bias_variable(name, shape):
			initial = tf.constant(0.1, shape=shape)
			return tf.Variable(initial, name=name)
		
		# Q variables
		self.variables_pnt = {
			"W1" : weight_variable("W_conv1_pnt", [filter_sizes[0],filter_sizes[0],
						pnt_dtype["num_c"],layer_elements[1]]),
			"b1" : bias_variable("b_conv1_pnt", [layer_elements[1]]),
			"W2" : weight_variable("W_conv2_pnt", [filter_sizes[1],filter_sizes[1],
						layer_elements[1],layer_elements[2]]),
			"b2" : bias_variable("b_conv2_pnt", [layer_elements[2]]),
			"W3" : weight_variable("W_conv3_pnt", [filter_sizes[2],filter_sizes[2],
						layer_elements[2],layer_elements[-2]]),
			"b3" : bias_variable("b_conv3_pnt", [layer_elements[-2]])
		}
		
		self.variables_aud = {
			"W1" : weight_variable("W_conv1_aud", [aud_filter_sizes[0][0],
						aud_filter_sizes[0][1],aud_dtype["num_c"],aud_layer_elements[1]]),
			"b1" : bias_variable("b_conv1_aud", [aud_layer_elements[1]]),
			"W2" : weight_variable("W_conv2_aud", [aud_filter_sizes[1][0],
						aud_filter_sizes[1][1],aud_layer_elements[1],aud_layer_elements[2]]),
			"b2" : bias_variable("b_conv2_aud", [aud_layer_elements[2]]),
			"W3" : weight_variable("W_conv3_aud", [aud_filter_sizes[2][0],
						aud_filter_sizes[2][1],aud_layer_elements[2],aud_layer_elements[3]]),
			"b3" : bias_variable("b_conv3_aud", [aud_layer_elements[3]])
		}
		
		self.variables_lstm = {
			"W_lstm" : weight_variable("W_lstm", [layer_elements[-2],layer_elements[-1]]),
			"b_lstm" : bias_variable("b_lstm", [layer_elements[-1]]),
			"W_fc" : weight_variable("W_fc", [layer_elements[-1]+1,layer_elements[-1]]),
			"b_fc" : bias_variable("b_fc", [layer_elements[-1]])
		}
		
		# Q^hat variables
		self.variables_pnt_hat = {
			"W1" : weight_variable("W_conv1_pnt_hat", [filter_sizes[0],filter_sizes[0],
						pnt_dtype["num_c"],layer_elements[1]]),
			"b1" : bias_variable("b_conv1_pnt_hat", [layer_elements[1]]),
			"W2" : weight_variable("W_conv2_pnt_hat", [filter_sizes[1],filter_sizes[1],
						layer_elements[1],layer_elements[2]]),
			"b2" : bias_variable("b_conv2_pnt_hat", [layer_elements[2]]),
			"W3" : weight_variable("W_conv3_pnt_hat", [filter_sizes[2],filter_sizes[2],
						layer_elements[2],layer_elements[-2]]),
			"b3" : bias_variable("b_conv3_pnt_hat", [layer_elements[-2]])
		}
		
		self.variables_aud_hat = {
			"W1" : weight_variable("W_conv1_aud_hat", [aud_filter_sizes[0][0],
						aud_filter_sizes[0][1],aud_dtype["num_c"],aud_layer_elements[1]]),
			"b1" : bias_variable("b_conv1_aud_hat", [aud_layer_elements[1]]),
			"W2" : weight_variable("W_conv2_aud_hat", [aud_filter_sizes[1][0],
						aud_filter_sizes[1][1],aud_layer_elements[1],aud_layer_elements[2]]),
			"b2" : bias_variable("b_conv2_aud_hat", [aud_layer_elements[2]]),
			"W3" : weight_variable("W_conv3_aud_hat", [aud_filter_sizes[2][0],
						aud_filter_sizes[2][1],aud_layer_elements[2],aud_layer_elements[3]]),
			"b3" : bias_variable("b_conv3_aud_hat", [aud_layer_elements[3]])
		}

		self.variables_lstm_hat = {
			"W_lstm" : weight_variable("W_lstm_hat", [layer_elements[-2],layer_elements[-1]]),
			"b_lstm" : bias_variable("b_lstm_hat", [layer_elements[-1]]),
			"W_fc" : weight_variable("W_fc_hat", [layer_elements[-1]+1,layer_elements[-1]]),
			"b_fc" : bias_variable("b_fc_hat", [layer_elements[-1]])
		}
		
		#---------------------------------------
		# Placeholder variables
		#---------------------------------------
		# placeholder for the RGB data
		self.img_ph = tf.placeholder("float", 
			[self.__batch_size, None, img_dtype["cmp_h"] * img_dtype["cmp_w"] * img_dtype["num_c"]],
			name="img_placeholder")
		
		# placeholder for the Optical Flow data
		self.pnt_ph = tf.placeholder("float", 
			[self.__batch_size, None, pnt_dtype["cmp_h"] * pnt_dtype["cmp_w"] * pnt_dtype["num_c"]], 
			name="pnt_placeholder")

		# placeholder for the Audio data
		self.aud_ph = tf.placeholder("float", 
			[self.__batch_size, None, aud_dtype["cmp_h"] * aud_dtype["cmp_w"] * aud_dtype["num_c"]], 
			name="aud_placeholder")

		# placeholder for the sequnce length
		self.seq_length_ph = tf.placeholder("int32", [self.__batch_size], 
					name="seq_len_placeholder")

		# placeholder for where each sequence ends in a matrix
		self.partitions_ph = tf.placeholder("int32",  [self.__batch_size, None], 
			name="partition_placeholder" )

		# placeholder for boolean listing whether the network is being trained or evaluated
		self.train_ph = tf.placeholder("bool", [], name="train_placeholder")

		# placeholder for how many prompts have been delivered
		self.prompts_ph = tf.placeholder("float32", [self.__batch_size], 
			name="prompts_placeholder")
		
		# placeholder for the reward values to classify with
		self.y_ph = tf.placeholder("float", [None, layer_elements[-1]], name="y_placeholder")
		
		#---------------------------------------
		# Build Model Structure
		#---------------------------------------

		# initalize all variables in the network
		self.pred_var_set = self.execute_model_DQN_var_set()#used to initialize variables

		#---------------------------------------
		# Set Transfer Learning Variables
		#---------------------------------------
		
		# layers of INRV2 to replace 
		exclusions = ["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"]

		# Q values
		self.variables_img = {}

		# restore all non-excluded variables from the INRV2 checkpoint
		init_transfer_learning = None
		variables_to_restore = []
		if(self.graphbuild[0]):
			for var in slim.get_model_variables():
				excluded = False
				for exclusion in exclusions:
					if var.op.name.startswith(exclusion):
						name = var.name[var.name.find('/')+1:-2]
						self.variables_img[name] = var
						excluded = True
						break
				
				if not excluded:
					variables_to_restore.append(var)

			init_transfer_learning = slim.assign_from_checkpoint_fn(
					inception_ckpt, variables_to_restore)


		# we only need to train the last FC layer so we can remove the other checkpoint variables
		# from our list of trainable variables
		variables_to_train = [x for x in tf.trainable_variables() if x not in variables_to_restore]
		
		# Variables are used when evaluating the Q network
		self.variables_img_main = {}
		# Q^hat values
		self.variables_img_hat = {}

		# set all img variables to be the same
		for k in self.variables_img.keys():
			self.variables_img_main[k] = tf.Variable(self.variables_img[k].initialized_value())
			self.variables_img_hat[k] = tf.Variable(self.variables_img[k].initialized_value())

		#---------------------------------------
		# Q-value Generation Functions
		#---------------------------------------

		# generate q-values for each of the actions using Q
		self.generate_q_values = self.execute_model_DQN()

		# return the action with the highest q-value
		self.generate_best_action = tf.argmax(self.generate_q_values,1)

		# generate q-values for each of the actions using Q^hat
		self.generate_q_hat_values = self.execute_model_DQN_hat()

		# get the highest q-value for an input from Q^hat
		self.max_q_hat_value = tf.reduce_max(self.generate_q_hat_values, axis = 1)

		#---------------------------------------
		# Optimization Functions
		#---------------------------------------

		# get the difference between the q-values and the true output
		self.diff = self.y_ph - tf.clip_by_value(self.generate_q_values,1e-10,100)

		# cross entropy
		self.cross_entropy = tf.reduce_mean(tf.square(self.diff))
		#tf.summary.scalar('cross_entropy', self.cross_entropy)

		# optimize the network
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.__alpha).minimize(
				self.cross_entropy, var_list=variables_to_train)

		#---------------------------------------
		# Evaluation Functions
		#---------------------------------------

		# return a boolean indicating whether the system correctly predicted the output
		self.correct_pred = tf.equal(tf.argmax(self.generate_q_values,1), tf.argmax(self.y_ph,1))

		# the accuracy of the current batch
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
		#tf.summary.scalar('accuracy', self.accuracy)

		#---------------------------------------
		# Initialization
		#---------------------------------------

		# Generate Session
		self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement = True))

		# Variable for generating a save checkpoint
		self.saver = tf.train.Saver()

		# initalize IRNV2 variables
		if(init_transfer_learning):
			init_transfer_learning(self.sess)

		if(len(self.params.restore_file) == 0):
			# initalize all model variables
			init_op = tf.global_variables_initializer()
			self.sess.run(init_op)
			print("VARIABLE VALUES INITIALIZED")
		else:
			# restore variables from a checkpoint
			self.saver.restore(self.sess, model_ckpt)
			print("VARIABLE VALUES RESTORED FROM: "+ model_ckpt)

		#---------------------------------------
		# Summary Functions
		#---------------------------------------

		self.merged_summary = tf.summary.merge_all()
		self.train_writer = tf.summary.FileWriter(self.params.log_dir + '/train', self.sess.graph)
		self.test_writer = tf.summary.FileWriter(self.params.log_dir + '/test')
		self.graph_writer = tf.summary.FileWriter(self.params.log_dir + '/projector', self.sess.graph)

#######################
## Helper Functions
#######################

	def saveModel(self, directory_identifier=""):
		'''
		save the model to a checkpoint file
			-name: (string) name of the checkpoint file
			-save_dir: (string) directory to save the file into
		'''
		dir_name = self.params.save_dir+'_'+directory_identifier
		if not os.path.exists(dir_name):
			os.makedirs(dir_name)
		path = self.saver.save(self.sess, dir_name+'/'+self.params.model_name)

	def variable_summaries(self, var, name):
		# append metric information to summary
		with tf.name_scope('summaries'+name):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stddev'):
					stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)

	#---------------------------------------
	# Variable Value Manipulators
	#---------------------------------------

	def restore_q_hat_vars(self, src, dst):
		'''
		set the value of variables in one dictionary to those in another. Primarily
		used when to switch the values of the network between those in Q and Q^hat
			-name: (string) name of the checkpoint file
			-save_dir: (string) directory to save the file into
		'''
		arr = []
		var_dst = []
		for k in dst.keys():
			arr.append(src[k])
			var_dst.append(dst[k])
		arr = self.sess.run(arr)
		for seq, var in zip(arr, var_dst):
			
			v = np.array(seq).reshape(np.array(seq).shape)
			var.load(v, session=self.sess)

	def assignVariables(self):
		# set the values of variables in Q^hat to be the same as those in Q
		self.restore_q_hat_vars(self.variables_img_main, self.variables_img_hat)
		self.restore_q_hat_vars(self.variables_pnt, self.variables_pnt_hat)
		self.restore_q_hat_vars(self.variables_aud, self.variables_aud_hat)
		self.restore_q_hat_vars(self.variables_lstm, self.variables_lstm_hat)

	#######################
	## Executor Functions
	#######################

	def execute_model_DQN_var_set(self):
		# Initalize the model's structure
		return self.model(
			self.seq_length_ph, 
			self.img_ph, 
			self.pnt_ph, 
			self.aud_ph, 
			self.partitions_ph, 
			self.train_ph, 
			self.prompts_ph, 
			tf.variable_scope("dqn"),
			tf.variable_scope("dqn"),
			"",
			self.variables_pnt,
			self.variables_aud,
			self.variables_lstm,
			False
			)

	def execute_model_DQN(self):
		# Generate the q-values of Q for the given input
		return self.model(
			self.seq_length_ph, 
			self.img_ph, 
			self.pnt_ph, 
			self.aud_ph, 
			self.partitions_ph, 
			self.train_ph, 
			self.prompts_ph, 
			tf.variable_scope("dqn"),
			tf.variable_scope("dqn", reuse=True),
			"",
			self.variables_pnt,
			self.variables_aud,
			self.variables_lstm
			)
	
	def execute_model_DQN_hat(self):
		# Generate the q-values of Q^hat for the given input
		return self.model(
			self.seq_length_ph, 
			self.img_ph, 
			self.pnt_ph, 
			self.aud_ph, 
			self.partitions_ph, 
			self.train_ph, 
			self.prompts_ph, 
			tf.variable_scope("dqn_hat"), 
			tf.variable_scope("dqn", reuse=True),
			"",
			self.variables_pnt_hat,
			self.variables_aud_hat,
			self.variables_lstm_hat
			)

	def genPrediction(self, num_frames, img_data, pnt_data, aud_data, num_prompts,
		 verbose=False):
		'''
		Generate q-values for an input passed in as seperate data points. Used when 
		by external systems (ROS) to run the model without having to import	tensorflow 
			-num_frames: (int) the number of frames in the video
			-img_data: (numpy array) an array that contains the RGB data
			-pnt_data: (numpy array) an array that contains the optical flow data
			-aud_data: (numpy array) an array that contains the audio data
			-num_prompts: (int) directory to save the file into
			-verbose: (bool) print additional information
		'''
		partitions = np.zeros((1, num_frames))
		partitions[0][-1] = 1
		
		#with tf.variable_scope(self.__name) as scope:
		q_vals = self.sess.run(self.generate_q_values, feed_dict={
			self.seq_length_ph: [num_frames], 
			self.img_ph: img_data, 
			self.pnt_ph: pnt_data,
			self.aud_ph: aud_data,
			self.partitions_ph: partitions,
			self.train_ph: False,
			self.prompts_ph: [num_prompts]
			})

		if(verbose):
			available_actions = ["PMT", "REW", "ABT"]
			print("num_prompts: ", num_prompts)
			print("Q-values: " + str(q_vals))
			print("Largest q-value: " + str(np.max(q_vals)))
			print("Best action: " + available_actions[np.argmax(q_vals)])

		return np.argmax(q_vals)

	#######################
	## The Model
	#######################
	
	def model(self, seq_length, img_ph, pnt_ph, aud_ph, partitions_ph, train_ph, prompts_ph,
		 variable_scope, variable_scope2, var_img, var_pnt, var_aud, var_lstm, incep_reuse=True):
		'''
		The DQN model
			-seq_length: (placeholder) the number of frames in the video
			-img_ph: (placeholder) an array that contains the RGB data
			-pnt_ph: (placeholder) an array that contains the optical flow data
			-aud_ph: (placeholder) an array that contains the audio data
			-partitions_ph: (placeholder) an 'I x seq_length' matrix that indicates where 
					seqences end. Used for obtaining the output of LSTM
			-train_ph: (placeholder) a bool indicating whether the variables are being trained
			-prompts_ph: (placeholder) the number of prompts that the network has delivered
			-variable_scope: (variable_scope) scope for the CNN stacks
			-variable_scope2: (variable_scope) scope for the temporal data
			-var_img: (dict) the variables for the RGB input
			-var_pnt: (dict) the variables for the optical flow input
			-var_aud: (dict) the variables for the audio input
			-var_lstm: (dict)  the variables for the LSTM
			-incep_reuse: (bool) a bool indicating whether the incpetion values should be re-used
		'''
		
		#---------------------------------------
		# Data Processing Functions											
		#---------------------------------------

		def process_vars(seq, data_type):
			# cast inputs to the correct data type
			seq_inp = tf.cast(seq, tf.float32)
			return tf.reshape(seq_inp, (self.__batch_size, -1, data_type["cmp_h"], data_type["cmp_w"], data_type["num_c"]))

		def check_legal_inputs(tensor, name):
			# ensure that the current tensor is finite (doesn't have any NaN values)
			return tf.verify_tensor_all_finite(tensor, "ERR: Tensor not finite - "+name,
					name=name)

		#---------------------------------------
		# Convolution Functions 												
		#---------------------------------------

		def convolve_data_inception(input_data, val, n, dtype):
			# pass data into the INRV2 Network
			with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
				data = tf.reshape(input_data, [-1, 299, 299, 3])
				logits, _ = inception_resnet_v2.inception_resnet_v2(data, 
						num_classes=output_sizes[-1]*output_sizes[-1]*layer_elements[-2], 
						is_training=False, reuse=incep_reuse)
			return logits		
		
		def convolve_data_3layer_pnt(input_data, val, variables, n, dtype):
			# pass data into through P_CNN 
			def pad_tf(x, p):
				return tf.pad(x, [[0,0],[p,p],[p,p],[0,0]], "CONSTANT")

			def gen_convolved_output(sequence, W, b, stride, num_hidden, new_size, train_ph, padding='SAME'):
				conv = tf.nn.conv2d(sequence, W, strides=[1, stride, stride, 1], padding=padding) + b
				return tf.nn.relu(conv)

			input_data = tf.reshape(input_data, [-1, dtype["cmp_h"], dtype["cmp_w"], dtype["num_c"]], name=n+"_inp_reshape")

			for i in range(3):
				si = str(i+1)

				input_data = pad_tf(input_data, padding_size[i])
				padding = "VALID"

				input_data = gen_convolved_output(input_data, variables["W"+si], variables["b"+si], 
						stride_sizes[i], layer_elements[i+1], output_sizes[i], train_ph, padding)
				#self.variable_summaries(input_data, dtype["name"]+"_conv"+si)
				input_data = check_legal_inputs(input_data, "conv"+si+"_"+n)

			return input_data

		def convolve_data_3layer_aud(input_data, val, variables, n, dtype):
			# pass data into through A_CNN 
			def pad_tf(x, padding):
				return tf.pad(x, [[0,0],[padding[0],padding[0]],[padding[1],padding[1]],[0,0]], "CONSTANT")

			def gen_convolved_output(sequence, W, b, stride, num_hidden, new_size, train_ph, padding='SAME'):
				conv = tf.nn.conv2d(sequence, W, strides=[1, stride[0], stride[1], 1], padding=padding) + b
				return tf.nn.relu(conv)

			input_data = tf.reshape(input_data, [-1, dtype["cmp_h"], dtype["cmp_w"], dtype["num_c"]], name=n+"_inp_reshape")

			for i in range(3):
				si = str(i+1)

				input_data = pad_tf(input_data, aud_padding_size[i])
				padding = "VALID"

				input_data = gen_convolved_output(input_data, variables["W"+si], variables["b"+si], aud_stride_sizes[i], 
						aud_layer_elements[i+1], aud_output_sizes[i], train_ph, padding)
				#self.variable_summaries(input_data, dtype["name"]+"_conv"+si)
				input_data = check_legal_inputs(input_data, "conv"+si+"_"+n)

			return input_data

		#=======================================
		#	(Model Execution Begins Here)
		#=======================================

		#---------------------------------------
		# CNN Stacks 												
		#---------------------------------------
		
		# Storage Variables
		# 0 - RGB, 1 - Optical Flow, 2 - Audio
		inp_data = [0]*TOTAL_PARAMS
		conv_inp = [0]*TOTAL_PARAMS

		with tf.device(DEVICE):
			# Inception Network (INRV2)
			if(self.graphbuild[0]):
				val = 0
				inp_data[val] = process_vars(img_ph, img_dtype)
				conv_inp[val] = convolve_data_inception(inp_data[val], val, "img", img_dtype)
			
			with variable_scope as scope:
				
				# P_CNN
				if(self.graphbuild[1]):
					val = 1
					inp_data[val] = process_vars(pnt_ph, pnt_dtype)
					conv_inp[val] = convolve_data_3layer_pnt(inp_data[val], val, var_pnt, "pnt", pnt_dtype)
				
				# A_CNN
				if(self.graphbuild[2]):
					val = 2
					inp_data[val] = process_vars(aud_ph, aud_dtype)
					conv_inp[val] = convolve_data_3layer_aud(inp_data[val], val, var_aud, "aud", aud_dtype)

				#---------------------------------------
				# Combine Output of CNN Stacks
				#---------------------------------------
				combined_data = None
				for i in range(TOTAL_PARAMS):

					if(self.graphbuild[i]):
						if(i < 2):
							conv_inp[i] = tf.reshape(conv_inp[i], [self.__batch_size, -1, 
									output_sizes[-1]*output_sizes[-1]*layer_elements[-2]], name="combine_reshape")
						else:
							conv_inp[i] = tf.reshape(conv_inp[i], [self.__batch_size, -1, 
									aud_output_sizes[-1][0]*aud_output_sizes[-1][0]*aud_layer_elements[-2]], name="combine_reshape_aud")
						
						if(combined_data == None):
							combined_data = conv_inp[i]
						else:
							combined_data = tf.concat([combined_data, conv_inp[i]], 2)

						combined_data = check_legal_inputs(combined_data, "combined_data")
									
				# capture variables before changing scope 
				W_lstm = var_lstm["W_lstm"]
				b_lstm = var_lstm["b_lstm"]
				W_fc = var_lstm["W_fc"]
				b_fc = var_lstm["b_fc"]		
		
		with variable_scope2 as scope:
			#---------------------------------------
			# Internal Temporal Information (LSTM)
			#---------------------------------------
			lstm_cell = tf.contrib.rnn.LSTMCell(layer_elements[-2], 
													use_peepholes=False,
													cell_clip=None,
													initializer=None,
													num_proj=None,
													proj_clip=None,
													forget_bias=1.0,
													state_is_tuple=True,
													activation=None,
													reuse=None
													)
			
			lstm_mat, _ = tf.nn.dynamic_rnn(
													cell=lstm_cell, 
													inputs=combined_data, 
													dtype=tf.float32,
													sequence_length=seq_length,
													time_major=False
													)

			#if lstm_out is NaN replace with 0 to prevent model breakage
			lstm_mat = tf.where(tf.is_nan(lstm_mat), tf.zeros_like(lstm_mat), lstm_mat)
			lstm_mat = check_legal_inputs(lstm_mat, "lstm_mat")
			
			# extract relevant information from LSTM output using partiitions
			num_partitions = 2
			lstm_out = tf.dynamic_partition(lstm_mat, partitions_ph, num_partitions)[1]
		
			# FC1
			fc1_out = tf.matmul(lstm_out, W_lstm) + b_lstm
			fc1_out = check_legal_inputs(fc1_out, "fc1")
			self.variable_summaries(fc1_out, "fc1")

			#---------------------------------------
			# External Temporal Information (number of promtps)
			#---------------------------------------
			# shape and append prompt information here
			prompts_ph = tf.reshape(prompts_ph, [-1, 1])
			fc1_prompt = tf.concat([fc1_out, prompts_ph], 1)

			# FC2: generate final q-values
			fc2_out = tf.matmul(fc1_prompt, W_fc) + b_fc
			fc2_out = check_legal_inputs(fc2_out, "fc2")
			self.variable_summaries(fc2_out, "fc")
			
			return fc2_out

if __name__ == '__main__':
	dqn = DQNModel([1,1,1])

