#!/usr/bin/env python

# generate_tfrecord_from_rosbag.py
# Madison Clark-Turner
# 12/11/2017

import tensorflow as tf
import numpy as np

#file IO
import rosbag
import os
from os.path import isfile, join
from basic_tfrecord_rw import *

#contains data type information
from constants import *

#used for performing pre-processing steps on rostopics
from packager import * 

#rostopic names
topic_names = [
	'/action_finished',
	'/nao_robot/camera/top/camera/image_raw',
	'/nao_robot/microphone/naoqi_microphone/audio_raw'
]

'''
Read contents of a Rosbag and store:
s  - the current observation
a  - action that followed s
s' - the subsequent observation
a' - the action that followed s'
p  - how many prompts had been delivered before s
'''

def gen_TFRecord_from_file(out_dir, out_filename, bag_filename, flip=False):
	#outdir, bagfile, state, name, flip=False, index=-1):
	'''
	out_dir - the desierd output directory
	out_filename - the filename of the generated TFrecord (should NOT include 
		suffix)
	bag_filename - the rosbag being read
	flip - whether the img and optical flow data should be flipped horizontally 
		or not
	'''

	# packager subscribes to rostopics and does pre-processing
	packager = DQNPackager(flip=flip)

	# The occurence and times when actions begin
	time_log = []
	all_actions = []
	
	# variables that are altered during the generation process
	past_actions = []
	mostrecent_act = -1
	history = []
	
	# counters
	time_log_index = 0
	count, formatcount = 0,0

	# read RosBag
	bag = rosbag.Bag(bagfile)	

	# setup file names
	begin_file = out_dir+out_filename+'_'

	end_file = ".tfrecord"
	if(flip):
		end_file = "_flip"+end_file

	# state is an un-used variable but we maintin it as 0 here to avoid issues with legacy code
	state = 0

	#######################
	## ALTER TIMING INFO ##
	#######################

	for topic, msg, t in bag.read_messages(topics=['/action_finished']):
		if(index >= 0 and index <= 4 and msg.data == 1 and state == 1):
			msg.data = 2
		if(msg.data == 0):
			t = t-rospy.Duration(2.5)
		if(msg.data == 1):
			t = t-rospy.Duration(2.5)
		if(msg.data == 2):
			t = t-rospy.Duration(1)
		time_log.append(t)
		all_actions.append(msg)

	#######################
	##     READ FILE     ##
	#######################

	for topic, msg, t in bag.read_messages(topics=topic_names):

		if(time_log_index < len(time_log) and t > time_log[time_log_index]):
			# Observed action: Need to either store observations in memory or 
			# write to TFRecord

			past_actions.append(all_actions[time_log_index].data)

			if(len(past_actions) >= 2):

				# performing pre-processing steps on previous observations
				packager.formatOutput(name=name +'_'+ str(formatcount))
				formatcount += 1

				if(len(past_actions) >= 3):
					# record actions
					pre_act = past_actions[-3]
					cur_act = past_actions[-2]
					next_act = past_actions[-1]

					if(cur_act != next_act): 
						# Because each TFRecord needs to store the subsequent state and 
						# action we only generate TFRecords for the turn after the action
						# happened 
						# (ie. (s_{t-1}, a_{t-1}, s_t, a_t) instead of 
						# (s_t, a_t, s_{t+1}, a_{t+1}))
						ex = make_sequence_example (
								history[0], img_dtype, 
								history[1], pnt_dtype, 
								history[2], aud_dtype, 
								pre_act, cur_act, next_act, 
								state,
								packager.getImgStack(), 
								packager.getPntStack(), 
								packager.getAudStack(), 
								name +'_'+ str(count))
						writefile_name = begin_file+str(cur_act)+end_file
						writer = tf.python_io.TFRecordWriter(writefile_name)
						writer.write(ex.SerializeToString())
						writer.close()
						count += 1

				# store current pre-processed observation in memory
				history = [
						packager.getImgStack(), 
						packager.getPntStack(), 
						packager.getAudStack()]

			if(past_actions[-1] == 2 or past_actions[-1] == 3):
				# break if terminate action
				break
			else:
				packager.reset()

			time_log_index+=1

		elif(topic == topic_names[1]):
			packager.imgCallback(msg)
		elif(topic == topic_names[2]):
			packager.audCallback(msg)

	# need to write the TFRecord for the final interaction
	pre_act = past_actions[-2]
	cur_act = past_actions[-1]
	next_act = -1
	
	ex = make_sequence_example (
			packager.getImgStack(), img_dtype, 
			packager.getPntStack(), pnt_dtype, 
			packager.getAudStack(), aud_dtype, 
			pre_act, cur_act, next_act, 
			state, 
			[], 
			[], 
			[], 
			name +'_'+ str(count))
	writefile_name = begin_file+str(cur_act)+end_file
	writer = tf.python_io.TFRecordWriter(writefile_name)
	writer.write(ex.SerializeToString())
	writer.close()

	bag.close()

def check(filenames):
	
	coord = tf.train.Coordinator()
	filename_queue = tf.train.string_input_producer(filenames, capacity=32)
	out = []
	with tf.Session() as sess:
		sess.run(tf.local_variables_initializer())
		context_parsed, sequence_parsed = parse_sequence_example(filename_queue)
		threads = tf.train.start_queue_runners(coord=coord)
		seq_len = context_parsed["length"]# sequence length
		n = context_parsed["example_id"]
		for i in range(len(filenames)*2):
			num_frames, name = sess.run([seq_len, n])
			num_frames = num_frames.tolist()
			
			print(str(i)+'/'+str(len(filenames)*2), num_frames, name)
			
			out.append([num_frames, name])

		
		coord.request_stop()
		sess.run(filename_queue.close(cancel_pending_enqueues=True))
		coord.join(threads)

	return out


if __name__ == '__main__':
	gen_single_file = True
	view_single_file = True
	process_all_files = False
	
	rospy.init_node('gen_tfrecord', anonymous=True)

#############################

	# USAGE: generate a single file and store it as a scrap.tfrecord; Used for Debugging

	bagfile = os.environ["HOME"] + 
			"/Documents/AssistiveRobotics/AutismAssistant/pomdpData/test_03/zga1.bag"

	outfile = "../../tfrecords/scrap/scrap.tfrecord"
	outdir = os.environ["HOME"]+'/'+"catkin_ws/src/deep_q_network/tfrecords/scrap/"

	if(gen_single_file):
		gen_TFRecord_from_file(out_dir=outdir, out_filename="scrap", bag_filename=bagfile, flip=False)

#############################
	
	# USAGE: read contents of scrap.tfrecord; Used for Debugging

	if(view_single_file):
		#Format the data to be read
		def processData(inp, data_type):
			data_s = tf.reshape(inp, [-1, data_type["cmp_h"], data_type["cmp_w"], data_type["num_c"]])
			return tf.cast(data_s, tf.uint8)

		#Use for visualizing Data Types
		def show(data, d_type):
			tout = []
			out = []
			for i in range(data.shape[0]):
				imf = np.reshape(data[i], (d_type["cmp_h"], d_type["cmp_w"], d_type["num_c"]))

				limit_size = 64

				if(d_type["cmp_h"] > limit_size):
					mod = limit_size/float(d_type["cmp_h"])
					imf = cv2.resize(imf,None,fx=mod, fy=mod, interpolation = cv2.INTER_CUBIC)
				if(imf.shape[2] == 2):
					
					imf = np.concatenate((imf, np.zeros((d_type["cmp_h"],d_type["cmp_w"],1))), axis=2)
					imf[..., 0] = imf[..., 1]
					imf[..., 2] = imf[..., 1]
					imf = imf.astype(np.uint8)

				if(i % 10 == 0 and i != 0 ):
					if(len(tout) == 0):
						tout = out.copy()
					else:
						tout = np.concatenate((tout, out), axis=0)
					
					out = []
				if(len(out) == 0):
					out = imf
				else:
					out = np.concatenate((out, imf), axis = 1)

			if(data.shape[0] % 10 != 0):

				fill = np.zeros((limit_size, limit_size*(10 - (data.shape[0] % 10)), d_type["num_c"]))
				fill.fill(255)
				out = np.concatenate((out, fill), axis = 1)

			return tout

		
		print("READING...")
		coord = tf.train.Coordinator()
		filename_queue = tf.train.string_input_producer([outfile])

		with tf.Session() as sess:
			sess.run(tf.local_variables_initializer())
			#parse TFrecord
			context_parsed, sequence_parsed = parse_sequence_example(filename_queue)
			threads = tf.train.start_queue_runners(coord=coord)
			
			seq_len = context_parsed["length"]# sequence length
			seq_len2 = context_parsed["length_t2"]# sequence length
			labels = context_parsed["act"]# label
			labels2 = context_parsed["pos_act"]# label

			img_raw = processData(sequence_parsed["image_raw"], img_dtype)
			opt_raw = processData(sequence_parsed["points"], pnt_dtype)
			aud_raw = processData(sequence_parsed["audio_raw"], aud_dtype)

		for i in range(3): # alter number of iterations to the number of files
			l, l2, i, p, a, n, n2 = sess.run(
					[labels, labels2, img_raw, opt_raw, aud_raw, seq_len, seq_len2])
			print(i.shape, p.shape, a.shape, n, n2, l , l2)

			coord.request_stop()
			coord.join(threads)

			# display the contents of the optical flow file
			show_from = 110
			img = show(p[show_from:], pnt_dtype)
			cv2.imshow("img", img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		
#############################

	# USAGE: write all rosbag demonstrations to TFRecords

	'''
	We assume that the file structure for the demonstartions is ordered as follows:

		-<demonstration_path>
			-<subject_id>_0
				-compliant
					-<demonstration_name_0>.bag
					-<demonstration_name_1>.bag
					-<demonstration_name_2>.bag
				-noncompliant
					-<demonstration_name_0>.bag

			-<subject_id>_1
			-<subject_id>_2

		-<tfrecord_output_directory>

	'''

	# setup input directory information
	demonstration_path = os.environ["HOME"]+
			'/'+"Documents/AssistiveRobotics/AutismAssistant/pomdpData/"
	subject_id = "long_sess"

	# setup output directory information
	tfrecord_output_directory = os.environ["HOME"]+
			'/'+"catkin_ws/src/deep_q_network/tfrecords/long/"

	if(process_all_files):

		for i in range(1,12): # each unique subject
			subject_dir = demonstration_path + subject_id + '_'
			
			if(i < 10): # fix naming issues with leading 0s
				subject_dir += '0'
			subject_dir += str(i) + '/'

			for s in ["compliant", "noncompliant"]: # each unique state
				subject_dir += s + '/'

				#get list of demonstration file names
				filename_list = [subject_dir+f for 
						f in os.listdir(subject_dir) if isfile(join(subject_dir, f))]
				filename_list.sort()

				for f in filename_list:

					#get demonstration name for output file name
					tag = f
					while(tag.find("/") >= 0):
						tag = tag[tag.find("/")+1:]
					tag = tag[:-(len(".bag"))]
					new_name = subject_id+'_'+str(i)+'_'+tag

					# print files to make it clear process still running
					print(tag + "......." + new_name)

					gen_TFRecord_from_file(out_dir=tfrecord_output_directory, 
							out_filename=new_name, bag_filename=f, flip=False)

					gen_TFRecord_from_file(out_dir=tfrecord_output_directory, 
							out_filename=new_name, bag_filename=f, flip=True)