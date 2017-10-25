#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import time

import os
from os.path import isfile, join

from constants import *


# WRITE
def make_sequence_example(image_raw, image_data, points, point_data, 
		audio_raw, audio_data, pre_act, act, pos_act, state, image_raw_t2, 
		points_t2, audio_raw_t2, example_id):

	# The object we return
	ex = tf.train.SequenceExample()
	# A non-sequential feature of our example
	sequence_length = image_raw.shape[1]
	sequence_length_t2 = 0
	if(len(image_raw_t2) != 0):
		sequence_length_t2 = image_raw_t2.shape[1]

	ex.context.feature["length"].int64_list.value.append(sequence_length)
	ex.context.feature["length_t2"].int64_list.value.append(sequence_length_t2)

	ex.context.feature["img_h"].int64_list.value.append(image_data["cmp_h"])
	ex.context.feature["img_w"].int64_list.value.append(image_data["cmp_w"])
	ex.context.feature["img_c"].int64_list.value.append(image_data["num_c"])

	ex.context.feature["pnt_h"].int64_list.value.append(point_data["cmp_h"])
	ex.context.feature["pnt_w"].int64_list.value.append(point_data["cmp_w"])
	ex.context.feature["pnt_c"].int64_list.value.append(point_data["num_c"])

	ex.context.feature["aud_h"].int64_list.value.append(audio_data["cmp_h"])
	ex.context.feature["aud_w"].int64_list.value.append(audio_data["cmp_w"])
	ex.context.feature["aud_c"].int64_list.value.append(audio_data["num_c"])

	ex.context.feature["pre_act"].int64_list.value.append(pre_act)# act perfomed prior to s
	ex.context.feature["act"].int64_list.value.append(act)#act performed in s
	ex.context.feature["pos_act"].int64_list.value.append(pos_act)# act performed in s'

	ex.context.feature["compliant"].int64_list.value.append(state)
	ex.context.feature["example_id"].bytes_list.value.append(example_id)

	# Feature lists for input data
	def load_array(example, name, data, dtype):
		fl_data = example.feature_lists.feature_list[name].feature.add().bytes_list.value
		fl_data.append(np.asarray(data).astype(dtype).tostring())

	load_array(ex, "image_raw", image_raw, np.uint8)
	load_array(ex, "points", points, np.float64)
	load_array(ex, "audio_raw", audio_raw, np.float64)
	load_array(ex, "image_raw_t2", image_raw_t2, np.uint8)
	load_array(ex, "points_t2", points_t2, np.float64)
	load_array(ex, "audio_raw_t2", audio_raw_t2, np.float64)

	return ex

# READ
def parse_sequence_example(filename_queue):
	#reads a TFRecord into its constituent parts
	reader = tf.TFRecordReader()
	_, example = reader.read(filename_queue)
	
	context_features = {
		"length": tf.FixedLenFeature([], dtype=tf.int64),
		"length_t2": tf.FixedLenFeature([], dtype=tf.int64),
		
		#"img_h": tf.FixedLenFeature([], dtype=tf.int64),
		#"img_w": tf.FixedLenFeature([], dtype=tf.int64),
		#"img_c": tf.FixedLenFeature([], dtype=tf.int64),

		#"pnt_h": tf.FixedLenFeature([], dtype=tf.int64),
		#"pnt_w": tf.FixedLenFeature([], dtype=tf.int64),
		#"pnt_c": tf.FixedLenFeature([], dtype=tf.int64),

		#"aud_h": tf.FixedLenFeature([], dtype=tf.int64),
		#"aud_w": tf.FixedLenFeature([], dtype=tf.int64),
		#"aud_c": tf.FixedLenFeature([], dtype=tf.int64),

		"pre_act": tf.FixedLenFeature([], dtype=tf.int64),
		"act": tf.FixedLenFeature([], dtype=tf.int64),
		"pos_act": tf.FixedLenFeature([], dtype=tf.int64),

		"compliant": tf.FixedLenFeature([], dtype=tf.int64),
		"example_id": tf.FixedLenFeature([], dtype=tf.string)
	}

	sequence_features = {
		"image_raw": tf.FixedLenSequenceFeature([], dtype=tf.string),
		"points": tf.FixedLenSequenceFeature([], dtype=tf.string),
		"audio_raw": tf.FixedLenSequenceFeature([], dtype=tf.string),
		"image_raw_t2": tf.FixedLenSequenceFeature([], dtype=tf.string),
		"points_t2": tf.FixedLenSequenceFeature([], dtype=tf.string),
		"audio_raw_t2": tf.FixedLenSequenceFeature([], dtype=tf.string)
	}
	
	# Parse the example
	context_parsed, sequence_parsed = tf.parse_single_sequence_example(
		serialized=example,
		context_features=context_features,
		sequence_features=sequence_features
	)
	
	sequence_data = {
		"image_raw": tf.decode_raw(sequence_parsed["image_raw"], tf.uint8),
		"points": tf.decode_raw(sequence_parsed["points"], tf.float64),
		"audio_raw": tf.decode_raw(sequence_parsed["audio_raw"], tf.float64),
		"image_raw_t2": tf.decode_raw(sequence_parsed["image_raw_t2"], tf.uint8),
		"points_t2": tf.decode_raw(sequence_parsed["points_t2"], tf.float64),
		"audio_raw_t2": tf.decode_raw(sequence_parsed["audio_raw_t2"], tf.float64)
	}
	
	return context_parsed, sequence_data

def set_input_shape(arr, data_type):
	return np.reshape(arr, (BATCH_SIZE, -1, data_type["size"] * data_type["size"] * data_type["num_c"]))