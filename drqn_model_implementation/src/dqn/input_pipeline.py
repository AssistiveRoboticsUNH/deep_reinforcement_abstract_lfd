from basic_tfrecord_rw import *

BATCH_SIZE = 1
NUM_EPOCHS = 10000

def input_pipeline(filenames):
	filename_queue = tf.train.string_input_producer(
			filenames, num_epochs=NUM_EPOCHS, shuffle=True)

	min_after_dequeue = 7 # buffer to shuffle with (bigger=better shuffeling) #7
	capacity = min_after_dequeue + 3 #* BATCH_SIZE
	
	# deserialize is a custom function that deserializes string
	# representation of a single tf.train.Example into tensors
	# (features, label) representing single training example
	context_parsed, sequence_parsed = parse_sequence_example(filename_queue)

	seq_len = context_parsed["length"]# sequence length
	seq_len_t2 = context_parsed["length_t2"]# sequence length
	pre_lab = context_parsed["pre_act"]# label
	labels = context_parsed["act"]# label
	name = context_parsed["example_id"]# label

	def extractFeature(name, data_type, sequence_parsed, fixsize=True, cast=tf.int32):
		#print(name, sequence_parsed[name].get_shape())
		data_t = tf.reshape(sequence_parsed[name], [-1])
		#next_ten = tf.Print(next_ten, [tf.shape(next_ten), name])
		if(fixsize):
			data_t = tf.reshape(data_t, [-1, data_type["cmp_h"] * data_type["cmp_w"] * data_type["num_c"]])
		data_t = tf.cast(data_t, cast)
		return data_t
	
	img_raw = extractFeature("image_raw", img_dtype, sequence_parsed, cast=tf.int32)
	points = extractFeature("points", pnt_dtype, sequence_parsed, cast=tf.float64)
	audio_raw = extractFeature("audio_raw", aud_dtype, sequence_parsed, cast=tf.float64)

	###################################################

	img_raw_t2 = extractFeature("image_raw_t2", img_dtype, sequence_parsed, False, cast=tf.int32)
	points_t2 = extractFeature("points_t2", pnt_dtype, sequence_parsed, False, cast=tf.float64)
	audio_raw_t2 = extractFeature("audio_raw_t2", aud_dtype, sequence_parsed, False, cast=tf.float64)

	pre_lab = tf.cast(pre_lab, tf.float32)
	labels -= 1
	labels_oh = tf.expand_dims(tf.one_hot(labels, 3), 0)
	lab = tf.cast(labels_oh, tf.float32)

	# Imagine inputs is a list or tuple of tensors representing single training example.
	# In my case, inputs is a tuple (features, label) obtained by reading TFRecords.
	NUM_THREADS = 1
	QUEUE_RUNNERS = 1

	inputs = [seq_len, seq_len_t2, img_raw, points, audio_raw, pre_lab, lab,  img_raw_t2, points_t2, audio_raw_t2, name]

	dtypes = list(map(lambda x: x.dtype, inputs))
	shapes = list(map(lambda x: x.get_shape(), inputs))

	queue = tf.RandomShuffleQueue(capacity, min_after_dequeue, dtypes)
	#queue = tf.FIFOQueue(capacity, dtypes)

	enqueue_op = queue.enqueue(inputs)
	qr = tf.train.QueueRunner(queue, [enqueue_op] * NUM_THREADS)

	tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr)
	inputs = queue.dequeue()

	for tensor, shape in zip(inputs, shapes):
		tensor.set_shape(shape)
	
	inputs_batch = tf.train.batch(inputs, 
																BATCH_SIZE, 
																capacity=capacity,
																dynamic_pad=True
																)
									
	return inputs_batch[0], inputs_batch[1], inputs_batch[2], inputs_batch[3], inputs_batch[4], inputs_batch[5], inputs_batch[6], inputs_batch[7], inputs_batch[8], inputs_batch[9], inputs_batch[10]
				 #seq_len,        #seq_len_2,      #img_raw,        #points,         #audio_raw,      #previous lab,   #lab,            #img_raw_t2,     #points_t2,      #audio_raw_t2			#example_id

def set_shape(arr, data_type):
	#print(arr.shape, data_type["cmp_h"] * data_type["cmp_w"] * data_type["num_c"])
	return np.reshape(arr, (BATCH_SIZE, -1, data_type["cmp_h"] * data_type["cmp_w"] * data_type["num_c"]))
