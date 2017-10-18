# dqn_packager_fast.py
# Madison Clark-Turner
# 10/14/2017

import tensorflow as tf
import numpy as np

from constants import *

import math
import threading

# ROS
import rospy, rospkg
from std_msgs.msg import Int8
from sensor_msgs.msg import Image

# image pre-processing and optical flow generation
import cv2
from cv_bridge import CvBridge, CvBridgeError

# audio pre-processing
from nao_msgs.msg import AudioBuffer
from noise_subtraction import reduce_noise
from scipy import signal
import librosa, librosa.display

# DQN
from dqn_model import DQNModel

topic_names = [
	'/action_finished',
	'/nao_robot/camera/top/camera/image_raw',
	'/nao_robot/microphone/naoqi_microphone/audio_raw'
]

'''
DQN Packager listens to the topics for images and audio.
Processes those inputs into sequences and passes the result to 
the DQN model.
'''
class DQNPackager:

	def __init__(self, dqn=None, flip=False):
		# dqn model
		self.__dqn = dqn
		self.__flip = flip
		self.p = 0

		# variables for tracking images received
		self.__most_recent_act = -1
		self.__lock = threading.Lock()
		self.reset()

		#variables for optical flow
		self.frame1 = []
		self.prvs, self.hsv = None, None

		# variables for audio
		self.counter = 0
		plt.gca().set_position([0, 0, 1, 1])
		src_dir = rospkg.RosPack().get_path('deep_q_network') + '/src/'
		self.__face_cascade = cv2.CascadeClassifier(src_dir+'haarcascade_frontalface_default.xml')
		self.rate = 16000 # Sampling rate

		# subscribers
		QUEUE_SIZE = 1
		self.sub_act = rospy.Subscriber(topic_names[0], 
			Int8, self.actCallback,  queue_size = QUEUE_SIZE)
		self.sub_img = rospy.Subscriber(topic_names[1], 
			Image, self.imgCallback,  queue_size = QUEUE_SIZE)
		self.sub_aud = rospy.Subscriber(topic_names[2], 
			AudioBuffer, self.audCallback,  queue_size = QUEUE_SIZE)

	def getRecentAct(self):
		return self.__most_recent_act

	def getImgStack(self):
		return self.__imgStack

	def getPntStack(self):
		return self.__pntStack

	def getAudStack(self):
		return self.__audStack

	############################
	# Collect Data into Frames #
	############################

	def setPrint(self,p):
		self.p = p

	def clearMsgs(self):
		self.__recent_msgs = [False]*2

	def reset(self, already_locked=False):
		if(not already_locked):
			self.__lock.acquire()
		self.clearMsgs()
		self.__imgStack = 0
		self.__pntStack = 0
		self.__audStack = 0

		self.frame1 = []
		self.prvs, self.hsv = None, None

		if(not already_locked):	
			self.__lock.release()
		#print("reset complete")

	def actCallback(self, msg):
		self.__most_recent_act = msg.data
		return

	def imgCallback(self, msg):
		self.__recent_msgs[0] = msg
		self.checkMsgs()
		return

	def audCallback(self, msg):
		self.__recent_msgs[1] = msg
		self.checkMsgs()
		return

	def formatAudMsg(self, aud_msg):
		# shapes the audio file for use later
		data = aud_msg.data
		data = np.reshape(data, (-1,4))
		data = data.transpose([1,0])

		return data[0]

	def checkMsgs(self):
		#may need to use mutexes on self.__recent_msgs
		self.__lock.acquire()
		if False in self.__recent_msgs:
			self.__lock.release()
			return
		if(self.p):
			print("FRAME ADDED!")
		#organize and send data
		img = self.__recent_msgs[0]
		aud = self.formatAudMsg(self.__recent_msgs[1]) # process all audio data together prior to sending

		if(type(self.__imgStack) == int):
			self.__imgStack = [img]
			self.__audStack = [aud]
		else:
			self.__imgStack.append(img)
			self.__audStack.append(aud)

		self.clearMsgs()
		self.__lock.release()

	###############
	# Format Data #
	###############

	def formatImgBatch(self, img_stack, name=""):
		# pre-process the RGB input and generate the optical flow
		img_out, pnt_out = [], []
		
		for x in img_stack:
			img = self.formatImg(x)
			img_out.append(np.asarray(img).flatten())
			pnt_out.append(self.formatOpt(img))

		return img_out, pnt_out

	def formatImg(self, img_msg):
		# pre-process the image data to crop it to an appropriate size

		# convert image to cv2 image
		img = CvBridge().imgmsg_to_cv2(img_msg, "bgr8")
		
		# identify location of face if possible using Haarcascade and
		# crop to center on it.
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = self.__face_cascade.detectMultiScale(gray, 1.3, 5)
		x,y,w,h = -1,-1,-1,-1

		# if face is locataed to the edge set then crop from the opposite side
		buff = img.shape[1]-img.shape[0]

		if(len(faces) > 0):
			for (xf,yf,wf,hf) in faces:
				if wf*hf > w*h:
					x,y,w,h = xf,yf,wf,hf

			if(x >= 0 and y >= 0 and x+w < 320 and y+h < 240):	
				y, h = 0, img.shape[0]
				mid = x+(w/2)
				x, w = mid-(img.shape[0]/2), img.shape[0]
				if(x < 0):
					x = 0
				elif(x > buff):
					x = buff/2
				img = img[y:y+h, x:x+w]
		else:
			# if no face visible set crop image to center of the video
			diff = img.shape[1]-img.shape[0]
			img = img[0:img.shape[0], (diff/2):(img.shape[1]-diff/2)]

		#resize image to 299 x 299
		y_mod = 1/(img.shape[0]/float(img_dtype["cmp_h"]))
		x_mod = 1/(img.shape[1]/float(img_dtype["cmp_w"]))
		img = cv2.resize(img,None,fx=x_mod, fy=y_mod, interpolation = cv2.INTER_CUBIC)

		if(self.__flip):
			# if flip set to true then mirror the image horizontally
			img = np.flip(img, 1)

		return img

	def formatOpt(self, img_src):
		# generate optical flow


		img = img_src.copy()
		empty = np.zeros(img.shape)
		opt_img = empty
		
		if(len(self.frame1)==0):
			# if img is the first frame then set optical flow to be black screen
			self.frame1 = img
			self.prvs = cv2.cvtColor(self.frame1,cv2.COLOR_BGR2GRAY)

		else:
			frame2 = img

			# generate optical flow
			next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
			flow = cv2.calcOpticalFlowFarneback(self.prvs,next, 0.5, 1, 2, 5, 7, 1.5, 1)

			mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
			# normalize the magnitude to between 0 and 255 (replace with other normalize to prevent precission issues)
			mag = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
			opt_img = mag
			self.prvs = next

		#make image 3 channel in order to perform re-size operations
		stack = np.dstack((opt_img, empty, empty))

		#resize image
		mod = pnt_dtype["cmp_h"]/float(img_dtype["cmp_h"])
		opt_img_resized = cv2.resize(stack,None,fx=mod, fy=mod, interpolation=cv2.INTER_CUBIC)
		#opt_img_resized = cv2.normalize(opt_img_resized,None,0,255,cv2.NORM_MINMAX) <----------------Should use this to normalize input, not the one on mag
		
		return np.asarray(opt_img_resized[...,0]).flatten()

	def formatAudBatch(self, aud_msg_array, name=""):
		# perform pre-processing on the audio input
		
		num_frames = len(aud_msg_array)
		input_data = np.reshape(aud_msg_array, (num_frames*len(aud_msg_array[0])))

		# modify data
		core_data = input_data
		
		# mute the first 2 seconds of audio (where the NAO speaks)
		mute_time = 2
		input_data[:np.argmax(input_data)+int(16000*mute_time)] = 0

		# get the indicies for the noise sample
		noise_sample_s, noise_sample_e = 16000*(-1.5), -1

		# perform spectral subtraction to reduce noise
		noise = core_data[int(noise_sample_s): noise_sample_e]
		filtered_input = reduce_noise(np.array(core_data), noise)

		# smooth signal
		b, a = signal.butter(3, 0.05)
		filtered_input = signal.lfilter(b,a,filtered_input)
		noise = filtered_input[int(noise_sample_s): noise_sample_e]

		# additional spectral subtraction to remove remaining noise
		filtered_input = reduce_noise(filtered_input, noise)

		# generate spectrogram
		S = librosa.feature.melspectrogram(y=filtered_input, sr=self.rate, n_mels=128, fmax=8000)
		S = librosa.power_to_db(S, ref=np.max)
		
		arr = []
		
		if(False):
			# if True then output spectrogram to png file
			plt.figure(figsize=(10,4))
		
			librosa.display.specshow(S,y_axis='mel', fmax=8000,x_axis='time')
			plt.colorbar(format='%+2.0f dB')
			plt.title('Mel-Spectrogram')
			plt.tight_layout()
			print("spectrogram ouput to file.")

			out_file = "spec_img/all_spec/"+name+".png"
			plt.savefig(out_file)
			self.counter += 1
			plt.clf()
		

		# split the spectrogram into A_i. This generates an overlap between 
		# frames with as set stride
		stride = S.shape[1]/float(num_frames)
		frame_len = aud_dtype["cmp_w"]

		#pad the entire spectrogram so that overlaps at either end do not fall out of bounds
		empty = np.zeros((S.shape[0], 3))
		empty_end = np.zeros((S.shape[0], 8))
		S = np.concatenate((empty, S, empty_end), axis = 1)

		split_data = np.zeros(shape=(num_frames, S.shape[0], frame_len),dtype=S.dtype)
		for i in range(0, num_frames):
			split_data[i]=S[:,int(math.floor(i*stride)):int(math.floor(i*stride))+frame_len]
			
		#normalize the output to be between 0 and 255
		split_data -= split_data.min()
		split_data /= split_data.max()/255.0

		return np.reshape(split_data, (num_frames, -1))

	#############
	# Send Data #
	#############

	def formatOutput(self, name=""):
		# Execute pre-processing on all strored input
		img_stack, opt_stack = self.formatImgBatch(self.__imgStack, name)

		self.__imgStack = np.expand_dims(img_stack, axis=0)
		self.__pntStack = np.expand_dims(opt_stack, axis=0)
		self.__audStack = np.expand_dims(self.formatAudBatch(self.__audStack, name), axis=0)
		
	def getNextAction(self, num_prompt, verbose =False):
		# Execute Pre-processing steps and pass data to the DQN
		if(self.__dqn == None):
			print("model not provided!")
			return -1

		self.__lock.acquire()
		num_frames = len(self.__imgStack)

		self.formatOutput()

		if(verbose):
			print("Prediction has "+str(num_frames)+" frames")
			print("imgStack has shape: ", self.__imgStack.shape)
			print("pntStack has shape: ", self.__pntStack.shape)
			print("audStack has shape: ", self.__audStack.shape)
			print("num_prompt is "+str(num_prompt))

		# generate the DQN prediction
		nextact = self.__dqn.genPrediction(num_frames, self.__imgStack, self.__pntStack, self.__audStack, num_prompt) +1

		# clear the input
		self.reset(already_locked=True)


		self.__lock.release()
		
		if(verbose):
			print("nextact: ",nextact)

		return nextact

if __name__ == '__main__':
	packager = DQNPackager()
