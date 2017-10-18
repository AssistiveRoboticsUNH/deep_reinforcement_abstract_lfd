#!/usr/bin/env python

# dqn_action_selector.py
# Madison Clark-Turner
# 10/14/2017

# collects observations and passes them into a DQN model in order
# to return the next action that should be performed

import rospy, rospkg
from std_srvs.srv import Empty

from dqn_model import DQNModel
filename = "ckpt_dir/model.ckpt"

from packager import DQNPackager

from deep_reinforcement_abstract_lfd.srv import *

############
# Services #
############

packager = None
num_prompt = 0

def get_next_action(req):
	print "Executing get_next_action"

	# Delay from action start to beginning of listening period.
	rospy.sleep(req.start_time-rospy.Time.now())
	
	packager.reset()
	packager.setPrint(0)
	rospy.sleep(req.end_time-rospy.Time.now())

	prompts = 0
	if(req.num_prompt > 0):
		prompts = 1

	next_act = packager.getNextAction(prompts)
	packager.setPrint(0)

	return DQNGetNextActionResponse(next_act)

def start_server(service_name, srv, func):
	rospy.init_node(service_name)
	s = rospy.Service(service_name, srv, func)
	print "Service "+service_name+" is ready."
	rospy.spin()

if __name__ == '__main__':
	
	rospack = rospkg.RosPack()
	path = rospack.get_path("deep_q_network")+'/';
	model = DQNModel([1,1,1], batch_size=1, learning_rate=0.0001, filename=path+filename, log_dir="LOG_DIR")
	print "DQN Model ready"
	packager = DQNPackager(model)
	print "DQN Packager ready"

	start_server("get_next_action", DQNGetNextAction, get_next_action)
