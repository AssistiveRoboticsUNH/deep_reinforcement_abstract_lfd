/*
main.cpp
Madison Clark-Turner
10/14/2017
*/

#include <ros/ros.h>
#include <ros/package.h>
#include <string>
#include <iostream>
#include "../include/deep_reinforcement_abstract_lfd/dqn_executor.h"

int main(int argc, char** argv){
	ros::init(argc, argv, "dqn");
	ros::NodeHandle n;
	DQNExecutor dqn(n);
	ROS_INFO("POMDP Executor ready");
	ros::spin();
}