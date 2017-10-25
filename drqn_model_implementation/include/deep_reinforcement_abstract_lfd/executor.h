/*
dqnexecutor.h
Madison Clark-Turner
10/14/2017
*/

/*
 receives the run command.
 passes the observations into the parser and receives information
 from the model
*/

#ifndef Executor_H
#define Executor_H

#include <ros/ros.h>

#include <string>

#include <std_msgs/Bool.h>
#include <std_msgs/Int8.h>
#include <nao_msgs/Bumper.h>

#include <cmath>

#include <iostream>

class Executor{
protected:
	//non-const Executor information 
	int* startTime;// number of seconds after action has concluded to start looking for contingency
	int* endTime;// number of seconds after starting to look for contingency that we stop #[7,7,2.197]

	// ROS topics
	ros::NodeHandle n;
	ros::Rate* r;
	ros::Subscriber sub_run, sub_bumper;
	ros::Publisher pub_nextAct;

	// callbacks
	void runCallback(const std_msgs::Bool::ConstPtr& msg);
	void bumperCallback(const nao_msgs::Bumper& msg);
	void callAction(int action);
	virtual int getNextAct(int act);

public:
	Executor(ros::NodeHandle);
	~Executor();

	void run();
};

#endif