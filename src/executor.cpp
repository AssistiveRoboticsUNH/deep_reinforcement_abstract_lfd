/*
pomdpexecutor.cpp
Madison Clark-Turner
10/14/2017
*/

#include "deep_reinforcement_abstract_lfd/executor.h"

Executor::Executor(ros::NodeHandle node): n(node)
{	
	//set times when listening occurs
	startTime = new int[3];
	startTime[0] = 0;
	startTime[1] = 0;
	startTime[2] = 0;
	endTime = new int[3];
	endTime[0] = startTime[0]+10;
	endTime[1] = startTime[1]+10;
	endTime[2] = startTime[2]+0;

	//setup ROS variables
	r = new ros::Rate(30);
	sub_run = n.subscribe("/asdpomdp/run_asd_auto", 100, &Executor::runCallback, this);
	sub_bumper = n.subscribe("/bumper", 100, &Executor::bumperCallback, this);

	pub_nextAct = n.advertise<std_msgs::Int8>("/asdpomdp/next_action", 100);
}

Executor::~Executor(){
	delete[] startTime;
	delete[] endTime;
	delete r;
}

void Executor::run(){
	//begin a session of the behavioral intervention 
	ROS_INFO("Beginning behavioral intervention.");
	
	int nextact = 0;
	while(nextact >= 0 && nextact < 2){
		callAction(nextact);
		nextact = getNextAct(nextact);
	}
	callAction(nextact);
	if(nextact == 2){
		ros::Duration(startTime[nextact]).sleep();
		callAction(3);
	}
	ROS_INFO("Ending behavioral intervention.");
}

void Executor::callAction(int action){
	//executes an action
	std_msgs::Int8 performAction;
	performAction.data = action;
	pub_nextAct.publish(performAction);
	ROS_INFO("executed action: %d", action);
}

int Executor::getNextAct(int act){
	int nextact = -1;

	ros::Time begin = ros::Time::now();
	ros::Time startRecord = begin + ros::Duration(startTime[act]);
	ros::Time endRecord = begin + ros::Duration(endTime[act]);

	// Delay from action start to beginning of listening period.
	while(ros::ok() && ros::Time::now() < startRecord){
		ros::spinOnce();
	}

	while(ros::ok() && ros::Time::now() < endRecord){
		// Gather observtaion information during listening period.
		ros::spinOnce();
	}
	
	//PARSE DATA STREAM TO PACKAGER

	return nextact;
}

void Executor::runCallback(const std_msgs::Bool::ConstPtr& msg){
	run();
}

void Executor::bumperCallback(const nao_msgs::Bumper& msg){
	if(msg.left && msg.state)
		run();
}