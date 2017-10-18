/*
pomdpexecutor.cpp
Madison Clark-Turner
10/14/2017
*/

#include "deep_reinforcement_abstract_lfd/dqn_executor.h"

DQNExecutor::DQNExecutor(ros::NodeHandle node): Executor(node)
{	
	srv_nextact = n.serviceClient<deep_q_network::DQNGetNextAction>(srv_nextact_name);
}

int DQNExecutor::srvNextAct(int act, ros::Time start_listening_time, ros::Time end_listening_time){
	ros::Time begin = ros::Time::now();
	ros::Time startRecord = begin + ros::Duration(startTime[act]);
	ros::Time endRecord = begin + ros::Duration(endTime[act]);

	deep_q_network::DQNGetNextAction srv;
	srv.request.start_time = start_listening_time;
	srv.request.end_time = end_listening_time;
	srv.request.num_prompt = act;
	std::cout << start_listening_time << ' ' <<  end_listening_time << std::endl;
	int nextact = -1;
	if (srv_nextact.call(srv)){
		ROS_INFO("Call to service: %s, succesful!", srv_nextact_name.c_str());
		nextact = srv.response.next_act;
	}
	else{
		ROS_INFO("Call to service: %s, failed.", srv_nextact_name.c_str());
	}

	return nextact;
}

int DQNExecutor::getNextAct(int act){
	int nextact = -1;

	ros::Time begin = ros::Time::now();
	ros::Time startRecord = begin + ros::Duration(startTime[act]);
	ros::Time endRecord = begin + ros::Duration(endTime[act]);

	nextact = srvNextAct(act, startRecord, endRecord);

	return nextact;
}