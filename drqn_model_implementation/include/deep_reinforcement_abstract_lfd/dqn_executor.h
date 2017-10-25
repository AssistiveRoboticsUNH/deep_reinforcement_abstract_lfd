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

#ifndef DQNExecutor_H
#define DQNExecutor_H

#include "deep_reinforcement_abstract_lfd/DQNGetNextAction.h"
#include <iostream>

#include "executor.h"

class DQNExecutor: public Executor{
private:
	// service names
	std::string srv_nextact_name = "get_next_action";

	// subscribers
	ros::ServiceClient srv_nextact;

	// services calls
	int srvNextAct(int act, ros::Time, ros::Time);
	int getNextAct(int act);

public:
	DQNExecutor(ros::NodeHandle);
};

#endif