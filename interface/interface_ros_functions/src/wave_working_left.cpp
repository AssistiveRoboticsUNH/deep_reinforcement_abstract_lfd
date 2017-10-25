#include <ros/ros.h>
#include <string>
#include <interface_ros_functions/control_states.h>
#include <iostream>
#include <nao_msgs/JointAnglesWithSpeed.h>
#include <naoqi_bridge_msgs/BodyPoseActionGoal.h>


interface_ros_functions::control_states states;

void cb(const interface_ros_functions::control_states States){
	states = States;
}

int main(int argc, char ** argv){
	ros::init(argc, argv, "nao_wave_work");
	ros::NodeHandle n;

	ros::Subscriber sub_control = n.subscribe("/control_msgs", 100, cb);
	ros::Publisher pub_control = n.advertise<interface_ros_functions::control_states>("/control_msgs", 100);
	ros::Publisher pub_move = n.advertise<nao_msgs::JointAnglesWithSpeed>("/joint_angles", 100);
	ros::Publisher pub_pose = n.advertise<naoqi_bridge_msgs::BodyPoseActionGoal>("/body_pose/goal", 100);

	ros::Rate loop_rate(15);

	nao_msgs::JointAnglesWithSpeed /*hy, hr,*/hp,  ler, ley, lh, lsp, lsr, lwy, rer,rey, rh, rsp, rsr, rwy;

	hp.joint_names.push_back("HeadPitch");
	//hr.joint_names.push_back("HeadRoll");
	ler.joint_names.push_back("RElbowRoll");
	ley.joint_names.push_back("RElbowYaw");
	lh.joint_names.push_back("RHand");
	lsp.joint_names.push_back("RShoulderPitch");
	lsr.joint_names.push_back("RShoulderRoll");
	lwy.joint_names.push_back("RWristYaw");
	rer.joint_names.push_back("LElbowRoll");
	rey.joint_names.push_back("LElbowYaw");
	rh.joint_names.push_back("LHand");
	rsp.joint_names.push_back("LShoulderPitch");
	rsr.joint_names.push_back("LShoulderRoll");
	rwy.joint_names.push_back("LWristYaw");
	//hy.joint_angles.push_back(0);
	hp.joint_angles.push_back(0.1);
	ler.joint_angles.push_back(0);
	ley.joint_angles.push_back(0);
	lh.joint_angles.push_back(0);
	lsp.joint_angles.push_back(0);
	lsr.joint_angles.push_back(0);
	lwy.joint_angles.push_back(0);
	rer.joint_angles.push_back(0);
	rey.joint_angles.push_back(0);
	rh.joint_angles.push_back(0);
	rsp.joint_angles.push_back(0);
	rsr.joint_angles.push_back(0);
	rwy.joint_angles.push_back(0);

	naoqi_bridge_msgs::BodyPoseActionGoal pose;
	//std::cout << "WAVE WORKING!" << std::endl;
	ROS_INFO("WAVE WORKING!");
	int i;
	
	while(ros::ok()){
		ros::spinOnce();
		if(states.startwave1 == false && states.startwave2 == false && states.shutdown == false){
			ros::spinOnce();
		}
		else if(states.shutdown == true){
			ROS_INFO("SHUTTING DOWN WAVE");
			ros::shutdown();
		}
		else if(states.startwave1 == true){
			ROS_INFO("WAVING LEFT");
		
			rsr.joint_angles[0] = 0.3142;
			rsp.joint_angles[0] = -1;
			rer.joint_angles[0] = -0.0349;
			rwy.joint_angles[0] = 0;
			rsr.speed = 0.7;
			rsp.speed = 0.7;
			rer.speed = 0.7;
			rwy.speed = 0.7;
			pub_move.publish(rsr);
			pub_move.publish(rsp);
			pub_move.publish(rer);
			pub_move.publish(rwy);
			ros::Duration(0.5).sleep();
	
			for(i = 0; i < 5; i++){
				rsr.joint_angles[0] = -1;
				rer.joint_angles[0] = -0.7;
				rsr.speed = 0.5;
				rer.speed = 0.5;
				pub_move.publish(rsr);
				pub_move.publish(rer);
				ros::Duration(0.5).sleep();

                        	rsr.joint_angles[0] = 0.342;
                        	rer.joint_angles[0] = -0.349;
                        	rsr.speed = 0.5;
                        	rer.speed = 0.5;
                        	pub_move.publish(rsr);
                        	pub_move.publish(rer);
                        	ros::Duration(0.5).sleep();
			}

			rsp.joint_angles[0] = 1.4;
			rsp.speed = 0.5;
			pub_move.publish(rsp);
			loop_rate.sleep();
	
			pose.goal.pose_name = "Stand";
			pub_pose.publish(pose);					

			// at end publish false so does not loop
			states.startwave1 = false;
			pub_control.publish(states);
		}
		else if(states.startwave2 == true){
			ROS_INFO("WAVING LEFT 2");
			rsr.joint_angles[0] = -0.3142;
			rsp.joint_angles[0] = -1;
			rer.joint_angles[0] = -0.0349;
			rwy.joint_angles[0] = 0;
			rsr.speed = 0.7;
			rsp.speed = 0.7;
			rer.speed = 0.7;
			rwy.speed = 0.7;
			pub_move.publish(rsr);
			pub_move.publish(rsp);
			pub_move.publish(rer);
			pub_move.publish(rwy);
			ros::Duration(0.5).sleep();
	
			for(i = 0; i < 5; i++){
				rsr.joint_angles[0] = 1;
				rer.joint_angles[0] = -0.7;
				rsr.speed = 0.5;
				rer.speed = 0.5;
				pub_move.publish(rsr);
				pub_move.publish(rer);
				ros::Duration(0.5).sleep();

                        	rsr.joint_angles[0] = -0.342;
                        	rer.joint_angles[0] = -0.349;
                        	rsr.speed = 0.5;
                        	rer.speed = 0.5;
                        	pub_move.publish(rsr);
                        	pub_move.publish(rer);
                        	ros::Duration(0.5).sleep();
			}

			rsp.joint_angles[0] = 1.4;
			rsp.speed = 0.5;
			pub_move.publish(rsp);
			loop_rate.sleep();

				
			//pose.goal.pose_name = "Stand";
			//pub_pose.publish(pose);			

			states.startwave2 = false;
			pub_control.publish(states);
		}
	}
	pub_move.publish(hp);
	return 0;
}
