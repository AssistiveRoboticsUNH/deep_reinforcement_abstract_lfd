/*
interface.hpp
Madison Clark-Turner
10/24/2017
*/
#ifndef INTERFACE_H
#define INTERFACE_H

#include <QWidget>
#include <QImage>
#include <QPainter>
#include <QPaintDevice>
#include <QtGui>
#include <QLCDNumber>
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <nao_msgs/JointAnglesWithSpeed.h>
#include <std_msgs/String.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Bool.h>
#include <naoqi_bridge_msgs/BodyPoseActionGoal.h>
#include <std_srvs/Empty.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <string>
#include <interface_ros_functions/control_states.h>
#include <nao_msgs/JointAnglesWithSpeed.h>

#include <tf/transform_listener.h>

namespace Ui{
	class ASDInterface;
}

class ASDInterface : public QWidget{
	Q_OBJECT

	public:
		std::string name = "";
		explicit ASDInterface(QWidget *parent = 0);
		~ASDInterface();

		void imageCallback(const sensor_msgs::ImageConstPtr& msg);
		void controlCallback(const interface_ros_functions::control_states States);
		void actionCallback(const std_msgs::Int8& msg);
		void readyCallback(const std_msgs::Bool& msg);

		void UpdateImage();
		void loopRate(int loop_rates);
		std::string getTimeStamp();
		void waveNao();
		void centerGaze();
	
	private Q_SLOTS:
		void on_Command_clicked();
		void on_Prompt_clicked();
		void on_Respond_clicked();
		void on_Bye_clicked();

		void on_StartRecord_clicked();
		void on_StopRecord_clicked();

		void on_Stand_clicked();
		void on_Rest_clicked();
		void on_AngleHead_clicked();
		void on_ToggleLife_clicked();
		
		void on_Start_clicked();
		void on_ShutDown_clicked();
		void on_Run_clicked();
		
		void on_MyClock_overflow();
		
	protected:
		QRect genRectangle(QWidget* tl, QWidget* br);
		void paintEvent(QPaintEvent *event);
		void timerEvent(QTimerEvent *event);

	private:
		Ui::ASDInterface *ui;
		QBasicTimer Mytimer;
		QTimer *timer;
		QString MyClockTimetext;
		QPixmap* icons;

		ros::NodeHandle n;
		ros::Publisher pub_custom, pub_move, pub_pose, pub_run, pub_speak, pub_act;
		ros::ServiceClient client_stiff, client_record_start, client_record_stop, client_wakeup, client_rest, life_enable, life_disable;
		ros::Subscriber sub_cam, sub_custom, sub_nextAct, sub_dqnReady;

		QImage NaoImg;
		int count;
		bool recording = false;
		bool life_on = true;
		interface_ros_functions::control_states controlstate;

		tf::TransformListener listener;
};

#endif
	
