/*
main.cpp
Madison Clark-Turner
10/24/2017
*/
#include "../include/interface/interface.hpp"
#include <QApplication>
#include <QtGui>

int main(int argc, char ** argv){
	ros::init(argc, argv, "interface");
	QApplication a(argc, argv);
	ASDInterface w;
	w.show();
	
	return a.exec();
}
