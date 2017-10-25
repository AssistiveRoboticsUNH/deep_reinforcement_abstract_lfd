/*
ui_interface.hpp
Madison Clark-Turner
10/24/2017
*/
#ifndef UI_INTERFACE_H
#define UI_INTERFACE_H

#include <QVariant>
#include <QAction>
#include <QApplication>
#include <QButtonGroup>
#include <QHeaderView>
#include <QLCDNumber>
#include <QPushButton>
#include <QWidget>
#include <QPalette>
#include <QTextEdit>
#include <QLabel>
#include <QPixmap>

#include <QPainter>
#include <QRect>

#include <iostream>
#include <fstream>
#include <ros/package.h>

QT_BEGIN_NAMESPACE

class Ui_ASDInterface{
	public:
		//buttons
		QPushButton *Start;
		QPushButton *Command;
		QPushButton *Prompt;
		QPushButton *StartRecord;
		QPushButton *StopRecord;
		QPushButton *Respond;
		QPushButton *Bye;
		QPushButton *ShutDown;
		QPushButton *AngleHead;
		QPushButton *Rest;
		QPushButton *Stand;
		QPushButton *ToggleLife;
		QPushButton *Run;

		//interfaces
		QTextEdit *Name;
		QLCDNumber *MyClock;

		//indicators
		QLabel *ReadyIndicator;
		QLabel *ReadyText;
		QPixmap *icons;

		void setupUi(QWidget *ASDInterface){
			if(ASDInterface->objectName().isEmpty())
				ASDInterface->setObjectName(QString("Deep Reinforcement Abstract LFD"));
			int blockw = 173;
			int blockh = 40;
			int buffer = 20;
			int indicator_diameter = 30;

			ASDInterface->resize(600, 600);

			

			// Buttons
			Command = new QPushButton(ASDInterface);
			Command->setObjectName(QString("Command"));
			Command->setGeometry(QRect(20, 20, blockw, blockh));
			Respond = new QPushButton(ASDInterface);
			Respond->setObjectName(QString("Respond"));
			Respond->setGeometry(QRect(blockw + buffer*2, buffer, blockw, blockh));
			Prompt = new QPushButton(ASDInterface);
			Prompt->setObjectName(QString("Prompt"));
			Prompt->setGeometry(QRect(buffer, blockh + buffer*2, blockw, blockh));
			Bye = new QPushButton(ASDInterface);
			Bye->setObjectName(QString("Bye"));
			Bye->setGeometry(QRect(blockw + buffer*2, blockh + buffer*2, blockw, blockh));

			StartRecord = new QPushButton(ASDInterface);
			StartRecord->setObjectName(QString("StartRecord"));
			StartRecord->setGeometry(QRect(blockw*2 + buffer*3, buffer, blockw, blockh));
			StopRecord = new QPushButton(ASDInterface);
			StopRecord->setObjectName(QString("StopRecord"));
			StopRecord->setGeometry(QRect(blockw*2 + buffer*3, blockh + buffer*2, blockw, blockh));

			AngleHead = new QPushButton(ASDInterface);
			AngleHead->setObjectName(QString("AngleHead"));
			AngleHead->setGeometry(QRect(blockw*2 + buffer*3, blockh*2 + buffer*3, blockw, blockh));
			Stand = new QPushButton(ASDInterface);
			Stand->setObjectName(QString("Stand"));
			Stand->setGeometry(QRect(blockw + buffer*2, blockh*2 + buffer*3, blockw, blockh));
			Rest = new QPushButton(ASDInterface);
			Rest->setObjectName(QString("Rest"));
			Rest->setGeometry(QRect(blockw + buffer*2, blockh*3 + buffer*4, blockw, blockh));
			ToggleLife = new QPushButton(ASDInterface);
			ToggleLife->setObjectName(QString("ToggleLife"));
			ToggleLife->setGeometry(QRect(blockw*2 + buffer*3, blockh*3 + buffer*4, blockw, blockh));

			Start = new QPushButton(ASDInterface);
			Start->setObjectName(QString("Start"));
			Start->setGeometry(QRect(blockw*2 + buffer*3, 600-buffer*2-blockh*2, blockw, blockh));
			ShutDown = new QPushButton(ASDInterface);
			ShutDown->setObjectName(QString("ShutDown"));
			ShutDown->setGeometry(QRect(blockw*2 + buffer*3, 600-blockh-buffer, blockw, blockh));
			Run = new QPushButton(ASDInterface);
			Run->setObjectName(QString("Run"));
			Run->setGeometry(QRect(blockw*2 + buffer*3, blockh*4 + buffer*5, blockw, blockh));
			//Run->setEnabled(false);

			// Name box

			Name = new QTextEdit(ASDInterface);
			Name->setObjectName(QString("Name"));
			Name->setGeometry(QRect(buffer, blockh*2 + buffer*3, blockw, blockh));

			// Clock

			MyClock = new QLCDNumber(ASDInterface);
			MyClock->setObjectName(QString("MyClock"));
			MyClock->setGeometry(QRect(370, 375, 201, 81));

			// Ready Indicator
			
			ReadyText = new QLabel(ASDInterface);
			ReadyText->setText("DQN Running: ");
			ReadyText->setGeometry(QRect(buffer*2, blockh*3 + buffer*4, blockw, blockh));
			
			ReadyIndicator = new QLabel(ASDInterface);

			std::string path = ros::package::getPath("interface")+"/images/";
			std::string icon_names [] = {"red.gif", "green.gif"};
			int num_icons = 2;
			icons = new QPixmap[num_icons];
			
			for(int i = 0; i < num_icons; i++){
				if (!icons[i].load( (path+icon_names[i]).c_str())) {
					qWarning("Failed to load %s", icon_names[i].c_str());
				}
				icons[i] = icons[i].scaled(indicator_diameter, indicator_diameter);
			}
			
			ReadyIndicator->setPixmap(icons[0]);
			ReadyIndicator->setGeometry(QRect((blockw-indicator_diameter)+buffer, blockh*3 + buffer*4, blockh, blockh));

			retranslateUi(ASDInterface);
	
			QMetaObject::connectSlotsByName(ASDInterface);
		}

		void retranslateUi(QWidget *ASDInterface){
			ASDInterface->setWindowTitle(QApplication::translate("ASDInterface", "ASDInterface", 0));

			Command->setText(QApplication::translate("ASDInterface", "Command", 0));
			Respond->setText(QApplication::translate("ASDInterface", "Respond", 0));
			Prompt->setText(QApplication::translate("ASDInterface", "Prompt", 0));
			Bye->setText(QApplication::translate("ASDInterface", "Bye", 0));

			StartRecord->setText(QApplication::translate("ASDInterface", "Start Record", 0));
			StopRecord->setText(QApplication::translate("ASDInterface", "Stop Record", 0));

			AngleHead->setText(QApplication::translate("ASDInterface", "Angle Head", 0));
			Rest->setText(QApplication::translate("ASDInterface", "Rest", 0));
			Stand->setText(QApplication::translate("ASDInterface", "Stand", 0));
			ToggleLife->setText(QApplication::translate("ASDInterface", "Toggle Life", 0));
			
			Start->setText(QApplication::translate("ASDInterface", "Start", 0));
			ShutDown->setText(QApplication::translate("ASDInterface", "Shut Down", 0));
			Run->setText(QApplication::translate("ASDInterface", "Run", 0));

		}
};

namespace Ui{
	class ASDInterface: public Ui_ASDInterface {};
}

QT_END_NAMESPACE

#endif
