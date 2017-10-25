/* Author: Mikhail Medvedev */

#ifndef DATA_LOGGER_H_
#define DATA_LOGGER_H_

#include <ros/ros.h>
#include <std_srvs/Empty.h>

namespace data_logger
{

class DataLogger
{
public:
  DataLogger();
  virtual ~DataLogger();

private:
  ros::NodeHandle nh_;
  ros::ServiceServer srv_start_;
  ros::ServiceServer srv_stop_;
  bool start(std_srvs::EmptyRequest &req, std_srvs::EmptyResponse &res);
  bool stop(std_srvs::EmptyRequest &req, std_srvs::EmptyResponse &res);

  pid_t bag_process_pid_;
  std::string rosbag_record_args_;
  std::string bag_path_;

};

}

#endif /* DATA_LOGGER_H_ */

