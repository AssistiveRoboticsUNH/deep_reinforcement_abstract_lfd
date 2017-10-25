cmake_minimum_required(VERSION 2.8.3)
project(interface_ros_functions)

find_package(catkin REQUIRED COMPONENTS
  nao_interaction_msgs
  nao_msgs
  naoqi_bridge_msgs
  roscpp
  std_msgs
  message_generation
)

# Generate Msg

add_message_files(
  FILES
  control_states.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

# Generate Executable

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(data_logger src/data_logger.cpp)
target_link_libraries(data_logger ${catkin_LIBRARIES})
add_dependencies(data_logger ${catkin_EXPORTED_TARGETS})

add_executable(wave_nao_right_hand src/wave_working.cpp)
target_link_libraries(wave_nao_right_hand ${catkin_LIBRARIES})
add_dependencies(wave_nao_right_hand ${catkin_EXPORTED_TARGETS})

add_executable(wave_nao_left_hand src/wave_working_left.cpp)
target_link_libraries(wave_nao_left_hand ${catkin_LIBRARIES})
add_dependencies(wave_nao_left_hand ${catkin_EXPORTED_TARGETS})
