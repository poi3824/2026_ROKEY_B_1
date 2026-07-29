cmake_minimum_required(VERSION 3.8)
project(fms_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(action_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FleetJob.msg"
  "msg/FleetReport.msg"
  "msg/AmrGoal.msg"
  "msg/AmrStatus.msg"
  "msg/BusbarGrasp.msg"
  "msg/NutPose.msg"
  "srv/GetGraspPose.srv"
  "srv/GetBoltPair.srv"
  "action/ExecuteArmTask.action"
  DEPENDENCIES geometry_msgs builtin_interfaces action_msgs
)

ament_package()
