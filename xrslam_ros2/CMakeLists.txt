cmake_minimum_required(VERSION 3.16)
project(xrslam_ros2 LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(yaml-cpp REQUIRED)

# Build the upstream XRSLAM superbuild separately. The ROS adapter is not
# included from its top-level CMakeLists.txt and does not use catkin.
set(XRSLAM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "XRSLAM repository root")
set(XRSLAM_LIBRARY "" CACHE FILEPATH "Full path to separately built libxrslam.so")
if(NOT XRSLAM_LIBRARY)
  find_library(XRSLAM_LIBRARY NAMES xrslam
    HINTS "${XRSLAM_ROOT}/lib" "${XRSLAM_ROOT}/build-eigenplaces-cpu/xrslam-interface"
    REQUIRED)
endif()
if(NOT EXISTS "${XRSLAM_LIBRARY}")
  message(FATAL_ERROR "Build XRSLAM first and set -DXRSLAM_LIBRARY=/absolute/path/libxrslam.so")
endif()
if(NOT EXISTS "${XRSLAM_ROOT}/xrslam-interface/include/XRSLAM.h")
  message(FATAL_ERROR "XRSLAM_ROOT must point to the XRSLAM fork")
endif()

add_executable(xrslam_ros2_node src/xrslam_ros2_node.cpp)
target_include_directories(xrslam_ros2_node PRIVATE
  "${XRSLAM_ROOT}/xrslam-interface/include")
target_link_libraries(xrslam_ros2_node "${XRSLAM_LIBRARY}" yaml-cpp)
ament_target_dependencies(xrslam_ros2_node
  rclcpp sensor_msgs nav_msgs geometry_msgs std_msgs tf2 tf2_ros tf2_geometry_msgs)
target_compile_options(xrslam_ros2_node PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS xrslam_ros2_node DESTINATION lib/${PROJECT_NAME})
ament_package()
