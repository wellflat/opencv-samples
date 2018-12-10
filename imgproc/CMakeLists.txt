cmake_minimum_required(VERSION 3.5.1)
project(lsd_sample)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-Wall -g")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(lsd_sample lsd_sample.cpp)
target_link_libraries(lsd_sample ${OpenCV_LIBRARIES})

