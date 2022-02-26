cmake_minimum_required( VERSION 2.8 )
project( geometry )

find_package( Eigen3 )
# 添加Eigen头文件
include_directories( ${EIGEN3_INCLUDE_DIRS} )

add_executable( eigenGeometry eigenGeometry.cpp )
