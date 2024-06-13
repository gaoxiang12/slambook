cmake_minimum_required( VERSION 2.8 )
project( useEigen )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-O3" )

find_package( Eigen3 )
# 添加Eigen头文件
include_directories( ${EIGEN3_INCLUDE_DIRS} )

add_executable( eigenMatrix eigenMatrix.cpp )
