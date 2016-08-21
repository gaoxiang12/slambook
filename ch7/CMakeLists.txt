cmake_minimum_required( VERSION 2.8 )
project( vo1 )

find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( feature_extraction feature_extration.cpp )
target_link_libraries( feature_extraction ${OpenCV_LIBS} )