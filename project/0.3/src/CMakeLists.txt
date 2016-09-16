add_library( myslam SHARED
    frame.cpp
    mappoint.cpp
    map.cpp
    camera.cpp
    config.cpp
    g2o_types.cpp
    visual_odometry2.cpp
)

target_link_libraries( myslam
    ${THIRD_PARTY_LIBS}
)