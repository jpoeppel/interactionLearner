cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set (CMAKE_CXX_FLAGS "-g -Wall")
include (FindPkgConfig)
if (PKG_CONFIG_FOUND)
  pkg_check_modules(GAZEBO gazebo)
  pkg_check_modules(SDF sdformat)  
endif()

include(FindBoost)
find_package(Boost ${MIN_BOOST_VERSION} REQUIRED system filesystem regex)

find_package(Protobuf REQUIRED)

include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

add_executable(listenerTest listenerTest.cc)
target_link_libraries(listenerTest ${GAZEBO_LIBRARIES} ${Boost_LIBRARIES} ${PROTOBUF_LIBRARIES} pthread)

add_executable (publisherTest publisherTest.cc)
target_link_libraries(publisherTest ${GAZEBO_LIBRARIES} ${Boost_LIBRARIES} ${PROTOBUF_LIBRARIES} ${SDF_LIBRARIES})

