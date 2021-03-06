find_package(Protobuf REQUIRED)
find_package(gazebo REQUIRED)

set(PROTOBUF_IMPORT_DIRS)
foreach(ITR ${GAZEBO_INCLUDE_DIRS})
  if(ITR MATCHES ".*gazebo-[0-9.]+$")
    set(PROTOBUF_IMPORT_DIRS "${ITR}/gazebo/msgs/proto")
  endif()
endforeach()

set (msgs
  $ENV{masterPath}/masterThesis/gazeboPlugins/msgs/actuatorCommand.proto
  $ENV{masterPath}/masterThesis/gazeboPlugins/msgs/modelState.proto
  $ENV{masterPath}/masterThesis/gazeboPlugins/msgs/modelState_v.proto
  $ENV{masterPath}/masterThesis/gazeboPlugins/msgs/worldState.proto
  ${PROTOBUF_IMPORT_DIRS}/vector3d.proto
  ${PROTOBUF_IMPORT_DIRS}/quaternion.proto
  ${PROTOBUF_IMPORT_DIRS}/pose.proto
  ${PROTOBUF_IMPORT_DIRS}/header.proto
  ${PROTOBUF_IMPORT_DIRS}/time.proto
  ${PROTOBUF_IMPORT_DIRS}/contacts.proto
  ${PROTOBUF_IMPORT_DIRS}/contact.proto
  ${PROTOBUF_IMPORT_DIRS}/joint_wrench.proto
  ${PROTOBUF_IMPORT_DIRS}/wrench.proto
)


PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${msgs})

execute_process(COMMAND protoc -I=$ENV{masterPath}/masterThesis/gazeboPlugins/msgs/ --proto_path=/usr/include/gazebo-2.2/gazebo/msgs/proto --python_out=$ENV{masterPath}/pygazebo/pygazebo/msg/ ${msgs})


add_library(actuatorCommand SHARED ${PROTO_SRCS})
target_link_libraries(actuatorCommand ${PROTOBUF_LIBRARY})
