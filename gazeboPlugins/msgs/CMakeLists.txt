find_package(Protobuf REQUIRED)

set(PROTOBUF_IMPORT_DIRS)
foreach(ITR ${GAZEBO_INCLUDE_DIRS})
  if(ITR MATCHES ".*gazebo-[0-9.]+$")
    set(PROTOBUF_IMPORT_DIRS "${ITR}/gazebo/msgs/proto")
  endif()
endforeach()

set (msgs
  /homes/jpoeppel/master/masterThesis/gazeboPlugins/msgs/gripperCommand.proto
  ${PROTOBUF_IMPORT_DIRS}/vector3d.proto
  ${PROTOBUF_IMPORT_DIRS}/header.proto
  ${PROTOBUF_IMPORT_DIRS}/time.proto
)

    
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${msgs})
execute_process(COMMAND echo ${PROTOBUF_IMPORT_DIRS})
execute_process(COMMAND protoc -I=/homes/jpoeppel/master/masterThesis/gazeboPlugins/msgs/ --proto_path=/usr/include/gazebo-2.2/gazebo/msgs/proto --python_out=/homes/jpoeppel/master/pygazebo/pygazebo/msg/ ${msgs})


add_library(gripperCommand SHARED ${PROTO_SRCS})
target_link_libraries(gripperCommand ${PROTOBUF_LIBRARY})
