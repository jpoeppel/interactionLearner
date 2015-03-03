#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:44:37 2015

@author: jpoeppel
"""

import pygazebo
import trollius
from trollius import From
import time
import logging
from pygazebo.msg import world_stats_pb2
from pygazebo.msg import pose_pb2
from pygazebo.msg import vector3d_pb2
from pygazebo.msg import model_pb2
from pygazebo.msg import quaternion_pb2
from pygazebo.msg import joint_cmd_pb2
logging.basicConfig()


@trollius.coroutine
def test():
    print "test"
    manager = yield From(pygazebo.connect(('127.0.0.1', 11345)))

    manager.subscribe('/gazebo/default/model/info',
                  'gazebo.msgs.Model',
                  callback)
                  
    publisher = yield From(
                  manager.advertise('/gazebo/default/unit_sphere_1/joint_cmd',
                  'gazebo.msgs.JointCmd'))

    message = pygazebo.msg.joint_cmd_pb2.JointCmd()
#    message = model_pb2.Model()
#    message.name = "unit_sphere_1"
    message.position.x = 1.5
    message.position.y = -1
    message.position.z = 1.5
#    orientation = quaternion_pb2.Quaternion()
#    orientation.x = 0.5
#    orientation.y = -1
#    orientation.z = 1.5
#    orientation.w = 1
    
    while True:
        yield From(publisher.publish(message))
        yield From(trollius.sleep(1.0))


def callback(data):
#    message = pygazebo.msg.gz_string_pb2.GzString.FromString(data)
    message = model_pb2.Model.FromString(data)
    print('Received message:', message.pose.position.x)
#print pygazebo.msg.camera_cmd_pb2
loop = trollius.get_event_loop()
loop.run_until_complete(test())