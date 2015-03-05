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
from pygazebo.msg import poses_stamped_pb2
from pygazebo.msg import world_stats_pb2
from pygazebo.msg import contacts_pb2
logging.basicConfig()


@trollius.coroutine
def test():
    print "test"
    manager = yield From(pygazebo.connect(('127.0.0.1', 11345)))
#
#    manager.subscribe('/gazebo/default/model/info',
#                  'gazebo.msgs.Model',
#                  modelCallback)
#    manager.subscribe('/gazebo/default/pose/info',
#                      'gazebo.msgs.PosesStamped',
#                      poseCallback)
                      
#    manager.subscribe('/gazebo/default/world_stats',
#                      'gazebo.msgs.WorldStatistics',
#                      worldCallback)
#    manager.subscribe('/gazebo/default/physics/contacts',
#                      'gazebo.msgs.Contacts',
#                      contactsCallback)
                  
#    publisher = yield From(
#                manager.advertise('/gazebo/default/model/modify',
#                  'gazebo.msgs.Model'))
                  
    publisher = yield From(
                manager.advertise('/gazebo/default/gripperMsg',
                    'gazebo.msgs.GzString'))                  
#                  manager.advertise('/gazebo/default/unit_sphere_1/joint_cmd',
#                  'gazebo.msgs.JointCmd'))
#                manager.advertise('/gazebo/default/pioneer2dx::right_wheel/joint_cmd',
#                          'gazebo.msgs.JointCmd'))
                    
    yield From(publisher.wait_for_listener())
    msg = pygazebo.msg.gz_string_pb2.GzString()
    msg.data = "blockC"


#    message = pygazebo.msg.joint_cmd_pb2.JointCmd()
#    message = model_pb2.Model()
#    message.name = "blockB"
#    message.id = 20
##    message.is_static = 0
#    message.pose.position.x = 0.5
#    message.pose.position.y = -1.5
#    message.pose.position.z = 2.0
#    message.pose.orientation.x = 0
#    message.pose.orientation.y = 0
#    message.pose.orientation.z = 0
#    message.pose.orientation.w = 0
    print "sending"
    yield From(publisher.publish(msg))
    
#    while True:
#        yield From(publisher.publish(msg))
#        yield From(trollius.sleep(0.1))
        
def contactsCallback(data):
#    message = pygazebo.msg.gz_string_pb2.GzString.FromString(data)#
    message = contacts_pb2.Contacts.FromString(data)
    print 'Received cntacts message:', str(message.contact)

def worldCallback(data):
#    message = pygazebo.msg.gz_string_pb2.GzString.FromString(data)#
    message = world_stats_pb2.WorldStatistics.FromString(data)
    print 'Received worldstats message:', str(message)

def poseCallback(data):
#    message = pygazebo.msg.gz_string_pb2.GzString.FromString(data)#
    message = poses_stamped_pb2.PosesStamped.FromString(data)
    print 'Received pose message:', str(message)

def modelCallback(data):
#    message = pygazebo.msg.gz_string_pb2.GzString.FromString(data)
    message = model_pb2.Model.FromString(data)
    print 'Received model message:', str(message)
#print pygazebo.msg.camera_cmd_pb2
loop = trollius.get_event_loop()
loop.run_until_complete(test())