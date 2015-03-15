#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:55:30 2015

@author: jpoeppel
"""

import pygazebo
import trollius
from trollius import From
from pygazebo.msg import model_pb2
from pygazebo.msg import model_v_pb2
from pygazebo.msg import gripperCommand_pb2
import logging
import numpy as np

logging.basicConfig()

GAZEBOCMDS = {"MOVE": 0, "GRAB": 1, "RELEASE": 2}

GRIPPERSTATES = {"OPEN":0, "CLOSED": 1}


class WorldObject(object):
    def __init__(self, model = None):
        self.pose =np.zeros(6)
        self.name = ""
        self.id = 0
        if model != None:
            self.parse(model)
        
    def parse(self, model):
        self.pose = np.array(model.pose.position._fields.values() + model.pose.orientation._fields.values())
        self.name = model.name
        self.id = model.id
        
    def __repr__(self):
        return "Name: " + str(self.name) + "\nId: " + str(self.id) + "\nPose: " + str(self.pose) 


class Gripper(WorldObject):
    def __init__(self):
        super(Gripper, self).__init__()
        self.state = GRIPPERSTATES["OPEN"]
        
    def __repr__(self):
        return super(Gripper,self).__repr__() + "\nState: " + GRIPPERSTATES.keys()[GRIPPERSTATES.values().index(self.state)]


class RawWorldState(object):
    
    def __init__(self):
        self.gripper = Gripper()
        self.objects = []
        
    def parseWorldState(self, models):
        for m in models:
            if m.name == "ground_plane":
                continue
            elif m.name == "gripper":
                self.gripper.parse(m)
            else:
                self.objects.append(WorldObject(m))
                
    def __repr__(self):
        return "Gripper: " + str(self.gripper) +"\nObjects: " + str(self.objects)


class Action(object):
    
    def __init__(self, cmd, direction= [0.0,0.0,0.0]):
        self.cmd = cmd
        self.direction = direction

class GazeboInterface():
    """
        Class that handles the interaction with gazebo.
    """
    
    def __init__(self):
         
        self.active = True

        
    @trollius.coroutine
    def loop(self):
        """
        Main Loop that keeps running until it is shutdown. Also sets up the
        subscriber and publisher
        """
        
        self.manager =  yield From(pygazebo.connect(('127.0.0.1', 11345)))
        self.manager.subscribe('/gazebo/default/worldstate',
                          'gazebo.msgs.Model_V',
                          self.modelsCallback)
                          
        self.publisher = yield From(
                self.manager.advertise('/gazebo/default/gripperMsg',
                    'gazeboPlugins.msgs.GripperCommand'))   
        
                          
        while self.active:
            yield From(trollius.sleep(0.1))
                          
                          
    def sendCommand(self, action):
        yield From(self.publisher.wait_for_listener())
        msg = pygazebo.msg.gripperCommand_pb2.GripperCommand()
        msg.cmd = action.cmd
        msg.direction.x = action.direction[0]
        msg.direction.y = action.direction[1]
        msg.direction.z = action.direction[2]
        yield From(self.publisher.publish(msg))
        
        
    def modelsCallback(self, data):
        """
        Callback function that registers gazebo model information
        
        Parameters
        ----------
        data: bytearry
            Protobuf bytearray containing a list of models
        """
        models = model_v_pb2.Model_V.FromString(data)
#        print 'Received # models message:', str(len(models.models))
#        print [str(m) for m in models.models]
        w = RawWorldState()
        w.parseWorldState(models.models)
        print w
    
    def stop(self):
        """
        Function to stop the loop.
        """
        self.active = False
    
if __name__ == "__main__":
    loop = trollius.get_event_loop()
    gi = GazeboInterface()
    loop.run_until_complete(gi.loop())
    