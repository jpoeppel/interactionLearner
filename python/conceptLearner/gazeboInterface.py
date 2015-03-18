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
from pygazebo.msg import modelState_pb2
from pygazebo.msg import modelState_v_pb2
from pygazebo.msg import gripperCommand_pb2
import logging
import numpy as np
import math

import model

logging.basicConfig()

GAZEBOCMDS = {"MOVE": 0, "GRAB": 1, "RELEASE": 2}

GRIPPERSTATES = {"OPEN":0, "CLOSED": 1}


class WorldObject(object):
    def __init__(self, model = None):
        self.pose =np.zeros(6)
        self.linVel = np.zeros(3)
        self.angVel = np.zeros(3)
        self.name = ""
        self.id = 0
        if model != None:
            self.parse(model)
        
    def parse(self, model):
        self.pose = np.array(model.pose.position._fields.values() + model.pose.orientation._fields.values())
        #TODO: make sure that the values in the vector3d fields are always sorted correctly!
        self.linVel = np.array(model.linVel._fields.values())
        self.angVel = np.array(model.angVel._fields.values())
        self.name = model.name
        self.id = model.id
        

    def toDict(self):
        d = {}
        d["pose"] = self.pose
        d["linVel"] = self.linVel
        d["angVel"] = self.angVel
        d["name"] = self.name
        d["id"] = self.id
        return d
        
    
    def __repr__(self):
        return "Name: " + str(self.name) + "\nId: " + str(self.id) + "\nPose: " + str(self.pose)  \
            + "\nLinVel: " + str(self.linVel) + "\nAngVel: " + str(self.angVel)


class Gripper(WorldObject):
    def __init__(self):
        super(Gripper, self).__init__()
        self.state = GRIPPERSTATES["OPEN"]
#        self.action = Action()
        
        
    def toDict(self):
        d = super(Gripper, self).toDict()
        d.update([("state", self.state)])
        return d
        
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
    
    def __init__(self, cmd=GAZEBOCMDS["MOVE"], direction= np.array([0.0,0.0,0.0])):
        self.cmd = cmd
        self.direction = direction
        
    def score(self, otherAction):
        if self.cmd == otherAction.cmd:
            #Cosine similarity
            return abs(self.direction.dot(otherAction.direction)/(np.linalg.norm(self.direction)*np.linalg.norm(otherAction.direction)))
        else:
            return 0
            
    def __repr__(self):
        return "Action with direction: " + str(self.direction)

class GazeboInterface():
    """
        Class that handles the interaction with gazebo.
    """
    
    def __init__(self):
         
        self.active = True
        self.lastState = None
        self.worldModel = model.ModelCBR()
        self.lastPrediction = None
        self.lastAction = Action()

        
    @trollius.coroutine
    def loop(self):
        """
        Main Loop that keeps running until it is shutdown. Also sets up the
        subscriber and publisher
        """
        
        self.manager =  yield From(pygazebo.connect(('127.0.0.1', 11345)))
        self.manager.subscribe('/gazebo/default/worldstate',
                          'gazebo.msgs.ModelState_V',
                          self.modelsCallback)
                          
        self.publisher = yield From(
                self.manager.advertise('/gazebo/default/gripperMsg',
                    'gazeboPlugins.msgs.GripperCommand'))   
        
                          
        while self.active:
            yield From(trollius.sleep(0.1))
                          
                          
    def sendCommand(self, action):
#        print "setting Action: " + str(action)
#        yield From(self.publisher.wait_for_listener())
        msg = pygazebo.msg.gripperCommand_pb2.GripperCommand()
        msg.cmd = action.cmd
        msg.direction.x = action.direction[0]
        msg.direction.y = action.direction[1]
        msg.direction.z = 0.0
#        msg.direction.z = action.direction[2] # ignore z for now
        self.publisher.publish(msg)
        
        
    def modelsCallback(self, data):
        """
        Callback function that registers gazebo model information
        
        Parameters
        ----------
        data: bytearry
            Protobuf bytearray containing a list of models
        """
        models = modelState_v_pb2.ModelState_V.FromString(data)
#        print 'Received world state', str(models)
        w = RawWorldState()
        w.parseWorldState(models.models)
        if self.lastPrediction != None:
            self.worldModel.update(self.lastAction,self.lastState, self.lastPrediction, w)
        tmp = self.worldModel.getAction(w)
        if tmp != None:
            self.lastAction = tmp
            self.sendCommand(self.lastAction)
#        print "lastAction: " + str(self.lastAction)
        
        self.lastState = w
        self.lastPrediction = self.worldModel.predict(self.lastAction, w)
        print "num cases:" + str(self.worldModel.numCases)
    
    def stop(self):
        """
        Function to stop the loop.
        """
        self.active = False
    
if __name__ == "__main__":


    loop = trollius.get_event_loop()
    gi = GazeboInterface()
    loop.run_until_complete(gi.loop())
    