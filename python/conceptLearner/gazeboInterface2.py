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
from pygazebo.msg import worldState_pb2
import logging
import numpy as np
import math

import model2 as model
import model3

logging.basicConfig()

GAZEBOCMDS = { "NOTHING": 0,"MOVE": 1, "GRAB": 2, "RELEASE": 3}

GRIPPERSTATES = {"OPEN":0, "CLOSED": 1}


class WorldObject(object):
    def __init__(self, model = None):
        self.pos =np.zeros(3)
        self.orientation = np.zeros(3)
        self.linVel = np.zeros(3)
        self.angVel = np.zeros(3)
        self.name = ""
        self.id = 0
        if model != None:
            self.parse(model)
        
    def parse(self, model):
        self.pos = np.array(model.pose.position._fields.values())
        self.orientation = np.array(model.pose.orientation._fields.values())
        #TODO: make sure that the values in the vector3d fields are always sorted correctly!
        self.linVel = np.array(model.linVel._fields.values())
        self.angVel = np.array(model.angVel._fields.values())
        self.name = model.name
        self.id = model.id
        

    def toDict(self):
        d = {}
        d["pos"] = self.pos
        d["orientation"] = self.orientation
        d["linVel"] = self.linVel
        d["angVel"] = self.angVel
        d["name"] = self.name
        d["id"] = self.id
        return d
        
    def __eq__(self, other): 
        return np.array_equal(self.__dict__,other.__dict__)
        
    def __ne__(self, other):
        return not self.__eq__(other)
        
    
    def __repr__(self):
        return "Name: " + str(self.name) + "\nId: " + str(self.id) + "\nPos: " + str(self.pos)  \
            + "\nOri: " + str(self.orientation) + "\nLinVel: " + str(self.linVel) + "\nAngVel: " + str(self.angVel)


class Gripper(WorldObject):
    def __init__(self):
        super(Gripper, self).__init__()
        self.state = GRIPPERSTATES["OPEN"]
#        self.action = Action()
        
        
    def toDict(self):
        d = super(Gripper, self).toDict()
        d.update([("state", self.state)])
        return d
        
#    def __eq__(self, other): 
#        return super(Gripper, self).__eq__(other) and self.state == other.state
#        
#    def __ne__(self, other):
#        return not self.__eq__(other)
        
    def __repr__(self):
        return super(Gripper,self).__repr__() + "\nState: " + GRIPPERSTATES.keys()[GRIPPERSTATES.values().index(self.state)]


class RawWorldState(object):
    
    def __init__(self):
        self.gripper = Gripper()
        self.objects = []
        self.contacts = []
        
    def parseWorldState(self, models):
        for m in models:
            if m.name == "ground_plane":
                continue
            elif m.name == "gripper":
                self.gripper.parse(m)
            else:
                self.objects.append(WorldObject(m))
                
    def __eq__(self, other):         
        return isinstance(other, self.__class__) and self.gripper == other.gripper #TODO extend to objects and contacts!
        
    def __ne__(self, other):
        return not self.__eq__(other)
            
                
    def __repr__(self):
        return "Gripper: " + str(self.gripper) #+"\nObjects: " + str(self.objects)


class Action(object):
    
    def __init__(self, cmd=GAZEBOCMDS["MOVE"], direction=np.array([0.0,0.0,0.0])):
        self.cmd = cmd
        self.direction = direction
        
    def score(self, otherAction):
        if self.cmd == otherAction.cmd:
            #Cosine similarity
            return 1 + abs(self.direction.dot(otherAction.direction)/(np.linalg.norm(self.direction)*np.linalg.norm(otherAction.direction)))
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
                          'gazebo.msgs.WorldState',
                          self.modelsCallback)
                          
        self.predictPublisher = yield From(
                self.manager.advertise('/gazebo/default/predictions',
                    'gazebo.msgs.Pose'))
#                    'gazeboPlugins.msgs.ModelState_V'))
                    
                          
        self.publisher = yield From(
                self.manager.advertise('/gazebo/default/gripperMsg',
                    'gazeboPlugins.msgs.GripperCommand'))   
                    
        
        
                          
        while self.active:
            yield From(trollius.sleep(0.1))
                          
                          
    def sendCommand(self, action):
#        print "setting Action: " + str(action)
#        yield From(self.publisher.wait_for_listener())
        msg = pygazebo.msg.gripperCommand_pb2.GripperCommand()
        msg.cmd = action["cmd"]
        msg.direction.x = action["dir"][0]
        msg.direction.y = action["dir"][1]
        msg.direction.z = 0.0
#        msg.direction.z = action.direction[2] # ignore z for now
#        print msg
        self.publisher.publish(msg)
        
    def sendPrediction(self):
        print "sending prediction"
        msg = pygazebo.msg.modelState_v_pb2.ModelState_V()
        gripper = pygazebo.msg.modelState_pb2.ModelState()
        gripper.name = "gripperShadow"
#        for k in self.lastPrediction.gripperState.keys():
        gripper.id = 99
        
        gripper.pose.position.x = self.lastPrediction.gripperState["pos"][0]
        gripper.pose.position.y = self.lastPrediction.gripperState["pos"][1]
        gripper.pose.position.z = self.lastPrediction.gripperState["pos"][2]
        gripper.pose.orientation.x = self.lastPrediction.gripperState["orientation"][0]
        gripper.pose.orientation.y = self.lastPrediction.gripperState["orientation"][1]
        gripper.pose.orientation.z = self.lastPrediction.gripperState["orientation"][2]
        gripper.pose.orientation.w = self.lastPrediction.gripperState["orientation"][3]
        msg.models.extend([gripper])
        self.predictPublisher.publish(msg)
#        pose = pygazebo.msg.pose_pb2.Pose()
#        pose.position.x = self.lastPrediction.gripperState["pos"][0]
#        pose.position.y = self.lastPrediction.gripperState["pos"][1]
#        pose.position.z = self.lastPrediction.gripperState["pos"][2]
#        
#        self.predictPublisher.publish(pose)
        
    def modelsCallback(self, data):
        """
        Callback function that registers gazebo model information
        
        Parameters
        ----------
        data: bytearry
            Protobuf bytearray containing a list of models
        """
        worldState = worldState_pb2.WorldState.FromString(data)
#        print 'Received world state', str(models)
        w = model.WorldState()
        #w2 = model3.WorldState()
        #w2.parse(worldState)
        
        w.parse(worldState.model_v.models)
        if self.lastPrediction != None:
            self.worldModel.update(self.lastState, self.lastAction,self.lastPrediction, w, self.lastCase)
        tmp = self.getAction()
#        tmp = self.getRightAction()
#        tmp["cmd"] = GAZEBOCMDS["MOVE"]
#        tmp["dir"] = np.array([0.0,-1.2,0.0])
        if tmp != None:
            self.lastAction = tmp
            self.sendCommand(self.lastAction)
#        print "lastAction: " + str(self.lastAction)
        
        self.lastState = w
        self.lastPrediction, self.lastCase = self.worldModel.predict(w, self.lastAction)
        self.sendPrediction()
        print "num cases: " + str(len(self.worldModel.cases))
        print "num abstract cases: " + str(len(self.worldModel.abstractCases))
        print "abstract lists: " + str([c.gripperAttribs for c in self.worldModel.abstractCases])


    def getRightAction(self):
        return model.Action(cmd = GAZEBOCMDS["MOVE"], direction=np.array([0.5,0.0,0.0]))

    def getAction(self):
        rnd = np.random.rand()
        a = model.Action()
        
        if rnd < 0.5:
            a["cmd"] = GAZEBOCMDS["MOVE"]
#            a["dir"] = np.array([1.2,0,0])
            a["dir"] = np.random.rand(3)*2-1
#        elif rnd < 0.4:
#            a["dir"] = np.array([-1.3,0,0])
#        elif rnd < 0.6:
#            a["dir"] = np.array([0,1,0])
#        elif rnd < 0.8:
#            a["dir"] = np.array([0,-1,0])
        elif rnd < 0.6:
            a["cmd"] = GAZEBOCMDS["MOVE"]
            a["dir"] = np.array([0,0,0])
        else:
            a["cmd"] = GAZEBOCMDS["NOTHING"]
        a["dir"] *= 2.0
        return a
    
    def stop(self):
        """
        Function to stop the loop.
        """
        self.active = False
    
if __name__ == "__main__":


    loop = trollius.get_event_loop()
    gi = GazeboInterface()
    loop.run_until_complete(gi.loop())
    