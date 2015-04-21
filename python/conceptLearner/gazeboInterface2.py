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
import copy


logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
import model2 as model
import model4

#logging.basicConfig()
#
#GAZEBOCMDS = { "NOTHING": 0,"MOVE": 1, "GRAB": 2, "RELEASE": 3}
#
#GRIPPERSTATES = {"OPEN":0, "CLOSED": 1}


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

#
#class Action(object):
#    
#    def __init__(self, cmd=GAZEBOCMDS["MOVE"], direction=np.array([0.0,0.0,0.0])):
#        self.cmd = cmd
#        self.direction = direction
#        
#    def score(self, otherAction):
#        if self.cmd == otherAction.cmd:
#            #Cosine similarity
#            return 1 + abs(self.direction.dot(otherAction.direction)/(np.linalg.norm(self.direction)*np.linalg.norm(otherAction.direction)))
#        else:
#            return 0
#            
#    def __repr__(self):
#        return "Action with direction: " + str(self.direction)

class GazeboInterface():
    """
        Class that handles the interaction with gazebo.
    """
    
    def __init__(self):
         
        self.active = True
        self.lastState = None
        self.worldModel = model4.ModelCBR()
        self.lastPrediction = None
        self.lastAction = model4.Action()

        
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
        
        
            
#        action["mvDir"] *= 0.5
        msg.direction.x = action["mvDir"][0]
        msg.direction.y = action["mvDir"][1]
        msg.direction.z = 0.0
#        msg.direction.z = action.direction[2] # ignore z for now
#        print msg
        self.publisher.publish(msg)
        
    def sendPrediction2(self):
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

    def sendPrediction(self):
        msg = pygazebo.msg.modelState_v_pb2.ModelState_V()
        
        
        for intState in self.lastPrediction.interactionStates.values():
            tmp = pygazebo.msg.modelState_pb2.ModelState()
            tmp.name =intState["sname"] +"Shadow"
            tmp.id = 99
            tmp.pose.position.x = intState["spos"][0]
            tmp.pose.position.y = intState["spos"][1]
            tmp.pose.position.z = intState["spos"][2]
            tmp.pose.orientation.x = intState["sori"][0]
            tmp.pose.orientation.y = intState["sori"][1]
            tmp.pose.orientation.z = intState["sori"][2]
            tmp.pose.orientation.w = intState["sori"][3]
            msg.models.extend([tmp])
        self.predictPublisher.publish(msg)
        
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
#        w = model.WorldState()
        w = model4.WorldState()
        #w2.parse(worldState)
        
        w.parse(worldState)
        if self.lastPrediction != None:
            self.worldModel.update(self.lastState, self.lastAction,self.lastPrediction, w)
            
        tmp = self.worldModel.getAction(w)
        
        norm = np.linalg.norm(tmp["mvDir"])
        if norm > 1:
            tmp["mvDir"] /= 2*norm
#        tmp = self.getRightAction()
#        tmp["cmd"] = GAZEBOCMDS["MOVE"]
#        tmp["dir"] = np.array([0.0,-1.2,0.0])
        self.lastAction = tmp
#        print "lastAction: " + str(self.lastAction)
        
        self.lastState = w
        self.lastPrediction = self.worldModel.predict(w, self.lastAction)
        self.sendPrediction()
        
        self.sendCommand(self.lastAction)
        print "num cases: " + str(len(self.worldModel.cases))
        print "num abstract cases: " + str(len(self.worldModel.abstractCases))
        print "num Predictions: ", self.worldModel.numPredictions
        print "% correctCase selected: ", self.worldModel.numCorrectCase/(float)(self.worldModel.numPredictions)
        if self.worldModel.numPredictions == 31206:
            raise Exception("Finished")
            
        if len(self.worldModel.cases) == 1000 or len(self.worldModel.cases) == 1001:
            self.worldModel.setTarget(self.getTarget(w))
#
#        for ac in self.worldModel.abstractCases:
#            print "number of refs: {} for abstract case variables: {}".format(len(ac.refCases),ac.variables)
#        print "abstract lists: " + str([c.variables for c in self.worldModel.abstractCases])


    def getTarget(self, worldState):
        gripper = None
        block = None
        for i in worldState.objectStates.values():
            if i["name"] == "gripper":
                gripper = copy.deepcopy(i)
            if i["name"] == "blockA":
                block = copy.deepcopy(i)
                
        
        gripper["pos"] = np.array([0.0,0.0,0.03])
        gripper["linVel"] = np.array([0.0,0.0,0.0])
        intState = model4.InteractionState(0, gripper)
        intState.relKeys = ["spos", "slinVel"]
        intState.fill(block)
        intState.weights["spos"] = 30
        return intState
    def getRightAction(self):
        return model.Action(cmd = GAZEBOCMDS["MOVE"], direction=np.array([0.5,0.0,0.0]))

    
    def stop(self):
        """
        Function to stop the loop.
        """
        self.active = False
    
if __name__ == "__main__":


    loop = trollius.get_event_loop()
    gi = GazeboInterface()
    loop.run_until_complete(gi.loop())
    