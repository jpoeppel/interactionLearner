#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:09:14 2015

@author: jpoeppel
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:58:30 2015

@author: jpoeppel
"""

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
from pygazebo.msg import world_reset_pb2
from pygazebo.msg import world_control_pb2
import logging
import numpy as np
import math
import copy
from common import GAZEBOCMDS
import math
import common

import state2 as state

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)



logging.basicConfig()


FREE_EXPLORATION = 0
PUSHTASK = 1
PUSHTASKSIMULATION = 2
MOVE_TO_TARGET = 3
MODE = PUSHTASKSIMULATION
#MODE = FREE_EXPLORATION


RANDOM_BLOCK_ORI = False
#RANDOM_BLOCK_ORI = True

DIFFERENTBLOCKORIENTATION = True
DIFFERENTBLOCKORIENTATION = False

DIRECTIONGENERALISATION = True
DIRECTIONGENERALISATION = False

SINGLE_INTSTATE= True

NUM_TRAIN_RUNS = 20
NUM_TEST_RUNS = 40

class GazeboInterface():
    """
        Class that handles the interaction with gazebo.
    """
    
    def __init__(self):
         
        self.active = True
        self.lastState = None
        self.worldModel = model.ModelCBR()
        self.lastPrediction = None
        self.lastAction = model.Action()
        
        self.trainRun = 0
        self.testRun = 0
        self.runStarted = False
        
        self.stepCounter = 0
        self.times = 0
        
        self.gripperErrorPos = 0.0
        self.gripperErrorOri = 0.0
        self.gripperErrors = []
        self.tmpGripperErrorPos = 0.0
        self.tmpGripperErrorOri = 0.0
        self.blockErrorPos = 0.0
        self.blockErrorOri = 0.0
        self.blockErrors = []
        self.tmpBlockErrorPos = 0.0
        self.tmpBlockErrorOri = 0.0
        self.direction = np.array([0.0,0.5,0.0])
        self.inputSeq = []
        self.outputSeq = []
        np.random.seed(1234)
        
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
                          
        self.posePublisher = yield From(
                self.manager.advertise('/gazebo/default/poses',
                    'gazebo.msgs.Pose'))
                    
                          
        self.cmdPublisher = yield From(
                self.manager.advertise('/gazebo/default/gripperMsg',
                    'gazeboPlugins.msgs.GripperCommand'))   
                    
        self.worldControlPublisher = yield From(
                self.manager.advertise('/gazebo/default/world_control',
                    'gazebo.msgs.WorldControl'))

        while self.active:
            yield From(trollius.sleep(0.1))
                          
                          
    def sendCommand(self, action):
        """
        Function to send a gripperCommand to gazebo
        
        Parameters
        ----------
        action: model.Action
            The action that should be send to gazebo
        """
        msg = pygazebo.msg.gripperCommand_pb2.GripperCommand()
        msg.cmd = action["cmd"]
        
        msg.direction.x = action["mvDir"][0]
        msg.direction.y = action["mvDir"][1]
        msg.direction.z = 0.0
        self.cmdPublisher.publish(msg)
        
    def sendStopCommand(self):
        """
        Function to send a stop command to gazebo to stop the gripper from moving.
        """
        msg = pygazebo.msg.gripperCommand_pb2.GripperCommand()
        msg.cmd = GAZEBOCMDS["MOVE"]
        msg.direction.x = 0.0
        msg.direction.y = 0.0
        msg.direction.z = 0.0
        self.cmdPublisher.publish(msg)
        
    def sendPrediction(self, prediction):
        """
        Function to send the last prediction to gazebo. This will move the shadow model
        positions.
        """
        raise NotImplemented
        msg = pygazebo.msg.modelState_v_pb2.ModelState_V()
        for intState in self.lastPrediction.interactionStates.values():
#            tmp = self.getModelState(intState["sname"] + "Shadow", intState["spos"], intState["sori"], 
#                                     self.lastPrediction.transM, self.lastPrediction.quat)
            tmp = self.getModelState(intState["sname"] + "Shadow", intState["spos"], intState["seuler"], 
                             self.lastPrediction.transM, self.lastPrediction.ori)

            msg.models.extend([tmp])
            if SINGLE_INTSTATE:
                tmp = self.getModelState(intState["oname"] + "Shadow", intState["spos"]+intState["dir"], intState["seuler"]+intState["deuler"], 
                             self.lastPrediction.transM, self.lastPrediction.ori)

                msg.models.extend([tmp])
        self.posePublisher.publish(msg)
        
        

    def getModelState(self, name, pos, euler, transM=None, eulerdif=None):
        """
        Function to create a ModeLState object from a name and the desired position and orientation.
        
        Parameters
        ----------
        name : String
            Name of the model
        pos : np.array() Size 3
            The position of the model
        ori : np.array() Size 4
            The quaternion for the orientation of the model
            
        Returns
        -------
        modelState_pb2.ModelState
            Protobuf object for a model state with fixed id to 99
        """
        msg = modelState_pb2.ModelState()
        msg.name = name
        msg.id = 99
#        print "getting model State for: ", name
        if transM != None:
            #Build inverse transformation matrix
            tmpPos = np.matrix(np.concatenate((pos,[1])))
            tpos = np.array((transM*tmpPos.T)[:3]).flatten()   
#            print "OriginalPosition: {}, resulting position: {}".format(tmpPos, pos)
            msg.pose.position.x = tpos[0] #* 2.0
            msg.pose.position.y = tpos[1] #* 2.0
            msg.pose.position.z = tpos[2] #* 2.0
        else:
            msg.pose.position.x = pos[0] #* 2.0
            msg.pose.position.y = pos[1] #* 2.0
            msg.pose.position.z = pos[2] #* 2.0
        
        if eulerdif != None:
            newOri = common.eulerToQuat(euler+eulerdif)
            msg.pose.orientation.x = newOri[0]
            msg.pose.orientation.y = newOri[1]
            msg.pose.orientation.z = newOri[2]
            msg.pose.orientation.w = newOri[3]
        else:
            ori= common.eulerToQuat(euler)
            msg.pose.orientation.x = ori[0]
            msg.pose.orientation.y = ori[1]
            msg.pose.orientation.z = ori[2]
            msg.pose.orientation.w = ori[3]
        return msg    
    
        
    def sendPose(self, name, pos, ori):
        msg = modelState_v_pb2.ModelState_V()
        msg.models.extend([self.getModelState(name, pos,ori)])
        self.posePublisher.publish(msg)
        
    def resetWorld(self):
        print "reset world"
        self.sendStopCommand()
        msg = world_control_pb2.WorldControl()
        msg.reset.all = True
        self.worldControlPublisher.publish(msg)
        self.lastAction = None
        self.lastPrediction = None
        self.lastState = None
        
    def pauseWorld(self):
        msg = world_control_pb2.WorldControl()
        msg.pause = True
        self.worldControlPublisher.publish(msg)
        
        
    def modelsCallback(self, data):
        """
        Callback function that registers gazebo model information
        
        Parameters
        ----------
        data: bytearry
            Protobuf bytearray containing a list of models
        """
        worldState = worldState_pb2.WorldState.FromString(data)
        if self.lastPrediction != None:
            resultWS = state.WorldState(self.lastPrediction.transM, self.lastPrediction.invTrans, self.lastPrediction.ori)
            resultWS.parse(worldState)
        else:
            resultWS = None
#        print "parsing new WorldState"
        newWS = state.WorldState()
        newWS.parse(worldState)
        
        if MODE == FREE_EXPLORATION:
            self.randomExploration(newWS, resultWS)
#        elif MODE == PUSHTASK:
#            self.pushTask(newWS, resultWS)
        elif MODE == PUSHTASKSIMULATION:
            self.pushTaskSimulation(newWS, resultWS)
#        elif MODE == MOVE_TO_TARGET:
#            self.moveToTarget(newWS, resultWS)
        else:
            raise AttributeError("Unknown MODE: ", MODE)


    def runEnded(self, worldState):
        """
        Function to determine if a run has ended. In this case a run has ended, when the
        gripper did not move more than a threshold compared to the last State.
        
        Parameters
        ----------
        worldState: model.WorldState
            The current world state.
        """
        gripperInt = worldState.getInteractionState("gripper")
        tmpPos = np.matrix(np.concatenate((gripperInt["spos"],[1])))
        tpos = np.array((worldState.transM*tmpPos.T)[:3]).flatten()   
        
        if np.linalg.norm(tpos) > 1.0 or self.stepCounter > 50:
            return True
        return False
        
    def runEnded2D(self, worldState):
        gripperInt = worldState.getInteractionState("gripper")
        tmpPos = np.matrix(np.concatenate((gripperInt["spos"],[1])))
        tpos = np.array((worldState.transM*tmpPos.T)[:2]).flatten()   
        
        if np.linalg.norm(tpos) > 1.0 or self.stepCounter > 50:
            return True
        return False

    def startRun(self, randomRange=0.5):
        """
        Function to start a new Run. This means that the inital action is set and the first
        prediction is made.

        """
        
        self.runStarted = True
         #Set up Starting position
        posX = ((np.random.rand()-0.5)*randomRange) #* 0.5
        self.sendPose("gripper", np.array([posX,0.0,0.03]), np.array([0.0,0.0,0.0,0.0]))
        if RANDOM_BLOCK_ORI:
            self.sendPose("blockA", np.array([0.0, 0.5, 0.05]) , np.array([0.0,0.0,1.0,np.random.rand()-0.5]))
        self.stepCounter = 0
        
    def startRun2(self, randomRange=0.5):
        self.runStarted = True
         #Set up Starting position
        posy = ((np.random.rand()-0.5)*randomRange) #* 0.5
        self.sendPose("gripper", np.array([0.0,posy,0.03]), np.array([0.0,0.0,0.0,0.0]))
        self.sendPose("blockA", np.array([-0.5, 0.0, 0.05]) , np.array([0.0,0.0,1.0,1.0]))
        self.stepCounter = 0
        
    def startRun3(self):
        self.runStarted = True
        self.sendPose("gripper", np.array([-0.5, 0.5,0.03]), np.array([0.0,0.0,1.0,-1.0]))
        self.stepCounter = 0
            
    def pushTaskSimulation(self, worldState, resultState=None):
        self.stepCounter += 1
        
        if self.runStarted:
            if self.runEnded(worldState):
                self.resetWorld()
                self.runStarted = False
                self.worldModel.update(self.inputSeq, self.outputSeq)
                self.inputSeq = []
                self.outputSeq = []
            else:
                self.inputSeq.append(self.getInputData(worldState))
        else:
            if self.testRun > 0:
                if DIRECTIONGENERALISATION:
                    self.startRun2(0.7)
                    self.direction = np.array([-0.5,0.0,0.0])
                elif DIFFERENTBLOCKORIENTATION:
                    self.startRun3()
                    self.direction = np.array([0.5,0.0,0.0])
                else:
                    self.startRun(0.7)
                    self.direction = np.array([0.0,0.5,0.0])
            else:
                self.startRun(0.7)
                self.direction = np.array([0.0,0.5,0.0])
            return
        if self.trainRun < NUM_TRAIN_RUNS:
            self.trainRun += 1
            
        elif self.testRun < NUM_TEST_RUNS:
            pred = self.worldModel.predict(self.getInputData(worldState))
            self.sendPrediction(pred)



    
    def stop(self):
        """
        Function to stop the loop.
        """
        self.active = False
    
if __name__ == "__main__":


    loop = trollius.get_event_loop()
    gi = GazeboInterface()
    loop.run_until_complete(gi.loop())
    