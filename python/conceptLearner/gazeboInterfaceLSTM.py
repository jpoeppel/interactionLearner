#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:09:14 2015

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

import modelLSTM as model

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

NUM_TRAIN_RUNS = 10000
NUM_TEST_RUNS = 100

class GazeboInterface():
    """
        Class that handles the interaction with gazebo.
    """
    
    def __init__(self):
         
        self.active = True
        self.lastState = None
        self.worldModel = model.ModelLSTM()#"testDataInput10.txt", "testDataOutput10.txt")
        self.lastPrediction = None
        self.lastAction = state.Action()
        
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
        
    def sendPrediction(self, prediction, worldState):
        """
        Function to send the last prediction to gazebo. This will move the shadow model
        positions.
        """
        msg = pygazebo.msg.modelState_v_pb2.ModelState_V()
        pred = np.round(prediction[0], 4)
        
        tmp = self.getModelState("gripper" + "Shadow", pred[8:11], pred[2:5], 
                         worldState.transM, worldState.ori)

        msg.models.extend([tmp])
        if SINGLE_INTSTATE:
            tmp = self.getModelState("blockA" + "Shadow", pred[8:11]+pred[25:28], pred[2:5]+pred[20:23],
                         worldState.transM, worldState.ori)

            msg.models.extend([tmp])
        self.posePublisher.publish(msg)
        
        

    def getModelState(self, name, pos, ori, transM=None, oridif=None):
        """
        Function to create a ModeLState object from a name and the desired position and orientation.
        
        Parameters
        ----------
        name : String
            Name of the model
        pos : np.array() Size 3
            The position of the model
        ori : np.array() Size 3 or 4
            The euler angles (3) or quaternion (4) for the orientation of the model
            
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
        
        if len(ori) == 3:
            if oridif != None:
                ori = common.eulerToQuat(ori+oridif)
            else:
                ori= common.eulerToQuat(ori)
        elif len(ori) != 4:
            raise AttributeError("ori must be either 3 euler angles or a 4 dim quaternion")

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
#        print "reset world"
        self.sendStopCommand()
        msg = world_control_pb2.WorldControl()
        msg.reset.all = True
        self.worldControlPublisher.publish(msg)
#        self.lastAction = None
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
        
        
                
        self.lastAction = state.Action(cmd = GAZEBOCMDS["MOVE"], direction=self.direction)
        
        if self.runStarted:
            if self.runEnded(worldState):
                self.resetWorld()
                self.runStarted = False
#                print "input sec: ", self.inputSeq
#                print "output sec: ", self.outputSeq
#                
            else:
                
                if self.trainRun < NUM_TRAIN_RUNS:
                    self.lastPrediction = worldState
                
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
            print "Train run #: ", self.trainRun
            if resultState != None:
                self.outputSeq.append(self.getOutputData(resultState))

            if self.runStarted:
                self.inputSeq.append(self.getInputData(worldState))
            else:                    
                print "len input: {}, len output: {}".format(len(self.inputSeq), len(self.outputSeq))
#                if len(self.inputSeq) > 0:
#                    self.worldModel.update(self.inputSeq, self.outputSeq)
#                    with open("testDataInput10.txt", "a") as f:
#                        np.savetxt(f, self.inputSeq)
#                        f.write("# New Seq\n")
#                    with open("testDataOutput10.txt", "a") as f:
#                        np.savetxt(f, self.outputSeq)
#                        f.write("# New Seq\n")
                self.inputSeq = []
                self.outputSeq = []

#                print "Training step"
                self.trainRun += 1
                if self.trainRun == NUM_TRAIN_RUNS:
                    self.pauseWorld()
            
            
        elif self.testRun < NUM_TEST_RUNS:
            print "Test run #: ", self.testRun
            if self.runStarted:
                if self.lastPrediction != None:
                    predictedWorldState = state.WorldState()
                    predictedWorldState.reset(self.lastPrediction)
                else:
                    print "lastPred = None"
                    predictedWorldState = worldState
                pred = self.worldModel.predict(np.array([self.getInputData(predictedWorldState)]))
                print "prediction: ", pred
                self.lastPrediction = self.createPrediction(pred[0], predictedWorldState) #TODO FIX!
                self.sendPrediction(pred, predictedWorldState)
            else:
                self.testRun += 1
                
        if self.lastAction != None:
            self.sendCommand(self.lastAction)
            
    def getInputData(self, worldState):
        gripperInt = worldState.getInteractionState("gripper")
#        print "gripperInt: ", gripperInt
#        print "lastAction: ", self.lastAction
        return np.concatenate((gripperInt.toVec(), self.lastAction.toVec()))
        
    def getOutputData(self, resultState):
        gripperInt = resultState.getInteractionState("gripper")
        return gripperInt.toVec()

    def createPrediction(self, pred, worldState):
        ws = copy.deepcopy(worldState)
        ws.getInteractionState("gripper").updateFromVec(pred)
        return ws

    
    def stop(self):
        """
        Function to stop the loop.
        """
        self.active = False
    
if __name__ == "__main__":


    loop = trollius.get_event_loop()
    gi = GazeboInterface()
    loop.run_until_complete(gi.loop())
    