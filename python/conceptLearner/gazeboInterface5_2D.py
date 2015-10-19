#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:32:53 2015
Adaption from interface5 in order to work with 2D coordinates
@author: jpoeppel
"""



import trollius
from trollius import From
import pygazebo
from pygazebo.msg import modelState_pb2
from pygazebo.msg import modelState_v_pb2
from pygazebo.msg import gripperCommand_pb2
from pygazebo.msg import worldState_pb2
from pygazebo.msg import sensor_pb2
from pygazebo.msg import physics_pb2
from pygazebo.msg import world_control_pb2
import logging
import numpy as np
np.set_printoptions(precision=3,suppress=True)
#import math
import copy
from common import GAZEBOCMDS
#import math
import common
import datetime

import config
from config import config
import itm

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

GATE = True
#GATE = False

if GATE:
    import modelGate_2D_config as model
else:
    import modelInteractions_config as model

#from sklearn.externals.six import StringIO
#import pydot


trainRuns = [5]
RECORD_SIMULATION = False
SIMULATION_FILENAME = "gateModel{}Runs_Gate_Act_NoDynsITMNewNeighbour"

logging.basicConfig()

#--- CONSTANTS

FREE_EXPLORATION = 0
PUSHTASK = 1
PUSHTASKSIMULATION = 2
MOVE_TO_TARGET = 3
PUSHTASKSIMULATION2 = 4
MODE = PUSHTASKSIMULATION
#MODE = FREE_EXPLORATION
#MODE = MOVE_TO_TARGET


NUM_TRAIN_RUNS = 3
NUM_TEST_RUNS = 20

class GazeboInterface():
    """
        Class that handles the interaction with gazebo.
    """
    
    def __init__(self):
         
        self.active = True
        self.lastState = None
        if GATE:
            self.worldModel = model.ModelGate()
        else:
            self.worldModel = model.ModelInteraction()
        self.lastAction = np.zeros(2)
        self.lastPrediction = None
        self.ignore = True
        self.target = None
        self.startup = True
        
        self.trainRun = 0
        self.testRun = 0
        self.runStarted = False
        
        
        self.stepCounter = 0
        self.times = 0
     
        self.direction = np.array([0.0,0.5,0.0])
        self.finalPrediction = None
        
        self.startPositions = []
        self.numSteps = 0       
        self.runNumber = 0
        
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
                    
        self.physicsControlPublisher = yield From(
                self.manager.advertise('/gazebo/default/physics',
                    'gazebo.msgs.Physics'))
                    
        self.sensorControlPublisher = yield From(
                self.manager.advertise('/gazebo/default/sensor',
                    'gazebo.msgs.Sensor'))
                    
        
        
        
                          
        while self.active:
            yield From(trollius.sleep(1))
                          
                          
    def sendCommand(self, action):
        """
        Function to send a gripperCommand to gazebo
        
        Parameters
        ----------
        action: model.Action
            The action that should be send to gazebo
        """
        msg = pygazebo.msg.gripperCommand_pb2.GripperCommand()
        msg.cmd = GAZEBOCMDS["MOVE"]
        
        
            
#        action["mvDir"] *= 0.5
        msg.direction.x = action[0]
        msg.direction.y = action[1]
        msg.direction.z = 0.0
#        msg.direction.z = action.direction[2] # ignore z for now
#        print msg
        self.cmdPublisher.publish(msg)
        
    def sendStopCommand(self):
        """
        Function to send a stop command to gazebo to stop the gripper from moving.
        """
        print "sending stop command"
        msg = pygazebo.msg.gripperCommand_pb2.GripperCommand()
        msg.cmd = GAZEBOCMDS["MOVE"]
        msg.direction.x = 0.0
        msg.direction.y = 0.0
        msg.direction.z = 0.0
        self.cmdPublisher.publish(msg)
        
    
        
    def sendPrediction(self):
        """
        Function to send the last prediction to gazebo. This will move the shadow model
        positions.
        """
        names = {15:"blockA", 8: "gripper"}
        msg = pygazebo.msg.modelState_v_pb2.ModelState_V()
        if GATE:
            msg.models.extend([self.getModelState("gripperShadow", self.lastPrediction.actuator.vec[0:2], self.lastPrediction.actuator.vec[2])])
        for objectState in self.lastPrediction.objectStates.values():
            tmp = self.getModelState(names[objectState.id]+"Shadow", objectState.vec[0:2], objectState.vec[2])
            msg.models.extend([tmp])
        self.posePublisher.publish(msg)
 

    def getModelState(self, name, pos, euler):
        """
        Function to create a ModelState object from a name and the desired position and orientation.
        
        Parameters
        ----------
        name : String
            Name of the model
        pos : np.array() Size 2
            The position of the model
        euler: Float
            The orientation around the z-axis in world coordinates

        Returns
        -------
        modelState_pb2.ModelState
            Protobuf object for a model state with fixed id to 99
        """
        
        assert not np.any(np.isnan(pos)), "Pos has nan in it: {}".format(pos)
        assert not np.any(np.isnan(euler)), "euler has nan in it: {}".format(euler)

        msg = modelState_pb2.ModelState()
        msg.name = name
        msg.id = 99
        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1] 
        msg.pose.position.z =  0.03 if name == "gripperShadow" else 0.05

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
        self.finalPrediction = self.lastPrediction
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
        print "called at: ", datetime.datetime.now()
        if self.startup:
            self.resetWorld()
            self.startup= False
            
        else:
            worldState = worldState_pb2.WorldState.FromString(data)
            print "parsing new WorldState"
            newWS = model.WorldState()
            newWS.parse(worldState)
            
            if self.lastPrediction != None and RECORD_SIMULATION:
                self.recordData(newWS)
            
            if MODE == FREE_EXPLORATION:
                self.randomExploration(newWS)
            elif MODE == PUSHTASK:
                self.pushTask(newWS)
            elif MODE == PUSHTASKSIMULATION:
                self.pushTaskSimulation(newWS)
            elif MODE == PUSHTASKSIMULATION2:
                self.pushTaskSimulation2(newWS)
            elif MODE == MOVE_TO_TARGET:
#                self.setTarget()
                self.moveToTarget(newWS)
            else:
                raise AttributeError("Unknown MODE: ", MODE)

    def changeUpdateRate(self, rate):
        msg = sensor_pb2.Sensor()
        msg.name = "gripperContact"
        msg.type = "contact"
        msg.parent = "gripper::finger"
        msg.parent_id = 9
        msg.update_rate = rate
        self.sensorControlPublisher.publish(msg)

    def changeRealTimeRate(self, rate):
        msg = physics_pb2.Physics()
        msg.type = physics_pb2.Physics.ODE
        msg.real_time_update_rate = rate
        self.physicsControlPublisher.publish(msg)

    def recordData(self, newWorldState):
        #Create the filename: Model_NrTrainRuns_
#        if GATE:
#            fileName = "gateModel_" + str(trainRuns[self.runNumber]) + 
        if GATE:
            s = "" #TODO Fold, run and timestep number
            for o in newWorldState.objectStates.values():
                keypoints = o.getKeyPoints()
                s += "{}; {}; {}; {}; {}; {}; {}; {}".format(o.id, 
                                    keypoints[0][0], keypoints[0][1], 
                                    keypoints[1][0], keypoints[1][1], 
                                    keypoints[2][0], keypoints[2][1], o.vec[2])
                s += ";"
            s += "{}; {};".format(newWorldState.actuator.vec[0], newWorldState.actuator.vec[1])
            
            for o in self.lastPrediction.objectStates.values():
                keypoints = o.getKeyPoints()
                s += "{}; {}; {}; {}; {}; {}; {}; {}".format(o.id, 
                                    keypoints[0][0], keypoints[0][1], 
                                    keypoints[1][0], keypoints[1][1], 
                                    keypoints[2][0], keypoints[2][1], o.vec[2])
                s += ";"
            s += "{}; {}\n".format(self.lastPrediction..actuator.vec[0], self.lastPrediction..actuator.vec[1])
        else:
            pass
        with open(fileName, "a") as f:
            f.write(s)
        pass
    
    def writeConfig(self):
        with open(fileName, "w") as f:
            f.write("Config for testrun recorded in {}\n".format(fileName))
            f.write(config.toString())

    def runEnded(self, worldState):
        """
        Function to determine if a run has ended. In this case a run has ended, when the
        gripper did not move more than a threshold compared to the last State.
        
        Parameters
        ----------
        worldState: model.WorldState
            The current world state.
        """
        gripperOs = worldState.actuator
        tPos = gripperOs.vec[0:2]
        
        if np.linalg.norm(tPos) > 1.0 or self.stepCounter > 150:
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
#        if self.trainRun == 0:
#            posX = -0.25
#        elif self.trainRun == 1:
#            posX = 0.25
#        elif self.trainRun == 2:
#            posX = 0
        if self.trainRun == trainRuns[self.runNumber]:
            self.startPositions.append(posX)    
            
        self.sendPose("gripper", np.array([posX,0.0,0.03]), 0.0)
        self.stepCounter = 0
        self.ignore = True
        
    def startRun2(self, randomRange=0.5):
        self.runStarted = True
         #Set up Starting position
        posy = ((np.random.rand()-0.5)*randomRange) #* 0.5
        self.sendPose("gripper", np.array([0.0,posy,0.03]), 0.0)
        self.sendPose("blockA", np.array([-0.5, 0.0, 0.05]) , 1.0)
        self.stepCounter = 0
        self.ignore = True
        
    def startRun3(self):
        self.runStarted = True
        self.sendPose("gripper", np.array([-0.5, 0.5,0.03]), -1.0)
        self.stepCounter = 0
        self.ignore = True

    def pushTask(self, worldState):
        raise NotImplementedError
        
    def pushTaskSimulation2(self, worldState):
        """
            Method to perform pushTask simulation. This means that predictions are made
            based on the previous prediction the model made.
            The difference to the standard pushTaskSimulation is that the model is still
            being updated after it made the prediction with what actually happened.
        """
        if self.runStarted:
            if self.runEnded(worldState):
                self.resetWorld()
                self.runStarted = False
        else:
            self.startRun(0.7)
            self.direction = np.array([0.0,0.5])
            return
        if self.testRun < NUM_TEST_RUNS:
            if self.runStarted:
                if self.lastPrediction == None:
                    self.lastPrediction = self.worldModel.predict(worldState, self.direction)
                    self.worldModel.resetObjects(worldState)
                    self.lastAction = np.zeros(2)
                else:
                    self.lastPrediction = self.worldModel.predict(self.lastPrediction, self.direction)
                self.worldModel.update(worldState, self.lastAction)
                self.lastAction = self.direction
                self.sendPrediction()
                self.sendCommand(self.lastAction)
            else:
                self.testRun += 1
        else:
            self.pauseWorld()
            
    def pushTaskSimulation(self, worldState):
        self.stepCounter += 1
#        print "num cases: " + str(len(self.worldModel.cases))
#        print "num abstract cases: " + str(len(self.worldModel.abstractCases))
        
        if self.runStarted:
            if self.runEnded(worldState):
                self.resetWorld()
                self.runStarted = False
        else:
            if self.testRun > 0:
                self.startRun(0.7)
                self.direction = np.array([0.0,0.5])
            else:
                self.startRun(0.7)
                self.direction = np.array([0.0,0.5])
            return
            
        if self.trainRun < trainRuns[self.runNumber]: #NUM_TRAIN_RUNS:
            print "Train run #: ", self.trainRun
            if self.runStarted:
                self.updateModel(worldState, self.direction)
            else:
                self.trainRun += 1
                if self.trainRun == trainRuns[self.runNumber]: # NUM_TRAIN_RUNS:
                    self.pauseWorld()
                    np.random.seed(4321) # Set new seed so that all test runs start identically, independent of the number of training runs
                    self.worldModel.training = False
                    
        elif self.testRun < NUM_TEST_RUNS:
            print "Test run #: ", self.testRun
            if self.runStarted:
                self.lastAction = self.direction
                if self.lastPrediction != None:
                    predictedWorldState = self.lastPrediction
#                    self.worldModel.actuator.vec = self.lastPrediction.actuator.vec
#                    curDifBlock, curDifActuator = self.compare(worldState, self.lastPrediction)
#                    self.accDifBlock += curDifBlock
#                    self.accDifActuator += curDifActuator
                    self.numSteps +=1
                else:
                    print "lastPrediction None"
                    predictedWorldState = worldState
                    self.worldModel.resetObjects(worldState)
                self.worldModel.update(predictedWorldState, self.lastAction)
                self.lastPrediction = self.worldModel.predict(predictedWorldState, self.lastAction)
                self.sendPrediction()
                self.sendCommand(self.lastAction)
            else:
                self.testRun += 1
                if RECORD_SIMULATION:
                    differenceBlock, differenceActuator = self.compare(worldState, self.finalPrediction)
                    with open("../../data/" + SIMULATION_FILENAME.format(trainRuns[self.runNumber]) + ".txt", "a") as f:
                        f.write("{}; ".format(differenceBlock))
                        f.write("{}; ".format(self.accDifBlock))
                        f.write("{}; ".format(self.accDifBlock/self.numSteps))
                        f.write("{}; ".format(differenceActuator))
                        f.write("{}; ".format(self.accDifActuator))
                        f.write("{} ".format(self.accDifActuator/self.numSteps))
                        f.write("\n")
                    self.accDifBlock = 0.0
                    self.accDifActuator = 0.0
                    self.numSteps = 0
        else:
#            self.pauseWorld()
            if RECORD_SIMULATION:
                with open("../../data/"+ SIMULATION_FILENAME.format(trainRuns[self.runNumber]) + "startPos.txt", "w") as f:
                    f.write("; ".join(["{:.4f}".format(x) for x in self.startPositions]))
                if self.runNumber < len(trainRuns)-1:
                    self.runNumber += 1
                    self.testRun = 0
                    self.trainRun = 0
                    np.random.seed(1234)
                    self.worldModel = model.ModelAction()
                    self.resetWorld()
                    self.finalPrediction = None
                    self.startPositions = []
                else:
                    self.pauseWorld()
            else:
                self.pauseWorld()
            
    def compare(self, worldState, prediction):
        blockOSReal = worldState.objectStates[15]
        blockOSPrediction = prediction.objectStates[15]
        gripperOSReal = worldState.actuator
        gripperOSPrediction = prediction.actuator
        return blockOSReal.compare(blockOSPrediction), gripperOSReal.compare(gripperOSPrediction)
        
            
    def updateModel(self, worldState, direction=np.array([0.0,0.5])):
        """
        Function to perform the world update and get the next prediction.
        Currently action NOTHING is performed in here.
        
        Paramters
        ---------
        worldState: mode.WorldState
            The current world state
        """
        if self.lastState != None and worldState != None:
            self.worldModel.update(worldState, self.lastAction)
        else:
            print "reset objects"
            self.worldModel.resetObjects(worldState)
        
        self.lastState = worldState
        self.lastAction = direction
        
        self.lastPrediction = self.worldModel.predict(worldState, self.lastAction)
        
        self.sendPrediction()
        self.sendCommand(self.lastAction)


    def randomExploration(self, worldState):
        raise NotImplementedError
        
    def startRunTarget(self, run):
        startPositions = {0:np.array([0.24,0.0,0.03]), 1: np.array([0.3, 0.25,0.03]),
                          2: np.array([0.24, 0.5, 0.03]), 3: np.array([0.0,0.5,0.03]), 
                          4: np.array([-0.24,0.5,0.03]), 5: np.array([-0.3,0.25,0.03]), 
                          6: np.array([-0.24,0.0,0.03]), 7: np.array([0.0,0.0,0.03]), 8: np.array([0.0,0.0,0.03])}
        directions = {0: np.array([0.0,0.5]), 1: np.array([-0.5, 0.0]),
                      2: np.array([0.0, -0.5]), 3: np.array([0.0,-0.5]), 4: np.array([0.0,-0.5]),
                      5: np.array([0.5,0.0]), 6: np.array([0.0,0.5]), 7:np.array([0.0,0.5]), 8:np.array([0.0,0.5]) }
        self.runStarted = True
        self.sendPose("gripper", startPositions[run], 0.0)
        self.stepCounter = 0
        self.direction = directions[run]
        
    def moveToTarget(self, worldState):
        self.stepCounter += 1
        if self.runStarted:
            if self.trainRun < NUM_TRAIN_RUNS and self.runEnded(worldState):
                self.resetWorld()
                self.runStarted = False
        else:
            self.startRunTarget(self.trainRun)
            return
            
        if self.trainRun < NUM_TRAIN_RUNS:
            print "trainrun : ", self.trainRun
            if self.runStarted:
                self.updateModel(worldState, self.direction)
            else:
                self.trainRun += 1
                if self.trainRun == NUM_TRAIN_RUNS:
#                    self.setTarget()
                    self.pauseWorld()
        elif self.testRun < NUM_TEST_RUNS:
            if self.worldModel.target ==  None:
                self.setRandomTarget()
            if self.lastAction != None and worldState != None:
#                print "updating with last action: ", self.lastAction
                self.worldModel.update(worldState, self.lastAction)
            else:
                self.worldModel.resetObjects(worldState)
            self.lastAction = self.worldModel.getAction()
#            print "sending action: ", self.lastAction
            self.sendCommand(self.lastAction)
#            self.lastPrediction = self.worldModel.predict(worldState, self.lastAction)
#            self.sendPrediction()
            if self.worldModel.target != None:
                self.sendPose("blockAShadow", self.worldModel.target.vec[0:2], self.worldModel.target.vec[2])
            
        
    def setRandomTarget(self):
        target = model.Object()
        target.id = 15
        target.vec = np.zeros(4)
        target.vec[0:2] = (np.random.rand(2)-0.5)*2.0
        target.vec[2] = (np.random.rand()-0.5)*2*np.pi
        self.worldModel.setTarget(target)

    def setTarget(self):
        target = model.Object()
        target.id = 15
        if model.USE_DYNS:
            target.vec = np.array([-0.5, 0.4, 0.05, 1.8 ,0.0,0.0,0.0])
        else:
            target.vec = np.array([0.75, -0.4, 0.05, -1.0])#, 0.0,0.0,0.0, 0.0])
        self.worldModel.setTarget(target)
    
    def stop(self):
        """
        Function to stop the loop.
        """
        self.active = False

    
if __name__ == "__main__":
    loop = trollius.get_event_loop()
    gi = GazeboInterface()
    loop.run_until_complete(gi.loop())
    