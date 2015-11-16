#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:32:53 2015
Adaption from interface5 in order to work with 2D coordinates
@author: jpoeppel
"""


import os
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

#import yappi

import numpy as np
np.set_printoptions(precision=6,suppress=True)

from common import GAZEBOCMDS
import common


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

GATE = True
#GATE = False

if GATE:
    import modelGate_2D_config as model
else:
    import modelInteractions_config as model

import time

import configuration
from configuration import config

#Select used configuration
CONFIGURATION = configuration.FIXFIRSTTHREETRAININGRUNS 
config.switchToConfig(CONFIGURATION)

#config.perfectTrainRuns = True

tmpL = range(len(config.testPositions))
np.random.shuffle(tmpL)
mapping = {i: tmpL[i] for i in range(len(tmpL))}

FILEEXTENSION = "_E11TestsOtherITM"

trainRuns = [3]
NUMBER_FOLDS = 2
RECORD_SIMULATION = True

logging.basicConfig()

#--- CONSTANTS

FREE_EXPLORATION = 0
PUSHTASK = 1
PUSHTASKSIMULATION = 2
PUSHTASKSIMULATION2 = 3
MOVE_TO_TARGET = 4

MODE = PUSHTASKSIMULATION
#MODE = FREE_EXPLORATION
#MODE = MOVE_TO_TARGET


NUM_TRAIN_RUNS = 8
NUM_TEST_RUNS = len(config.testPositions)

class GazeboInterface():
    """
        Class that handles the interaction with gazebo.
    """
    
    def __init__(self):
         
        self.active = True
        self.lastState = None
        self.done = False
        if GATE:
            self.worldModel = model.ModelGate()
        else:
            self.worldModel = model.ModelInteraction()
        self.lastAction = np.zeros(2)
        self.lastPrediction = None
        self.ignore = False
        self.target = None
        self.startup = True
        self.startedRun = False
        
        self.trainRun = 0
        self.testRun = 0
        self.runStarted = False
        
        self.foldNumber = 0
        self.numSteps = 0   
        self.runNumber = 0
        self.times = 0
     
        self.direction = np.array([0.0,0.5,0.0])
        self.lastStartConfig = None
        self.stepCounter = 0    
        self.configNummer = CONFIGURATION
        
        self.manager = None
        
        self.runString = ""
        
        if GATE:
            self.fileName = "gateModel_" + str(trainRuns[self.runNumber]) \
                    + "_TrainRuns_Mode" +  str(MODE) \
                    + "_Configuration_" + str(self.configNummer) + FILEEXTENSION
        else:
            self.fileName = "interactionModel_"+str(trainRuns[self.runNumber]) \
                + "_TrainRuns_Mode" + str(MODE) \
                + "_Configuration_" + str(self.configNummer) + FILEEXTENSION
        
        self.numTooSlow = 0
        self.resetErrors = 0

        
        if config.fixedTrainSeed:
            np.random.seed(config.trainSeed)
        
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
        msg = gripperCommand_pb2.GripperCommand()
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
        msg = gripperCommand_pb2.GripperCommand()
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
        names = {27: "blockB", 15:"blockA", 8: "gripper"}
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
        self.lastPrediction = None
        self.lastState = None
        self.numSteps = 0
        self.ignore= True
        
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
        if self.done:
            return
        start = time.time()
        if self.startup:
            self.changeUpdateRate(config.frequency)    
            self.resetWorld()
            self.startup= False
        elif self.ignore:
            self.ignore = False
        else:
            
            worldState = worldState_pb2.WorldState.FromString(data)
#            print "parsing new WorldState"
            newWS = model.WorldState()
            newWS.parse(worldState)
            
            if self.startedRun:
                self.startedRun = False
                if not self.checkPositions(newWS):
                    return
            
            if self.runStarted and RECORD_SIMULATION:
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
        end = time.time()
        print "Execution took: ", end-start
        if end-start > 2*1.0/config.frequency:
            self.numTooSlow += 1
#            raise TypeError("Execution took too long")
            
    def checkPositions(self, newWS):
        blocks = []
        if GATE:
            act = newWS.actuator
            blocks = newWS.objectStates.values()
        else:
            for o in newWS.objectStates.values():
                if o.id == 8:
                    act = o
                else:
                    blocks.append(o)
        if np.linalg.norm(act.vec-self.lastStartConfig[8]) > 0.01:
            self.resetErrors += 1
            self.sendPose("gripper", np.concatenate((self.lastStartConfig[8][:2],[0.03])), self.lastStartConfig[8][2])
            self.startedRun = True
            return False
        for block in blocks:
            if np.linalg.norm(block.vec-self.lastStartConfig[block.id]) > 0.01:
                self.resetErrors += 1
                self.sendPose("blockA", np.concatenate((self.lastStartConfig[block.id][:2],[0.05])), self.lastStartConfig[block.id][2])
                self.startedRun = True
                return False
        return True
        

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

        #Fold, isTraining, run and timestep number
        if self.trainRun == trainRuns[self.runNumber]:
            training = 0
            runs = self.testRun
        else:
            training = 1
            runs = self.trainRun
        s = "{};{};{};{};".format(self.foldNumber, training, runs, self.stepCounter) 
    
        if GATE:
            for o in newWorldState.objectStates.values():
                keypoints = o.getKeyPoints()
                s += "{}; {}; {}; {}; {}; {}; {}; {}".format(o.id, 
                                    keypoints[0][0], keypoints[0][1], 
                                    keypoints[1][0], keypoints[1][1], 
                                    keypoints[2][0], keypoints[2][1], o.vec[2])
                s += ";"
            s += "{}; {};".format(newWorldState.actuator.vec[0], newWorldState.actuator.vec[1])
            if self.lastPrediction != None:
                
                for o in self.lastPrediction.objectStates.values():
                    keypoints = o.getKeyPoints()
                    s += "{}; {}; {}; {}; {}; {}; {}; {}".format(o.id, 
                                        keypoints[0][0], keypoints[0][1], 
                                        keypoints[1][0], keypoints[1][1], 
                                        keypoints[2][0], keypoints[2][1], o.vec[2])
                    s += ";"
                s += "{}; {}\n".format(self.lastPrediction.actuator.vec[0], self.lastPrediction.actuator.vec[1])
            else:
                #No prediction values possible
                s += "0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0\n"
        else:
            for o in newWorldState.objectStates.values():
                bS = ""
                if o.id == 8:
                    actS = "{}; {};".format(o.vec[0], o.vec[1])
                else:
                    keypoints = o.getKeyPoints()
                    bS += "{}; {}; {}; {}; {}; {}; {}; {}".format(o.id, 
                                        keypoints[0][0], keypoints[0][1], 
                                        keypoints[1][0], keypoints[1][1], 
                                        keypoints[2][0], keypoints[2][1], o.vec[2])
                    bS += ";"
            s += bS + actS
            if self.lastPrediction != None:
                for o in self.lastPrediction.objectStates.values():
                    bS = ""
                    if o.id == 8:
                        actS = "{}; {}\n".format(o.vec[0], o.vec[1])
                    else:
                        keypoints = o.getKeyPoints()
                        bS += "{}; {}; {}; {}; {}; {}; {}; {}".format(o.id, 
                                            keypoints[0][0], keypoints[0][1], 
                                            keypoints[1][0], keypoints[1][1], 
                                            keypoints[2][0], keypoints[2][1], o.vec[2])
                        bS += ";"
                    
                s += bS + actS
            else:
                s += "0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0\n"

        self.runString += s
        return
       
    def writeData(self):
        if not RECORD_SIMULATION:
            return

        if not os.path.isfile("../../evalData/" + self.fileName + ".txt"):
            self.runString = "#FoldNumber; isTraining; RunNumber; StepNumber; BlockId; px; py; kx1; \
                        ky1; kx2; ky2; ori; actx; acty; PredBlockId; Pred px; py; ...\n" + self.runString
#            self.writeConfig()
            
        with open("../../evalData/" + self.fileName + ".txt", "a") as f:
            f.write(self.runString)
            
        self.runString = ""
    
    def writeConfig(self):
        with open("../../evalData/" + self.fileName + "_config.txt", "w") as f:
            f.write("Configuration for experiment recorded in {}\n".format(self.fileName))
            f.write(config.toString(GATE))
            
    def writeITMInformation(self, info):
        with open("../../evalData/" + self.fileName + "_ITMInformation.txt", "a") as f:
            f.write(info)

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
        if config.fixedFirstThreeTrains:
            if self.trainRun == 0:
                posX = -0.25
            elif self.trainRun == 1:
                posX = 0.25
            elif self.trainRun == 2:
                posX = 0
                                
        if config.perfectTrainRuns:
            posX = config.testPositions[mapping[self.trainRun]]
                
        self.lastStartConfig = {8:np.array([posX,0.0,0.0]), 15:np.array([0.0,0.25,0.0]), 27:np.array([-0.3,0.35,0.0])}
            
        self.sendPose("gripper", np.array([posX,0.0,0.03]), 0.0)
        self.stepCounter = 0
        self.ignore = True
        self.startedRun = True
        
    def startTestRun(self):
        self.runStarted = True
        posX = config.testPositions[self.testRun]
        
        self.lastStartConfig = {8:np.array([posX,0.0,0.0]), 15:np.array([0.0,0.25,0.0]), 27:np.array([-0.3,0.35,0.0])}
        
        self.sendPose("gripper", np.array([posX, 0.0,0.03]), 0.0)
        self.stepCounter = 0
        self.ignore = True
        self.startedRun = True
        
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
            self.startTestRun()
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
                self.writeData()
                self.runStarted = False
        else:
            if self.trainRun == trainRuns[self.runNumber]:
                self.startTestRun()
            else:
                self.startRun(config.startRunRange)
            self.direction = np.array([0.0,0.5])
            return
            
        if self.trainRun < trainRuns[self.runNumber]: #NUM_TRAIN_RUNS:
            print "Train run #: ", self.trainRun
            if self.runStarted:
                self.updateModel(worldState, self.direction)
            else:
                self.trainRun += 1
                if self.trainRun == trainRuns[self.runNumber]: # NUM_TRAIN_RUNS:
                    if not RECORD_SIMULATION:
                        self.pauseWorld()
                    else:
                        self.writeITMInformation("####ITM Information for fold: {}\n".format(self.foldNumber) 
                                            + self.worldModel.getITMInformation())
                    if config.fixedTestSeed:
                        # Set new seed so that all test runs start identically, independent of the number of training runs
                        np.random.seed(config.testSeed) 
                    self.worldModel.training = False
#                    fileName = ""
                    
                    
                    
        elif self.testRun < NUM_TEST_RUNS:
            print "Test run #: ", self.testRun
            if self.runStarted:
                self.lastAction = self.direction
                if self.lastPrediction != None:
                    predictedWorldState = self.lastPrediction

                    
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
                if self.testRun == NUM_TEST_RUNS:
                    #Continue with next fold
                    if self.foldNumber+1 < NUMBER_FOLDS:
                        self.foldNumber += 1
                        self.resetExperiment()
                        return
                    else:
                        if self.runNumber+1 < len(trainRuns):
                            self.runNumber += 1
                            self.foldNumber = 0
                            self.resetExperiment()
                            return
                        else:
                            config.numTooSlow = self.numTooSlow
                            config.resetErrors = self.resetErrors
                            self.writeConfig()
                            self.active = False
                            self.done = True
                            
        else:
#            import sys
#            sys.exit()
            pass
                
    def resetExperiment(self):
        self.trainRun = 0
        self.testRun = 0
        if GATE:
            self.worldModel = model.ModelGate()
            self.fileName = "gateModel_" + str(trainRuns[self.runNumber]) \
                        + "_TrainRuns_Mode" +  str(MODE) \
                        + "_Configuration_" + str(self.configNummer) + FILEEXTENSION
        else:
            self.worldModel = model.ModelInteraction()
            self.fileName = "interactionModel_"+str(trainRuns[self.runNumber]) \
                        + "_TrainRuns_Mode" + str(MODE) \
                        + "_Configuration_" + str(self.configNummer) + FILEEXTENSION
        if config.fixedTrainSeed:
            np.random.seed(config.trainSeed)
        else:
            np.random.seed()
        self.resetWorld()
            
    def updateModel(self, worldState, direction=np.array([0.0,0.5])):
        """
        Function to perform the world update and get the next prediction.
        Currently action NOTHING is performed in here.
        
        Paramters
        ---------
        worldState: mode.WorldState
            The current world state
        """
        if self.lastPrediction != None and worldState != None:
            try:
                self.worldModel.update(worldState, self.lastAction)
            except:
                self.pauseWorld()
#                raise NotImplementedError
        else:
            print "reset objects"
            self.worldModel.resetObjects(worldState)
        
#        self.lastState = worldState
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
                if GATE:
                    target = self.worldModel.target
                else:
                    target, o2 = model.Object.fromInteractionState(self.worldModel.target)
                self.sendPose("blockAShadow", target.vec[0:2], target.vec[2])
            
        
    def setRandomTarget(self):
        target = model.Object()
        target.id = 15
#        target.vec = np.array([0.0, 0.7, 0.0])
        target.vec = np.zeros(3)
        target.vec[0:2] = (np.random.rand(2)-0.5)*2.0
        target.vec[2] = (np.random.rand()-0.5)*2*np.pi
        self.worldModel.setTarget(target)

    def setTarget(self):
        target = model.Object()
        target.id = 15
        if model.USE_DYNS:
            target.vec = np.array([-0.5, 0.4, 1.8 ,0.0,0.0])
        else:
            target.vec = np.array([0.75, -0.4, -1.0])#, 0.0,0.0,0.0, 0.0])
        self.worldModel.setTarget(target)
    
    def stop(self):
        """
        Function to stop the loop.
        """
        self.active = False

    
if __name__ == "__main__":
#    yappi.start()
    loop = trollius.get_event_loop()
    gi = GazeboInterface()
    loop.run_until_complete(gi.loop())
    print "Number of times too slow: ", gi.numTooSlow
#    yappi.get_func_stats().print_all(columns={0:("name",200), 1:("ncall", 5), 2:("tsub", 8), 3:("ttot", 8), 4:("tavg",8)})
