#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:32:53 2015
Python interface to communicate with gazebo.
Adapted to better suit the 2d case.
@author: jpoeppel
"""


import os
import trollius
from trollius import From

import pygazebo
from pygazebo.msg import modelState_pb2
from pygazebo.msg import modelState_v_pb2
from pygazebo.msg import actuatorCommand_pb2
from pygazebo.msg import worldState_pb2
from pygazebo.msg import sensor_pb2
from pygazebo.msg import physics_pb2
from pygazebo.msg import world_control_pb2


import numpy as np
np.set_printoptions(precision=6,suppress=True)

from common import GAZEBOCMDS
import common


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

#Use gating or interaction model
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
CONFIGURATION = 0#configuration.FIXFIRSTTHREETRAININGRUNS #| configuration.HARDCODEDACT
config.switchToConfig(CONFIGURATION)

#If training on perfect test positions, testpositions are shuffeled
tmpL = range(len(config.testPositions))
np.random.shuffle(tmpL)
mapping = {i: tmpL[i] for i in range(len(tmpL))}

FILEEXTENSION = ""

trainRuns = [3]
NUMBER_FOLDS = 1
RECORD_SIMULATION = False

TWO_OBJECTS = False
BLOCKNAMES = {15: "blockA", 27:"blockB"}

logging.basicConfig()

#--- CONSTANTS

FREE_EXPLORATION = 0
PUSHTASK = 1
PUSHTASKSIMULATION = 2
PUSHTASKSIMULATION2 = 3
MOVE_TO_TARGET = 4
DEBUG = 5

#MODE = PUSHTASKSIMULATION
#MODE = FREE_EXPLORATION
MODE = MOVE_TO_TARGET
#MODE = DEBUG

#Num training runs for move to target
NUM_TRAIN_RUNS = 8
if MODE == PUSHTASKSIMULATION:
    NUM_TEST_RUNS = len(config.testPositions)
elif MODE == PUSHTASKSIMULATION2:
    NUM_TEST_RUNS = len(config.testPositions)
elif MODE == MOVE_TO_TARGET:
    NUM_TEST_RUNS = len(config.targets)
    trainRuns = range(NUM_TEST_RUNS)

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
        self.lastAction = None
        self.lastPrediction = None
        self.ignore = False
        self.target = None
        self.startup = True
        self.startedRun = False
        
        self.trainRun = 0
        self.testRun = 0
        self.runStarted = False
        
        self.foldNumber = 0
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
                self.manager.advertise('/gazebo/default/actuatorMsg',
                    'gazeboPlugins.msgs.ActuatorCommand'))   
                    
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
            Function to send an actuatorCommand to gazebo
            
            Parameters
            ----------
            action: model.Action
                The action that should be send to gazebo
        """
        msg = actuatorCommand_pb2.ActuatorCommand()
        msg.cmd = GAZEBOCMDS["MOVE"]
        msg.direction.x = action[0]
        msg.direction.y = action[1]
        msg.direction.z = 0.0
        self.cmdPublisher.publish(msg)
        
    def sendStopCommand(self):
        """
            Function to send a stop command to gazebo to stop the actuator from moving.
        """
        msg = actuatorCommand_pb2.ActuatorCommand()
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
        names = {27: "blockB", 15:"blockA", 8: "actuator"}
        msg = pygazebo.msg.modelState_v_pb2.ModelState_V()
        if GATE:
            msg.models.extend([self.getModelState("actuatorShadow", 
                                                self.lastPrediction.actuator.vec[0:2], 
                                                self.lastPrediction.actuator.vec[2])])
        for objectState in self.lastPrediction.objectStates.values():
            tmp = self.getModelState(names[objectState.id]+"Shadow", 
                                     objectState.vec[0:2], objectState.vec[2])
            msg.models.extend([tmp])
        self.posePublisher.publish(msg)
 

    def getModelState(self, name, pos, ori):
        """
            Function to create a ModelState object from a name and the desired 
            position and orientation.
            
            Parameters
            ----------
            name : String
                Name of the model
            pos : np.array() Size 2
                The position of the model
            ori: Float
                The orientation around the z-axis in world coordinates
    
            Returns
            -------
            modelState_pb2.ModelState
                Protobuf object for a model state with fixed id to 99
        """
        assert not np.any(np.isnan(pos)), "Pos has nan in it: {}".format(pos)
        assert not np.any(np.isnan(ori)), "ori has nan in it: {}".format(ori)

        msg = modelState_pb2.ModelState()
        msg.name = name
        msg.id = 99
        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1] 
        msg.pose.position.z =  0.03 if name == "actuatorShadow" else 0.05

        ori= common.eulerToQuat(ori)
        msg.pose.orientation.x = ori[0]
        msg.pose.orientation.y = ori[1]
        msg.pose.orientation.z = ori[2]
        msg.pose.orientation.w = ori[3]
        
        return msg    
    
        
    def sendPose(self, name, pos, ori):
        """
            Function to send the pose of an object to the simulation.
            
            Parameters
            ----------
            name : String
                Name of the model
            pos : np.array() Size 2
                The position of the model
            ori: Float
                The orientation around the z-axis in world coordinates
    
            Returns
            -------
            modelState_pb2.ModelState
                Protobuf object for a model state with fixed id to 99
        """
        msg = modelState_v_pb2.ModelState_V()
        msg.models.extend([self.getModelState(name, pos, ori)])
        self.posePublisher.publish(msg)
        
    def resetWorld(self):
        """
            Function to reset the world and all helper variables in this interface.
            Afterwards all objects should have their original position and orientation in the 
            simulation.
        """
        self.sendStopCommand()
        msg = world_control_pb2.WorldControl()
        msg.reset.all = True
        self.worldControlPublisher.publish(msg)
        self.lastAction = None
        self.lastPrediction = None
        self.lastState = None
        self.stepCounter = 0
        self.ignore= True
        
    def pauseWorld(self):
        """
            Sends a pause command to the simulation.
        """
        msg = world_control_pb2.WorldControl()
        msg.pause = True
        self.worldControlPublisher.publish(msg)
        
        
    def modelsCallback(self, data):
        """
            Callback function that registers gazebo model information.
            
            
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
                self.moveToTarget(newWS)
            elif MODE == DEBUG:
                self.sendPose("blockAShadow", config.targets[1][:2], config.targets[1][2])
            else:
                raise AttributeError("Unknown MODE: ", MODE)
        end = time.time()
#        print "Execution took: ", end-start
        if end-start > 2*1.0/config.frequency:
            self.numTooSlow += 1
            
    def checkPositions(self, newWS):
        """
            Function to check the position of all objects at the start of a run.
            If positions do not match, another sendPose command is used and the check
            is performed again at the next timestep.
            
            Paramters
            ---------
            newWS : model.WorldState
                Current worldstate at the beginning of a run, containing all object information
            
            Returns
            -------
                bool:
                    True if all objects have the desired configuration
                    False otherwise
        """
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
            self.sendPose("actuator", np.concatenate((self.lastStartConfig[8][:2],[0.03])), self.lastStartConfig[8][2])
            self.startedRun = True
            return False
        for block in blocks:
            if np.linalg.norm(block.vec-self.lastStartConfig[block.id]) > 0.01:
                self.resetErrors += 1
                self.sendPose(BLOCKNAMES[block.id], np.concatenate((self.lastStartConfig[block.id][:2],[0.05])), self.lastStartConfig[block.id][2])
                self.startedRun = True
                return False
        return True
        

    def changeUpdateRate(self, rate):
        """
            Allows to change the update rate of the contact sensor in the simulation. Basically
            determines the frequency with which this interface is called.
            
            Parameters
            ----------
            rate : int
                Desired update frequency
        """
        msg = sensor_pb2.Sensor()
        msg.name = "actuatorContact"
        msg.type = "contact"
        msg.parent = "actuator::finger"
        msg.parent_id = 9
        msg.update_rate = rate
        self.sensorControlPublisher.publish(msg)

    def changeRealTimeRate(self, rate):
        """
            Allows to change the real time update rate of the simulation. Determines the number of
            simulation steps performed in one second.
            
            Parameters
            ----------
            rate : int
                Desired update rate
        """
        msg = physics_pb2.Physics()
        msg.type = physics_pb2.Physics.ODE
        msg.real_time_update_rate = rate
        self.physicsControlPublisher.publish(msg)

    def recordData(self, newWorldState):
        """
            Helper function that records the data of the last prediction and the current 
            worldstate in order to evaluate it later.
            
            Paramters
            --------
            newWorldState : model.WorldState
                The current worldstate received from the simulation.
        """

        #Fold, isTraining, run and timestep number
        if MODE == PUSHTASKSIMULATION:
            if self.trainRun == trainRuns[self.runNumber]:
                training = 0
                runs = self.testRun
            else:
                training = 1
                runs = self.trainRun
        elif MODE == MOVE_TO_TARGET:
            if self.trainRun == NUM_TRAIN_RUNS:
                training = 0
                runs = self.runNumber
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
#                s += "0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0\n"
                s += "0.0;"*8*len(newWorldState.objectStates) + "0.0;0.0\n"
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
                #In the interaction model, the actuator is part of the object states, therefore
                #reduce by 1
                s += "0.0;"*8*(len(newWorldState.objectStates)-1) + "0.0;0.0\n"

        self.runString += s
       
    def writeData(self):
        """
            Writes the recorded data in self.runString to the text file specified by
            self.filename in the folder ../../evalData if RECORD_SIMULATION = True
        """
        if not RECORD_SIMULATION:
            return

        if not os.path.isfile("../../evalData/" + self.fileName + ".txt"):
            self.runString = "#FoldNumber; isTraining; RunNumber; StepNumber; BlockId; px; py; kx1; \
                        ky1; kx2; ky2; ori; actx; acty; PredBlockId; Pred px; py; ...\n" + self.runString
            
        with open("../../evalData/" + self.fileName + ".txt", "a") as f:
            f.write(self.runString)
            
        self.runString = ""
    
    def writeConfig(self):
        """
            Writes the configuration string to the file, specified by self.fileName + "_config"
            in the folder ../../evalData.
        """
        with open("../../evalData/" + self.fileName + "_config.txt", "w") as f:
            f.write("Configuration for experiment recorded in {}\n".format(self.fileName))
            f.write(config.toString(GATE))
            
    def writeITMInformation(self, info):
        """
        Writes the ITM information info to the file, specified by self.fileName + "_ITMInformation"
            in the folder ../../evalData.
            
        Parameters
        ----------
        info : String
            String containing information about the used aitm.
        """
        with open("../../evalData/" + self.fileName + "_ITMInformation.txt", "a") as f:
            f.write(info)

    def runEnded(self, worldState):
        """
        Function to determine if a run has ended. In this case a run has ended, when the
        actuator did not move more than a threshold compared to the last State.
        
        Parameters
        ----------
        worldState: model.WorldState
            The current world state.
        
        Returns
        -------
        bool:
            True if the actuator has moved at least 1m or the stepCounter passed 150
            False otherwise
        """
        actuatorOs = worldState.actuator
        tPos = actuatorOs.vec[0:2]
        
        if np.linalg.norm(tPos) > 1.0 or self.stepCounter > 150:
            return True
        return False
       
    def runEndedMoveToTarget(self):
        """
            Function to determine if a test run during moveToTarget has ended.
            Depends only on the maximum number of steps since target reached is
            evaluated by in the moveToTarget function.
            
            Returns
            -------
            bool
                True if the number of steps exceed the limit
        """
        if self.stepCounter > 3000:
            return True
        return False
        
    def startRun(self, randomRange=0.5):
        """
        Function to start a new Run. Determines the starting position of the actuator and sets up
        local variables.
        
        Paramters
        --------
        randomRange : float, optional
            Random positions will be drawn from [-randomRange/2, randomRange/2]
            Default 0.5

        """
        
        self.runStarted = True
         #Set up Starting position
        posX = ((np.random.rand()-0.5)*randomRange) #* 0.5
        if config.fixedFirstThreeTrains:
            if self.trainRun == 0:
#                posX = -0.18
                posX = -0.14
            elif self.trainRun == 1:
#                posX = 0.18
                posX = 0.14
            elif self.trainRun == 2:
                posX = 0
            if TWO_OBJECTS:
                if self.trainRun == 3:
                    posX = -0.73
                elif self.trainRun == 4:
                    posX = -0.47
                elif self.trainRun == 5:
                    posX = -0.6
                elif self.trainRun == 6:
                    posX = -0.4
                                
        if config.perfectTrainRuns:
            posX = config.testPositions[mapping[self.trainRun]]
                
        self.lastStartConfig = {8:np.array([posX,0.0,0.0]), 
                                15:np.array([0.0,0.25,0.0]), 
                                27:np.array([-0.6,0.35,0.0])}
            
        self.sendPose("actuator", np.array([posX,0.0,0.03]), 0.0)
        self.stepCounter = 0
        self.ignore = True
        self.startedRun = True
        
    def startTestRun(self):
        """
        Function to start a new test run. Determines the starting position of the actuator and sets up
        local variables.
        Starting testposition is chosen from the config file.
        """
        self.runStarted = True
        posX = config.testPositions[self.testRun]
        
        self.lastStartConfig = {8:np.array([posX,0.0,0.0]), 
                                15:np.array([0.0,0.25,0.0]), 
                                27:np.array([-0.6,0.35,0.0])}
        
        self.sendPose("actuator", np.array([posX, 0.0,0.03]), 0.0)
        self.stepCounter = 0
        self.ignore = True
        self.startedRun = True
    
        
    def pushTaskSimulation2(self, worldState):
        """
            Method to perform pushTask simulation. This means that predictions are made
            based on the previous prediction the model made.
            The difference to the standard pushTaskSimulation is that the model is still
            being updated after it made the prediction with what actually happened.
            No training is performed.
            
            Paramters
            ---------
            worldState : model.WorldState
                The current worldstate containing the current object information
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
                if self.testRun == NUM_TEST_RUNS:
                    self.done = True
                    self.active = False
        else:
            self.pauseWorld()
            
    def pushTaskSimulation(self, worldState):
        """
            Method to perform pushTask simulation. This means that predictions are made
            based on the previous prediction the model made. Model is not updated after
            training.
            
            Paramters
            ---------
            worldState : model.WorldState
                The current worldstate containing the current object information
        """
        self.stepCounter += 1
#        print "step counter: ", self.stepCounter
        
        if self.runStarted:
            if self.runEnded(worldState):
#                self.pauseWorld()
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
                    
        elif self.testRun < NUM_TEST_RUNS:
            print "Test run #: ", self.testRun
            if self.runStarted:
                self.lastAction = self.direction
                if self.lastPrediction != None:
                    predictedWorldState = self.lastPrediction
                else:
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

                
    def resetExperiment(self):
        """
            Function to reset the experiment to start with the next run.
            The current model is replaced by a new, untrained one and the new filename
            is setup.
        """
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
            self.worldModel.update(worldState, self.lastAction)
        else:
            self.worldModel.resetObjects(worldState)
            
        self.lastAction = direction
        self.lastPrediction = self.worldModel.predict(worldState, self.lastAction)
        self.sendPrediction()
        self.sendCommand(self.lastAction)

    def startRunTarget(self, run):
        """
        Function to start a new training run for the moveToTarget task. 
        Determines the starting position and direction of the actuator and sets up
        local variables.
        
        Paramters
        --------
        run : int
            Number of the current training run.

        """
        
#        runmapping = {0:7, 1:7}
#        run = runmapping[run]


        startPositions = {0:np.array([0.24,0.0,0.0]), 1: np.array([0.3, 0.25,0.0]),
                          2: np.array([0.24, 0.5, 0.0]), 3: np.array([0.0,0.5,0.0]), 
                          4: np.array([-0.24,0.5,0.0]), 5: np.array([-0.3,0.25,0.0]), 
                          6: np.array([-0.24,0.0,0.0]), 7: np.array([0.0,0.0,0.0]), 
                          8: np.array([0.0,0.0,0.0])}
                          
        directions = {0: np.array([0.0,0.5]),   1: np.array([-0.5, 0.0]),
                      2: np.array([0.0, -0.5]), 3: np.array([0.0,-0.5]), 
                      4: np.array([0.0,-0.5]),  5: np.array([0.5,0.0]), 
                      6: np.array([0.0,0.5]),   7:np.array([0.0,0.5]), 
                      8:np.array([0.0,0.5]) }
                      
        self.lastStartConfig = {8:startPositions[run], 
                                15:np.array([0.0,0.25,0.0]), 
                                27:np.array([-0.3,0.35,0.0])}
        self.startedRun = True
        self.runStarted = True
        self.sendPose("actuator", startPositions[run], 0.0)
        self.stepCounter = 0
        self.direction = directions[run]
        
    def moveToTarget(self, worldState):
        """
            Actual MoveToTarget task handler. First makes sure the model is trained on the provided
            training positions. Afterwards sets the desired target from the config file.
            If a target is reached or not is determined by the model.
            
            Parameters
            ----------
            worldState : model.WorldState
                Current worldstate provided by the simulation.
        """
        #Setup run and check for runEnded
        self.stepCounter += 1
        if self.runStarted:
            #During training the normal runEnded can be used
            if self.trainRun < NUM_TRAIN_RUNS and self.runEnded(worldState):
                self.resetWorld()
                self.writeData()
                self.runStarted = False
        else:
            self.startRunTarget(self.trainRun)
            return
            
        #Update model during training
        if self.trainRun < NUM_TRAIN_RUNS:
            if self.runStarted:
                self.updateModel(worldState, self.direction)
            else:
                self.trainRun += 1
                if self.trainRun == NUM_TRAIN_RUNS:
#                    self.setTarget()
                    if not RECORD_SIMULATION:
                        self.pauseWorld()
                    else:
                        self.writeITMInformation("####ITM Information for fold: {} after training\n".format(self.foldNumber) 
                                            + self.worldModel.getITMInformation())
                    #Setup first target after training
                    curTarget = model.Object()
                    curTarget.id = 15
                    curTarget.vec = config.targets[self.runNumber]
                    self.worldModel.setTarget(curTarget)
                    
        elif self.testRun < NUM_TEST_RUNS:
            #Update model with current information
            if self.lastAction != None and worldState != None:
                self.worldModel.update(worldState, self.lastAction)
            else:
                self.worldModel.resetObjects(worldState)
                
            #Query for action to use
            self.lastAction = self.worldModel.getAction()
            self.sendCommand(self.lastAction)

            if self.worldModel.target != None and not self.runEndedMoveToTarget():
                if GATE:
                    target = self.worldModel.target
                else:
                    target, o2 = model.Object.fromInteractionState(self.worldModel.target)
                #Project target in gui
                self.sendPose("blockAShadow", target.vec[0:2], target.vec[2])
            else:
                #Setup next target
                self.runStarted = False
                if not RECORD_SIMULATION:
                    self.pauseWorld()
                else:
                    self.writeITMInformation("####ITM Information for fold: {} after reaching target\n".format(self.foldNumber) 
                                        + self.worldModel.getITMInformation())
                    self.writeData()
                #Continue with next fold
                if self.foldNumber+1 < NUMBER_FOLDS:
                    self.foldNumber += 1
                    self.resetExperiment()
                    return
                else:
                    #Setup next target
                    if self.runNumber+1 < len(config.targets):
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
                
            
        
    def setRandomTarget(self):
        """
            Function allowing to set a random target in the valid area.
        """
        target = model.Object()
        target.id = 15
        target.vec = np.zeros(3)
        target.vec[0:2] = (np.random.rand(2)-0.5)*2.0
        target.vec[2] = (np.random.rand()-0.5)*2*np.pi
        self.worldModel.setTarget(target)

    def setTarget(self):
        """
            Function to set a fixed target in the valid area.
        """
        target = model.Object()
        target.id = 15
        if model.USE_DYNS:
            target.vec = np.array([-0.5, 0.4, 1.8 ,0.0,0.0])
        else:
            target.vec = np.array([0.75, -0.4, -1.0])
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
    print "Number of times too slow: ", gi.numTooSlow
