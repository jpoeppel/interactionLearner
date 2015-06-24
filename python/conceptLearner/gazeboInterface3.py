#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:58:30 2015

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
from config import DIFFERENCES, SINGLE_INTSTATE, INTERACTION_STATES

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
if INTERACTION_STATES:
    import model4 as model
else:
    import modelActions as model
#import model6 as model

from sklearn.externals.six import StringIO
import pydot


logging.basicConfig()

#--- CONSTANTS

FREE_EXPLORATION = 0
PUSHTASK = 1
PUSHTASKSIMULATION = 2
MOVE_TO_TARGET = 3
MODE = PUSHTASKSIMULATION
#MODE = FREE_EXPLORATION
MODE = MOVE_TO_TARGET


RANDOM_BLOCK_ORI = False
#RANDOM_BLOCK_ORI = True

DIFFERENTBLOCKORIENTATION = True
DIFFERENTBLOCKORIENTATION = False

DIRECTIONGENERALISATION = True
DIRECTIONGENERALISATION = False



NUM_TRAIN_RUNS = 0
NUM_TEST_RUNS = 50

class GazeboInterface():
    """
        Class that handles the interaction with gazebo.
    """
    
    def __init__(self):
         
        self.active = True
        self.lastState = None
        if INTERACTION_STATES:
            self.worldModel = model.ModelCBR()
            self.lastAction = model.Action()
        else:
            self.worldModel = model.ModelAction()
            self.lastAction = model.GripperAction()
        self.lastPrediction = None
        
        self.target = None
        
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
        msg.cmd = action["cmd"]
        
        
            
#        action["mvDir"] *= 0.5
        msg.direction.x = action["mvDir"][0]
        msg.direction.y = action["mvDir"][1]
        msg.direction.z = 0.0
#        msg.direction.z = action.direction[2] # ignore z for now
#        print msg
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
        
    def sendPrediction(self):
        """
        Function to send the last prediction to gazebo. This will move the shadow model
        positions.
        """
        msg = pygazebo.msg.modelState_v_pb2.ModelState_V()
        if INTERACTION_STATES:
            for intState in self.lastPrediction.interactionStates.values():
    #            tmp = self.getModelState(intState["sname"] + "Shadow", intState["spos"], intState["sori"], 
    #                                     self.lastPrediction.transM, self.lastPrediction.quat)
                tmp = self.getModelState2(intState["sname"] + "Shadow", intState["spos"], intState["seuler"], 
                                 self.lastPrediction.transM, self.lastPrediction.ori)
    
                msg.models.extend([tmp])
                if SINGLE_INTSTATE:
                    if DIFFERENCES:
                        tmp = self.getModelState2(intState["oname"] + "Shadow", intState["spos"]+intState["dir"], intState["seuler"]+intState["deuler"], 
                                 self.lastPrediction.transM, self.lastPrediction.ori)
                    else:
                        tmp = self.getModelState2(intState["oname"] + "Shadow", intState["opos"], intState["oeuler"],
                                 self.lastPrediction.transM, self.lastPrediction.ori)
    
                    msg.models.extend([tmp])
        else:
            for objectState in self.lastPrediction.objectStates.values():
                tmp = self.getModelState2(objectState["name"]+"Shadow", objectState["pos"], objectState["euler"], self.lastPrediction.transM, self.lastPrediction.ori)
                msg.models.extend([tmp])
        self.posePublisher.publish(msg)
        
    def sendPrediction2D(self):
        """
        Function to send the last prediction to gazebo. This will move the shadow model
        positions.
        """
        msg = pygazebo.msg.modelState_v_pb2.ModelState_V()
        for intState in self.lastPrediction.interactionStates.values():
#            tmp = self.getModelState(intState["sname"] + "Shadow", intState["spos"], intState["sori"], 
#                                     self.lastPrediction.transM, self.lastPrediction.quat)
            tmp = self.getModelState2D(intState["sname"] + "Shadow", intState["spos"], intState["seuler"], 
                             self.lastPrediction.transM, self.lastPrediction.ori)

            msg.models.extend([tmp])
            if SINGLE_INTSTATE:
                tmp = self.getModelState2D(intState["oname"] + "Shadow", intState["spos"]+intState["dir"], intState["seuler"]+intState["deuler"], 
                             self.lastPrediction.transM, self.lastPrediction.ori)

                msg.models.extend([tmp])
        self.posePublisher.publish(msg)
        
    def getModelState2D(self, name, pos, euler, transM=None, eulerdif=None):
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
#            matrix = np.matrix(np.zeros((4,4)))
#            matrix[:3,:3] = transM[:3,:3].T
#            matrix[:3,3] = -transM[:3,:3].T*transM[:3,3]
#            matrix[3,3] = 1.0
##            print "Original matrix: {} \n, transpose: {}".format(transM, matrix)
            tmpPos = np.matrix(np.concatenate((pos,[1])))
            tpos = np.array((transM*tmpPos.T)[:2]).flatten()   
#            print "OriginalPosition: {}, resulting position: {}".format(tmpPos, pos)
            msg.pose.position.x = tpos[0] 
            msg.pose.position.y = tpos[1] 
            msg.pose.position.z = 0.05
        else:
            msg.pose.position.x = pos[0] #* 2.0
            msg.pose.position.y = pos[1] #* 2.0
            msg.pose.position.z = 0.05 #* 2.0
        
        if eulerdif != None:
            newOri = common.eulerToQuat(np.array([0.0,0.0,euler+eulerdif]))
            msg.pose.orientation.x = newOri[0]
            msg.pose.orientation.y = newOri[1]
            msg.pose.orientation.z = newOri[2]
            msg.pose.orientation.w = newOri[3]
        else:
            ori= common.eulerToQuat(np.array([0.0,0.0,euler]))
            msg.pose.orientation.x = ori[0]
            msg.pose.orientation.y = ori[1]
            msg.pose.orientation.z = ori[2]
            msg.pose.orientation.w = ori[3]
        return msg    

    def getModelState2(self, name, pos, euler, transM=None, eulerdif=None):
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
        
        assert not np.any(np.isnan(pos)), "Pos has nan in it: {}".format(pos)
        assert not np.any(np.isnan(euler)), "euler has nan in it: {}".format(euler)
        if transM != None:
            assert not np.any(np.isnan(transM)), "transM has nan in it: {}".format(transM)
            assert not np.any(np.isnan(eulerdif)), "eulerDif has nan in it: {}".format(eulerdif)
        msg = modelState_pb2.ModelState()
        msg.name = name
        msg.id = 99
#        print "getting model State for: ", name
        if transM != None:
            #Build inverse transformation matrix
#            matrix = np.matrix(np.zeros((4,4)))
#            matrix[:3,:3] = transM[:3,:3].T
#            matrix[:3,3] = -transM[:3,:3].T*transM[:3,3]
#            matrix[3,3] = 1.0
##            print "Original matrix: {} \n, transpose: {}".format(transM, matrix)
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
    
        
    def getModelState(self, name, pos, ori, transM=None, quat=None):
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

        assert not np.any(np.isnan(pos)), "Pos has nan in it: {}".format(pos)
        assert not np.any(np.isnan(ori)), "Ori has nan in it: {}".format(ori)
        if transM != None:
            assert not np.any(np.isnan(transM)), "transM has nan in it: {}".format(transM)
            assert not np.any(np.isnan(quat)), "quat has nan in it: {}".format(quat)
        msg = modelState_pb2.ModelState()
        msg.name = name
        msg.id = 99
#        print "getting model State for: ", name
        if transM != None:
            #Build inverse transformation matrix
#            matrix = np.matrix(np.zeros((4,4)))
#            matrix[:3,:3] = transM[:3,:3].T
#            matrix[:3,3] = -transM[:3,:3].T*transM[:3,3]
#            matrix[3,3] = 1.0
##            print "Original matrix: {} \n, transpose: {}".format(transM, matrix)
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
        
        if quat != None:
            q2 = ori
            q1 = quat
            newOri = np.zeros(4)
            newOri[0] = q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0]
            newOri[1] = -q1[0]*q2[2]+q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1]
            newOri[2] = q1[0]*q2[1]-q1[1]*q2[0]+q1[2]*q2[3]+q1[3]*q2[2]
            newOri[3] = -q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]+q1[3]*q2[3]
            newOri /= np.linalg.norm(newOri)
            msg.pose.orientation.x = newOri[0]
            msg.pose.orientation.y = newOri[1]
            msg.pose.orientation.z = newOri[2]
            msg.pose.orientation.w = newOri[3]
        else:
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
#            print "Parsing worldState with last coordinate system."
#            resultWS = model.WorldState(self.lastPrediction.transM, self.lastPrediction.invTrans, self.lastPrediction.quat)
            resultWS = model.WorldState(self.lastPrediction.transM, self.lastPrediction.invTrans, self.lastPrediction.ori)
            resultWS.parse(worldState)
        else:
            resultWS = None
#        print "parsing new WorldState"
        newWS = model.WorldState()
        newWS.parse(worldState)
        
        if MODE == FREE_EXPLORATION:
            self.randomExploration(newWS, resultWS)
        elif MODE == PUSHTASK:
            self.pushTask(newWS, resultWS)
        elif MODE == PUSHTASKSIMULATION:
            self.pushTaskSimulation(newWS, resultWS)
        elif MODE == MOVE_TO_TARGET:
            self.getTarget()
            self.moveToTarget(newWS, resultWS)
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
        if self.trainRun == 0:
            posX = -0.2656
        elif self.trainRun == 1:
            posX = 0.2656
        elif self.trainRun == 2:
            posX = 0
            
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
        

    def updateTmpErrors(self, worldState):
        """
        Function to update the auxialary error of the last prediction compared to the current world state.
        Only the difference in position and orientation are taken into account.
        
        Paramters
        ---------
        worldState: mode.WorldState
            The current world state
        """
        if self.lastPrediction != None:
            
            predictedWorldState = model.WorldState()
            predictedWorldState.reset(self.lastPrediction)
            gripperPrediction = predictedWorldState.getInteractionState("gripper")
            gripperInt = worldState.getInteractionState("gripper")
            self.tmpGripperErrorPos += np.linalg.norm(gripperPrediction["spos"]-gripperInt["spos"]) 
            self.tmpGripperErrorOri += np.linalg.norm(gripperPrediction["seuler"]-gripperInt["seuler"]) 
            self.tmpBlockErrorPos += np.linalg.norm((gripperPrediction["spos"]+gripperPrediction["dir"])-(gripperInt["spos"]-gripperInt["dir"])) 
            self.tmpBlockErrorOri +=  np.linalg.norm((gripperPrediction["seuler"]+gripperPrediction["deuler"])-(gripperInt["spos"]+gripperInt["deuler"])) 

    def pushTask(self, worldState, resultState=None):
        """
        Task to push straight against a block for NUM_TRAIN_RUNS times to train the model,
        which is then tested for NUM_TEST_RUNS afterwards to compute the prediction error.
        
        Paramters
        ---------
        worldState: mode.WorldState
            The current world state
        """
        self.stepCounter += 1
        if self.runStarted:
            if self.runEnded(worldState):
                self.resetWorld()
                self.runStarted = False
        else:
            if self.testRun > 0:
                if DIRECTIONGENERALISATION:
    #                print "bigger starting variance"
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
            if self.runStarted:
                self.updateModel(worldState, resultState, self.direction)
            else:
                self.trainRun += 1
#                self.startRun()
        
        elif self.testRun < NUM_TEST_RUNS:
            print "Test run #: ", self.testRun
            if self.runStarted:
                print "run started"
                self.updateTmpErrors(worldState)
                self.updateModel(worldState, resultState, self.direction)
            else:
                self.gripperErrorPos += self.tmpGripperErrorPos/self.stepCounter
                self.gripperErrorOri += self.tmpGripperErrorOri/self.stepCounter
                self.tmpGripperErrorPos = 0.0
                self.tmpGripperErrorOri = 0.0
                self.blockErrorPos += self.tmpBlockErrorPos/self.stepCounter
                self.blockErrorOri += self.tmpBlockErrorOri/self.stepCounter
                self.tmpBlockErrorPos = 0.0
                self.tmpBlockErrorOri = 0.0
                self.testRun += 1
#                self.startRun()
        else:
#            self.sendStopCommand()
##            self.pauseWorld()
#            self.blockErrors.append(self.blockError/self.testRun)
#            self.gripperErrors.append(self.gripperError/self.testRun)
            self.times += 1
            with open("Tree1.txt", 'a') as f:
                s = str(self.times* NUM_TEST_RUNS) + ", " + str(self.gripperErrorPos/self.testRun) + ", "  \
                    + str(self.gripperErrorOri/self.testRun) + ', '+ str(self.blockErrorPos/self.testRun) + ', ' \
                    + str(self.blockErrorOri/self.testRun) +"\n"
                f.write(s)
            
#            print "Average Gripper prediction error: {}".format(self.gripperError/self.testRun)
#            print "Average Block prediction error: {}".format(self.blockError/self.testRun)
            self.testRun = 0
            self.gripperErrorPos = 0.0
            self.gripperErrorOri = 0.0
            self.blockErrorPos = 0.0
            self.blockErrorOri = 0.0
        if self.worldModel.numPredictions > 0:
            print "% correctCase selected: ", self.worldModel.numCorrectCase/(float)(self.worldModel.numPredictions)
            
    def pushTaskSimulation(self, worldState, resultState=None):
        self.stepCounter += 1
        print "num cases: " + str(len(self.worldModel.cases))
#        print "num abstract cases: " + str(len(self.worldModel.abstractCases))
        
        if self.runStarted:
            if self.runEnded(worldState):
                self.resetWorld()
                self.runStarted = False
        else:
            if self.testRun > 0:
                if DIRECTIONGENERALISATION:
    #                print "bigger starting variance"
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
            if self.runStarted:
                self.updateModel(worldState, resultState, self.direction)
            else:
                self.trainRun += 1
                if self.trainRun == NUM_TRAIN_RUNS:
                    self.pauseWorld()
#                    dot_data = StringIO()
#                    self.worldModel.getGraphViz(dot_data)
#                    graph = pydot.graph_from_dot_data(dot_data.getvalue())
#                    if graph != None:
#                        graph.write_pdf("../../data/miniMalTree.pdf")
#                    print "ACs: ", [(ac.id, ac.variables) for ac in self.worldModel.abstractCases.values() ]
#                    np.random.seed(1234)
        elif self.testRun < NUM_TEST_RUNS:
            print "Test run #: ", self.testRun
            if self.runStarted:
                self.lastAction = model.Action.getGripperAction(cmd = GAZEBOCMDS["MOVE"], direction=self.direction)
                if self.lastPrediction != None:
                    predictedWorldState = model.WorldState()
                    if INTERACTION_STATES:
                        predictedWorldState.reset(self.lastPrediction)
                    else:
                        predictedWorldState.reset2(self.lastPrediction)
                else:
                    predictedWorldState = worldState
                    #Retransform
#                    print "lastPrediction: {}, worldState: {} ".format(self.lastPrediction.interactionStates, worldState.interactionStates)
                self.lastPrediction = self.worldModel.predict(predictedWorldState, self.lastAction)
#                print "lastAction: ", self.lastAction
                self.sendPrediction()
                self.sendCommand(self.lastAction)
            else:
                self.testRun += 1
#                self.startRun()
        else:
            self.pauseWorld()
        
#        if self.worldModel.numPredictions > 0:
#            print "% correctCase selected: ", self.worldModel.numCorrectCase/(float)(self.worldModel.numPredictions)
#        print "numPredictions: ", self.worldModel.numPredictions
        
            
    def updateModel(self, worldState, resultState, direction=np.array([0.0,0.5,0.0])):
        """
        Function to perform the world update and get the next prediction.
        Currently action NOTHING is performed in here.
        
        Paramters
        ---------
        worldState: mode.WorldState
            The current world state
        """
        if self.lastPrediction != None and resultState != None:
            self.worldModel.update(self.lastState, self.lastAction, self.lastPrediction, resultState)
        
        self.lastState = worldState
#        if self.stepCounter == 1:
#        if self.trainRun < NUM_TRAIN_RUNS-1:
        self.lastAction = model.Action.getGripperAction(cmd = GAZEBOCMDS["MOVE"], direction=direction)
#        else:
#            self.lastAction = model.Action(cmd=GAZEBOCMDS["NOTHING"])
        self.lastPrediction = self.worldModel.predict(worldState, self.lastAction)
        self.sendPrediction()
        self.sendCommand(self.lastAction)


    def randomExploration(self, worldState, resultState):
        """
        Task to perform random/free exploration, driven by the model itself.
                
        
        Paramters
        ---------
        worldState: mode.WorldState
            The current world state
        """
        if self.lastPrediction != None:
            self.worldModel.update(self.lastState, self.lastAction,self.lastPrediction, resultState)
            
        tmp = self.worldModel.getAction(worldState)
        tmp.transform(worldState.transM)
#        tmp = self.getRightAction()
#        tmp["cmd"] = GAZEBOCMDS["MOVE"]
#        tmp["dir"] = np.array([0.0,-1.2,0.0])
        self.lastAction = tmp
        print "lastAction: " + str(self.lastAction)
        
        self.lastState = worldState
        self.lastPrediction = self.worldModel.predict(worldState, self.lastAction)
        self.sendPrediction()
        
        self.sendCommand(self.lastAction)
        print "num cases: " + str(len(self.worldModel.cases))
        print "num abstract cases: " + str(len(self.worldModel.abstractCases))
        print "num Predictions: ", self.worldModel.numPredictions
        print "% correctCase selected: ", self.worldModel.numCorrectCase/(float)(self.worldModel.numPredictions)
        print "avgCorrectPredictionScore: ", self.worldModel.avgCorrectPrediction
#        if self.worldModel.numPredictions == 100:
#            self.pauseWorld()
#            from sklearn.decomposition import PCA, KernelPCA
#            
##            kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
#            kpca = PCA()
#            transformed = kpca.fit_transform(self.worldModel.data)
#            print "Shape data: ", np.shape(self.worldModel.data)
#            print "Shape transformed: ", np.shape(transformed)
#            print "transformed: ", transformed
#            dot_data = StringIO()
#            self.worldModel.getGraphViz(dot_data)
#            graph = pydot.graph_from_dot_data(dot_data.getvalue())
#            if graph != None:
#                graph.write_pdf("treeExploration.pdf")
#            self.worldModel.setTarget(self.getTarget(worldState))
        
    def startRunTarget(self, randomRange=0.5):
        self.runStarted = True
        posX = ((np.random.rand()-0.5)*randomRange) #* 0.5
        self.sendPose("gripper", np.array([posX,0.0,0.03]), np.array([0.0,0.0,0.0,0.0]))
        self.sendPose("blockA", np.array([0.0,0.075,0.05]), np.array([0.0,0.0,0.0,0.0]))
        self.stepCounter = 0
        
            
    def moveToTarget(self, worldState, resultState=None):
        if self.trainRun == NUM_TRAIN_RUNS:
            self.worldModel.setTarget(self.target)
        self.stepCounter += 1
        if self.runStarted:
            #Check if run has ended
        
            gripperInt = worldState.getInteractionState("gripper")
            if DIFFERENCES:
                tmpBlockPos = np.matrix(np.concatenate((gripperInt["spos"]+gripperInt["dir"],[1])))
                tmpBlockOri = np.matrix(np.concatenate((gripperInt["seuler"]+gripperInt["deuler"],[1])))
            else:
                tmpBlockPos = np.matrix(np.concatenate((gripperInt["opos"],[1])))
                tmpBlockOri = np.matrix(np.concatenate((gripperInt["oeuler"],[1])))
            blockPos = np.array((worldState.transM*tmpBlockPos.T)[:3]).flatten()   
            blockOri = np.array((worldState.transM*tmpBlockOri.T)[:3]).flatten()   
            tmpGPos = np.matrix(np.concatenate((gripperInt["spos"],[1])))
            gPos = np.array((worldState.transM*tmpGPos.T)[:3]).flatten()   
#            print "Block pos: ", blockPos
            targetOs = gripperInt.getObjectState(self.target["name"])
            targetOs.transform(worldState.transM, worldState.ori)
            
            if self.target.score(targetOs) > 0.95*len(self.target.relKeys) or blockPos[1] > 1.4 or self.stepCounter > 500 or np.linalg.norm(gPos) > 1.5:
                self.resetWorld()
                self.runStarted = False
            else:
                if np.linalg.norm(blockPos-gPos) < 0.2:
                    self.direction = self.target["pos"] - gPos
                else:
                    self.direction =  blockPos - gPos
                norm = np.linalg.norm(self.direction)
                if norm > 0.5:
#                    print "adapt norm"
                    self.direction /= 2*norm
                if norm < 0.2:
                    if np.random.rand() > 0.8:
                        self.direction= (np.random.rand(3)*2-1) / 5
                    else:
                        self.direction *= 2
        else:
            self.startRunTarget()
            return
            
        if self.trainRun < NUM_TRAIN_RUNS:
            print "Train run #: ", self.trainRun
            if self.runStarted:
                if self.lastPrediction != None:
                    self.worldModel.update(self.lastState, self.lastAction, self.lastPrediction, resultState)
                
#                self.lastAction = model.Action(cmd = GAZEBOCMDS["MOVE"], direction=self.direction)

                self.lastAction = self.worldModel.getAction(worldState)
                self.lastAction.transform(worldState.transM)
                
                
                self.lastState = worldState
                self.lastPrediction = self.worldModel.predict(worldState, self.lastAction)
                self.sendPrediction()
                self.sendCommand(self.lastAction)
            else:
                self.trainRun += 1
                if self.trainRun == NUM_TRAIN_RUNS:
                    self.pauseWorld()
                
            
        elif self.testRun < NUM_TEST_RUNS:
            print "Test run #: ", self.testRun
            if self.runStarted:
                if self.lastPrediction != None:
                    self.worldModel.update(self.lastState, self.lastAction, self.lastPrediction, resultState)
                
                self.lastAction = self.worldModel.getAction(worldState)
                self.lastAction.transform(worldState.transM)
#                print "global action: ", self.lastAction
                self.lastState = worldState
                self.lastPrediction = self.worldModel.predict(worldState, self.lastAction)
                self.sendPrediction()
                self.sendCommand(self.lastAction)
            else:
                self.testRun += 1
        if self.worldModel.numPredictions > 0:
            print "% correctCase selected: ", self.worldModel.numCorrectCase/(float)(self.worldModel.numPredictions)


        
    def getTarget(self):
        target = model.ObjectState()
#        target["name"] = "gripper"
#        target["pos"] = np.array([0.0, -0.5, 0.03])
        target["name"] = "blockA"
        target["pos"] = np.array([0.0, 0.75, 0.05])
        target["euler"] = np.zeros(3)
        target["euler"][2] = 0.5*math.pi
#        target.relKeys = ["pos", "euler"]
        target.relKeys = ["euler"]
#        target.weights = {"pos":20, "euler": 1}
        self.target = target

    def getTargetOld(self, worldState):
        gripper = None
        block = None
        for i in worldState.objectStates.values():
            if i["name"] == "gripper":
                gripper = copy.deepcopy(i)
            if i["name"] == "blockA":
                block = copy.deepcopy(i)
                
        
        gripper["pos"] = np.array([0.0,0.0,0.03])
        gripper["linVel"] = np.array([0.0,0.0,0.0])
        intState = model.InteractionState(0, gripper)
        intState.relKeys = ["spos", "slinVel"]
        intState.fill(block)
        intState.weights["spos"] = 30
        return intState

    
    def stop(self):
        """
        Function to stop the loop.
        """
        self.active = False
    
if __name__ == "__main__":


    loop = trollius.get_event_loop()
    gi = GazeboInterface()
    loop.run_until_complete(gi.loop())
    