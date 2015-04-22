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


logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
import model4 as model

logging.basicConfig()
#

FREE_EXPLORATION = 0
PUSHTASK = 1
MODE = PUSHTASK



NUM_TRAIN_RUNS = 0
NUM_TEST_RUNS = 10

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
        
        self.gripperError = 0.0
        self.gripperErrors = []
        self.tmpGripperError = 0.0
        self.blockError = 0.0
        self.blockErrors = []
        self.tmpBlockError = 0.0
        
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
#                    'gazeboPlugins.msgs.ModelState_V'))
                    
                          
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
        for intState in self.lastPrediction.interactionStates.values():
            tmp = self.getModelState(intState["sname"] + "Shadow", intState["spos"], intState["sori"])
            msg.models.extend([tmp])
        self.posePublisher.publish(msg)
        
    def getModelState(self, name, pos, ori):
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
        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1]
        msg.pose.position.z = pos[2]
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

        w = model.WorldState()
        
        w.parse(worldState)
        
        if MODE == FREE_EXPLORATION:
            self.randomExploration(w)
        elif MODE == PUSHTASK:
            self.pushTask(w)
        else:
            raise AttributeError("Unknown MODE: ", MODE)
            
#            
#        if self.worldModel.numPredictions == 31206:
#            raise Exception("Finished")
#            
#        if len(self.worldModel.cases) == 100 or len(self.worldModel.cases) == 101:
##            self.worldModel.setTarget(self.getTarget(w))
#            self.resetWorld()
#
#        for ac in self.worldModel.abstractCases:
#            print "number of refs: {} for abstract case variables: {}".format(len(ac.refCases),ac.variables)
#        print "abstract lists: " + str([c.variables for c in self.worldModel.abstractCases])

    def runEnded(self, worldState):
        """
        Function to determine if a run has ended. In this case a run has ended, when the
        gripper did not move more than a threshold compared to the last State.
        
        Parameters
        ----------
        worldState: model.WorldState
            The current world state.
        """
        if self.lastState != None:
#            gripperIntLast = self.lastState.getInteractionState("gripper")
            gripperInt = worldState.getInteractionState("gripper")
            if np.linalg.norm(gripperInt["spos"]) > 0.9 or self.stepCounter > 100:
                return True
        return False

    def startRun(self, worldState):
        """
        Function to start a new Run. This means that the inital action is set and the first
        prediction is made.
        
        Paramters
        ---------
        worldState: mode.WorldState
            The current world state
        """
        
        self.runStarted = True
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
            gripperPrediction = self.lastPrediction.getInteractionState("gripper")
            gripperInt = worldState.getInteractionState("gripper")
            self.tmpGripperError += np.linalg.norm(gripperPrediction["spos"]-gripperInt["spos"]) + \
                np.linalg.norm(gripperPrediction["sori"]-gripperInt["sori"]) 
            blockPrediction= self.lastPrediction.getInteractionState("blockA")
            blockInt = worldState.getInteractionState("blockA")
            self.tmpBlockError += np.linalg.norm(blockPrediction["spos"]-blockInt["spos"]) + \
                np.linalg.norm(blockPrediction["sori"]-blockInt["sori"])

    def pushTask(self, worldState):
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
                #Set up Starting position
                posX = np.random.rand()*0.5-0.25
                self.sendPose("gripper", np.array([posX,0.0,0.0]), np.array([0.0,0.0,0.0,0.0]))
                self.runStarted = False
            
        if self.trainRun < NUM_TRAIN_RUNS:
            print "Train run #: ", self.trainRun
            if self.runStarted:
                self.updateModel(worldState)
            else:
                self.trainRun += 1
                self.startRun(worldState)
        
        elif self.testRun < NUM_TEST_RUNS:
            print "Test run #: ", self.testRun
            if self.runStarted:
                print "run started"
                self.updateTmpErrors(worldState)
                self.updateModel(worldState)
            else:
                self.gripperError += self.tmpGripperError/self.stepCounter
                self.tmpGripperError = 0.0
                self.blockError += self.tmpBlockError/self.stepCounter
                self.tmpBlockError = 0.0
                self.testRun += 1
                self.startRun(worldState)
        else:
#            self.sendStopCommand()
##            self.pauseWorld()
#            self.blockErrors.append(self.blockError/self.testRun)
#            self.gripperErrors.append(self.gripperError/self.testRun)
            self.times += 1
            with open("testfile.txt", 'a') as f:
                s = str(self.times* NUM_TEST_RUNS) + ", " + str(self.gripperError/self.testRun) + ", " + str(self.blockError/self.testRun) + "\n"
                f.write(s)
            
            print "Average Gripper prediction error: {}".format(self.gripperError/self.testRun)
            print "Average Block prediction error: {}".format(self.blockError/self.testRun)
            self.testRun = 0
            self.gripperError = 0
            self.blockError = 0
            
    def updateModel(self, worldState):
        """
        Function to perform the world update and get the next prediction.
        Currently action NOTHING is performed in here.
        
        Paramters
        ---------
        worldState: mode.WorldState
            The current world state
        """
        print "update model stepCounter: ", self.stepCounter
        if self.lastPrediction != None:
            self.worldModel.update(self.lastState, self.lastAction,self.lastPrediction, worldState)
        
        self.lastState = worldState
#        if self.stepCounter == 1:
        self.lastAction = model.Action(cmd = GAZEBOCMDS["MOVE"], direction=np.array([0.0,0.5,0.0]))
#        else:
#            self.lastAction = model.Action(cmd=GAZEBOCMDS["NOTHING"])
        self.lastPrediction = self.worldModel.predict(worldState, self.lastAction)
        self.sendPrediction()
        self.sendCommand(self.lastAction)

    def randomExploration(self, worldState):
        """
        Task to perform random/free exploration, driven by the model itself.
                
        
        Paramters
        ---------
        worldState: mode.WorldState
            The current world state
        """
        if self.lastPrediction != None:
            self.worldModel.update(self.lastState, self.lastAction,self.lastPrediction, worldState)
            
        tmp = self.worldModel.getAction(worldState)
        
        norm = np.linalg.norm(tmp["mvDir"])
        if norm > 1:
            tmp["mvDir"] /= 2*norm
#        tmp = self.getRightAction()
#        tmp["cmd"] = GAZEBOCMDS["MOVE"]
#        tmp["dir"] = np.array([0.0,-1.2,0.0])
        self.lastAction = tmp
#        print "lastAction: " + str(self.lastAction)
        
        self.lastState = worldState
        self.lastPrediction = self.worldModel.predict(worldState, self.lastAction)
        self.sendPrediction()
        
        self.sendCommand(self.lastAction)
        print "num cases: " + str(len(self.worldModel.cases))
        print "num abstract cases: " + str(len(self.worldModel.abstractCases))
        print "num Predictions: ", self.worldModel.numPredictions
        print "% correctCase selected: ", self.worldModel.numCorrectCase/(float)(self.worldModel.numPredictions)
        

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
    