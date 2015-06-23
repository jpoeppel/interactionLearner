#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:12:06 2015
Similar to state2, but worldstate will only consist out of 1 InteractionState!
@author: jpoeppel
"""

import common
from common import NUMDEC
from common import GAZEBOCMDS as GZCMD
from metrics import similarities
from metrics import differences

import numpy as np
import math
import copy
from operator import methodcaller, itemgetter
from state2 import State, ObjectState, InteractionState, Action
import state2
from config import INTERACTION_STATES
            
class WorldState(state2.WorldState):
    

    def parseInteractions(self):
        tmpList = self.objectStates.values()
        gripperO = None
        for o in tmpList:
            #Transform to local block coordinate system
            o.transform(self.invTrans, -self.ori)
            if o["name"] == "gripper":
                gripperO = o
        for o1 in self.objectStates.values():
            intState = InteractionState(self.numIntStates, gripperO)
            if not np.array_equal(o1,gripperO):             
                intState.fill(o1)
                self.addInteractionState(intState)
                
    def reset2(self, worldState):
        self.objectStates = {}
        self.interactionStates = {}
        ws = copy.deepcopy(worldState)
        for os in ws.objectStates.values():
            #Transform back to world coordinate system first
            os.transform(ws.transM, ws.ori)
            if INTERACTION_STATES and os["name"] == "blockA":
                self.transM = common.eulerPosToTransformation(os["euler"],os["pos"])
                self.invTrans = common.invertTransMatrix(self.transM)
                self.ori = np.copy(os["euler"])
            self.addObjectState(os)
        if not INTERACTION_STATES:
            self.transM = np.identity(4)
            self.invTrans = np.identity(4)
            self.ori = 0.0            
            
        self.parseInteractions()
                   
                
    def reset(self, worldState):
        self.objectStates = {}
        self.interactionStates = {}
        ws = copy.deepcopy(worldState)
        for intState in ws.interactionStates.values():
            tmp = ObjectState()
            tmp.fromInteractionState(intState)
            #Transform back to world coordinate system first
            tmp.transform(ws.transM, ws.ori)
            tmp2 = ObjectState()
            tmp2.fromInteractionState2(intState)
            tmp2.transform(ws.transM, ws.ori)
#            print "Tmp after back transformation: ", tmp
            if not self.objectStates.has_key(tmp["name"]):
                self.objectStates[tmp["name"]] = tmp
            if not self.objectStates.has_key(tmp2["name"]):
                self.objectStates[tmp2["name"]] = tmp2
            if INTERACTION_STATES:
                if tmp["name"] == "blockA":
                    self.transM = common.eulerPosToTransformation(tmp["euler"],tmp["pos"])
                    self.invTrans = common.invertTransMatrix(self.transM)
                    self.ori = np.copy(tmp["euler"])
                if tmp2["name"] == "blockA":
                    self.transM = common.eulerPosToTransformation(tmp2["euler"],tmp2["pos"])
                    self.invTrans = common.invertTransMatrix(self.transM)
                    self.ori = np.copy(tmp2["euler"])
        self.parseInteractions()
#        
#        print "InteractionStates: ", self.interactionStates.values()


   
    def items(self):
        return self.getInteractionState("gripper").relevantItems()
        
    def toVec(self, unusedFeatures):
        return self.getInteractionState("gripper").toVec(unusedFeatures)
        
    def __getitem__(self, k):
        return self.getInteractionState("gripper")[k]