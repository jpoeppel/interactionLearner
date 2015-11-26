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
#        tmpList = self.objectStates.values()
        gripperO = None
        for o in self.objectStates.values():
            #Transform to local block coordinate system
#            o.transform(self.invTrans, -self.ori)
            if o["name"] == "gripper":
                gripperO = o.transform(self.invTrans, -self.ori)
        for o1 in self.objectStates.values():
            intState = InteractionState(self.numIntStates, gripperO)
            if not np.array_equal(o1,gripperO):   
                intState.fill(o1.transform(self.invTrans, -self.ori))
                self.addInteractionState(intState)
#        
#        print "InteractionStates: ", self.interactionStates.values()


   
    def items(self):
        return self.getInteractionState("gripper").relevantItems()
        
    def toVec(self, unusedFeatures):
        return self.getInteractionState("gripper").toVec(unusedFeatures)
        
    def __getitem__(self, k):
        return self.getInteractionState("gripper")[k]