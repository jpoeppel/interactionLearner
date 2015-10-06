#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:01:44 2015
Compact reimplementation of the interaction model very closely to the idea 
written in the thesis.
Already focuses on 2D only
By reimplementing from scatrch, we can hopefully avoid problems with the legacy components in 
the old model.
@author: jpoeppel
"""

import numpy as np

USE_DYNS = False

class Object(object):
    
    def __init__(self):
        self.id = 0
        self.vec = np.array([])
        self.lastVec = np.array([])
        pass
    
    @classmethod
    def parse(cls, m):
        res = cls()
        res.id = m.id 
        if USE_DYNS:
            res.vec = np.zeros(6)
            res.vec[0] = npround(m.pose.position.x, NUMDEC) #posX
            res.vec[1] = npround(m.pose.position.y, NUMDEC) #posY
            res.vec[2] = npround(common.quaternionToEuler(np.array([m.pose.orientation.x,m.pose.orientation.y,
                                                m.pose.orientation.z,m.pose.orientation.w])), NUMDEC)[2] #ori
            res.vec[3] = npround(m.linVel.x, NUMDEC) #linVelX
            res.vec[4] = npround(m.linVel.y, NUMDEC) #linVelY
            res.vec[5] = npround(m.angVel.z, NUMDEC) #angVel
        else:
            res.vec = np.zeros(3)
            res.vec[0] = npround(m.pose.position.x, NUMDEC) #posX
            res.vec[1] = npround(m.pose.position.y, NUMDEC) #posY
            res.vec[2] = npround(common.quaternionToEuler(np.array([m.pose.orientation.x,m.pose.orientation.y,
                                                m.pose.orientation.z,m.pose.orientation.w])), NUMDEC)[2] #ori
        res.lastVec = np.copy(res.vec)
        return res
        
    @classmethod
    def fromInteractionState(cls, intState):
        pass
    
class InteractionState(object):
    
    def __init__(self):
        self.id = ""
        self.vec = np.array([])
        self.lastVec = np.array([])
        pass
    
    @classmethod
    def fromObjectStates(cls, o1, o2):
        pass
    
class WorldState(object):
    
    def __init__(self):
        self.objectStates = {}
        self.interactionStates = {}
        
    def parseModels(self, models):
        for m in models:
            if m.name == "ground_plane" or "wall" in m.name or "Shadow" in m.name:
                continue
            else:
                tmp = Object.parse(m)               
                self.objectStates[tmp.id] = tmp
                
    def parseInteractions(self):
        for n1, os1 in self.objectStates.items():
            if n1 != 15:
                for n2, os2 in self.objectStates.items():
                    if n1 != n2:
                        intState = InteractionState.fromObjectStates(os1,os2)
                        self.interactionStates[intState.id] = intState
        
    def parse(self, gzWS):
        self.parseModels(gzWS.model_v.models)   
        self.parseInteractions()
    
class Action(object):
    
    def __init__(self):
        self.vec = np.zeros(3)
        pass

class Episode(object):
    
    def __init__(self):
        self.preState = None
        self.action = None
        self.postState = None
        pass
    
class AbstractCollection(object):
    
    def __init__(self):
        pass


class ModelInteraction(object):

    def __init__(self):
        pass    