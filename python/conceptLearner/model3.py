#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:20:05 2015

@author: jpoeppel
"""

"""
Metrics to be used when comparing features
"""
from metrics import metrics
import numpy as np

THRESHOLD = 0.1

class BaseCase(object):
    
    def __init__(self):
        self.preState
        self.postState
        self.action
        self.dif = {}
        
    def getListOfAttribs(self):
        """
            Returns the list of attributes that changed more than THRESHOLD.
        """
        r = []
        for k in self.dif.keys():
            if np.linalg.norm(self.dif[k]) > THRESHOLD:
                r.append(k)
        return r
        
class AbstractCase(object):
    
    def __init__(self):
        self.refCases = []
        self.avgPrediction = 0.0
        self.name = ""
        self.attribs = []
        
    def predict(self):
        pass
    
class State(object):
    """
        Base class representing the state of something. Will be used to
        derive specific state classes from.
    """
    
    def __init__(self):
        self.features = {}
        pass
    
    def score(self, otherState):
        assert isinstance(otherState, State), "{} is not a State object".format(otherState)
        s = 0.0
        for k in self.features.keys():
            s += metrics[k](self.features[k], otherState.features[k]) #* weights[k] 
        return s
    
class ObjectState(State):
    """
        State class used to represent object states.
        Holds information about object position, orientation, type, id, name 
        and other properties.
    """
    
    def __init__(self):
        self.features = {"id": 0, "name": "", "type":0, "pos":np.zeros(3), 
                         "orientation": np.zeros(4), "linVel": np.zeros(3), 
                         "angVel": np.zeros(3)}
        
        
class InteractionState(State):
    """
        State class used to represent interaction states.
        Holds information about object pairs: Participants A and B, distance, 
        direction, contact yes/no etc.
    """
    
    def __init__(self, objA, objB):
        assert isinstance(objA, ObjectState), "{}(A) is not an ObjectState object".format(objA)
        assert isinstance(objB, ObjectState), "{}(B) is not an ObjectState object".format(objB)
        self.features = {"A":objA, "B": objB, "dir":objB["pos"]-objA["pos"],
                         "dist": np.linalg.norm(objB["pos"]-objA["pos"]), "contact": 0}
        pass
    
class WorldState(object):
    """
        State class used to represent a current world situation.
        Holds object and interaction states for all objects and interactions.
    """
    
    def __init__(self):
        self.objectStates = {}
        self.interactionStates = {}
        pass
    
    def parse(self, models):
        pass
    
        
        
if __name__ == "__main__":
    a = State()
    b = State()
    print a.score(b)
    print metrics["dir"](1,2)