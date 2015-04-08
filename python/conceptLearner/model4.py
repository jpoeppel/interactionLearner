#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:34:43 2015

@author: jpoeppel
"""

import numpy as np
from metrics import metrics2 as metrics

THRESHOLD = 0.1
NUMDEC = 5

class State(dict):
    """
        Base class representing the state of something. Will be used to
        derive specific state classes from.
    """
    
    def __init__(self):
        pass
    
    def score(self, otherState):
        assert isinstance(otherState, State), "{} is not a State object".format(otherState)
        s = 0.0
        for k in self.keys():
            s += metrics[k](self[k], otherState[k]) #* weights[k] 
        return s
        
    def toVec(self):
        r = np.array([])
        for v in self.values():
            if isinstance(v, np.ndarray):
                r = np.concatenate((r,v))
            elif not isinstance(v, unicode):
                r = np.concatenate((r,[v]))
        return r
        
    def __repr__(self):
        return str(self.features)
        
        
class InteractionState(State):
    
    def __init__(self):
        self.update({"sid":0, "sname": "", "stype": 0, "spos":np.zeros(3), 
                     "sori": np.zeros(4), "slinVel": np.zeros(3), 
                     "sangVel": np.zeros(3), "dist": 0, "dir": np.zeros(3),
                     "contact": 0, "oid": 1, "oname": "", "otype": 0, 
                     "opos": np.zeros(3), "oori": np.zeros(4), 
                     "olinVel": np.zeros(3), "oangVel":np.zeros(3)})
                     
    def parse(self, s, o):
        pass
    

class BaseCase(object):
    
    def __init__(self, pre, action, post):
        assert isinstance(pre, State), "{} is not a State object.".format(pre)
        assert isinstance(post, State), "{} is not a State object.".format(post)
        assert isinstance(action, Action), "{} is not an Action object.".format(action)
        assert (pre.keys()==post.keys()), "Pre and post states have different keys: {}, {}.".format(pre.keys(), post.keys())
        self.preState = pre
        self.postState = posr
        self.action = action
        self.dif = {}
        for k in pre.keys():
            self.dif[k] = post[k]-pre[k]
        
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
        self.numPredictions = 0
        self.name = ""
        self.attribs = [] #List of attributes that changed 
        self.constants = {} # Dictionary holding the attributs:values whose values are constant for all references
        self.predictors = {}
        
    def predict(self, state, action):
        pass