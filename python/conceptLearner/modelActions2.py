#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:56:19 2015
Action model that separates actuators from simple objects.
All object changes can only be induced by actuators!

For now (12.8) assume, that object's static properties can only be changed directly
by actuators (no sliding etc after contact is broken)

Idea for multiple objects: For gate function: Start from actuator and only consider 
objects that are changed to have a potential influence on other objects.

@author: jpoeppel
"""

import numpy as np
from sklearn import neighbors
from sklearn import svm

from state4 import WorldState as ws4

import common

class Object(object):
    
    def __init__(self):
        self.id = 0
        self.vec = np.array([])
        self.intStates = None
        
    def getRelativeVec(self, other):
        """ Computes a feature vector for the interaction relative to other.
            vector: [refId(other), otherId(self), dist, 
                     relPosx, relPosy, relPosz,
                     relVelx, relVely, relVelz, 
                     relAng]
        """
#        vec = np.zeros(10)
#        vec[0] = other.id
#        vec[1] = self.id
#        vec[2] = 0.0 #TODO Dist
#        vec[3:6], vec[6:9] = common.relPosVel(other.vec[1:4], other.vec[5:8], other.vec[4], self.vec[1:4],self.vec[5:8])
#        vec[9] = self.vec[8]-other.vec[8]
        for intS in other.intStates:
            if intS["oid"] == self.id:
                return intS.vec
#        return vec
        
        
    def __repr__(self):
        return "{}".format(self.id)

class Actuator(Object):
    
    def __init__(self):
        Object.__init__(self)
        pass
    
class WorldState(object):
    
    def __init__(self):
        self.actuator = Actuator()
        self.objectStates = {}
        
    def parse(self, gzWS):
        ws = ws4()
        ws.parse(gzWS)
        for oN, o in ws.objectStates.items():
            if oN == "gripper":
                self.actuator.id = o["id"][0]
                self.actuator.vec = np.copy(o.vec)
                self.actuator.intStates = ws.getInteractionStates("gripper")
            else:
                newO = Object()
                newO.id = o["id"][0]
                newO.vec = np.copy(o.vec)
                newO.intStates = ws.getInteractionStates(oN)
                self.objectStates[newO.id] = newO
                
    
class Classifier(object):
    
    def __init__(self):
#        self.clas = neighbors.KNeighborsClassifier(n_neighbors=2, weights='uniform')
        self.clas = svm.SVC()
        self.inputs = []
        self.targets = []
        
    def train(self, o1vec, avec, label):
#        self.inputs.append(np.concatenate((o1vec,avec)))
#        self.targets.append(label)
#        if max(self.targets) > 0:
#            self.clas.fit(self.inputs, self.targets)
        pass
    
    def test(self, ovec, avec):
        print "closing: {}, dist: {}".format(ovec[3], ovec[2])
        if ovec[3] <= -10*ovec[2]:
            return 1
        else:
            if ovec[3] == 0 and np.linalg.norm(ovec[8:11]) < 0.01:
                return 1    
            else:
                return 0
#        if len(self.targets) > 0 and max(self.targets) > 0:
#            return self.clas.predict(np.concatenate((ovec,avec)))[0]
#        else:
#            return 0
    
    
class GateFunction(object):
    
    def __init__(self):
        self.classifier = Classifier()
        pass
    
    def test(self, o1, o2, action):
        vec = o2.getRelativeVec(o1)
        return self.classifier.test(vec,action)
        
    def checkChange(self, pre, post):
        dif = np.abs(pre.vec - post.vec)
#        print "dif: ", dif
        if np.linalg.norm(dif[1:4]) > 0.015 or dif[4] > 0.02:
#            print "Change"
            return True
#        print "No change"
        return False
        
        
    def update(self, o1Pre, o1Post, o2, action):
        #TODO Causal determination, make hypothesis and test these!
        vec = o2.getRelativeVec(o1Pre)
        if self.checkChange(o1Pre, o1Post):
            self.classifier.train(vec,action, 1)
        else:
            self.classifier.train(vec,action, 0)


class ModelAction(object):
    
    def __init__(self):
        self.gate = GateFunction()
        self.actuator = Actuator()
        self.predictors = None
        
    def predict(self, ws, action):
        for o in ws.objectStates.values():
            print "Testresult for {}: {}".format(o,self.gate.test(o, ws.actuator, action))
        
    def update(self, oldWS, action, newWS):
        for o in oldWS.objectStates.values():
            #TODO extent to more objects
            self.gate.update(o, newWS.objectStates[o.id], oldWS.actuator, action)
    
    