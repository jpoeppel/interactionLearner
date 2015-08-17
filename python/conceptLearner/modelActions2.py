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
#from sklearn import neighbors
from sklearn import svm

from state4 import WorldState as ws4

import common
from topoMaps import ITM
from network import Node
import copy

HARDCODED = True

class Object(object):
    
    def __init__(self):
        self.id = 0
        self.vec = np.array([])
        self.intStates = None
        
                
    def getRelVec(self, other):
        """
            Computes the relative (interaction) vector of the other object with respect
            to the own reference frame.
        """
        vec = np.zeros(10)
        vec[0] = self.id
        vec[1] = other.id
        vec[2], vec[3] = common.computeDistanceClosing(self.id, self.vec[1:4],self.vec[5:8], 
                        self.vec[4], other.id, other.vec[1:4], other.vec[5:8], other.vec[4])
        vec[4:7], vec[7:10] = common.relPosVel(self.vec[1:4], self.vec[5:8], self.vec[4], other.vec[1:4], other.vec[5:8])
     #   vec[10] = other.vec[8]-self.vec[8]
        return vec
                
    def getIntVec(self, other):
        """
            Returns the interaction state with the other
        """
        for intS in self.intStates:
            if intS["oid"] == other.id:
                return intS.vec

    def predict(self, predictor, other):
        resO = copy.deepcopy(self)
        print "object before: ", resO.vec[1:4]
#        print "relVec: ", self.getRelVec(other)
        pred = predictor.predict(self.getRelVec(other))
#        print "prediction for o: {}: {}".format(self.id, pred)
        resO.vec += pred
        resO.vec = np.round(resO.vec, common.NUMDEC)
        print "resulting object: ", resO.vec[1:4]
        return resO
        
    def update(self, newO):
        self.vec = np.copy(newO.vec)
        self.intStates = newO.intStates
        
    def __repr__(self):
        return "{}".format(self.id)

class Actuator(Object):
    
    def __init__(self):
        Object.__init__(self)
        self.predictor = ITM()
        self.vec = np.zeros(9)
        pass
    
    def predict(self, action):
        res = copy.deepcopy(self)
        res.vec[5:8] = action #Set velocity
        #Hardcorded version
        if HARDCODED:
            res.vec = np.copy(self.vec)
            res.vec[5:8] = action
            res.vec[1:4] += 0.01*action
        else:
            #Only predict position
            res.vec[1:4] += self.predictor.predict(action)
        res.vec = np.round(res.vec, common.NUMDEC)
        return res
            
    def update(self, newAc, action):
        if HARDCODED:
            pass
        else:
            self.predictor.train(Node(0, wIn=action, wOut=newAc.vec[1:4]-self.vec[1:4]))
        self.vec = np.copy(newAc.vec)
        self.intStates = newAc.intStates
    
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
        if HARDCODED:
            pass
        else:
            self.inputs.append(np.concatenate((o1vec,avec)))
            self.targets.append(label)
            if max(self.targets) > 0:
                self.clas.fit(self.inputs, self.targets)
    
    def test(self, ovec, avec):
#        print "closing: {}, dist: {}".format(ovec[3], ovec[2])
        if HARDCODED:
            if ovec[3] <= -100*ovec[2]:
                return 1
            else:
                if ovec[3] == 0 and np.linalg.norm(ovec[8:11]) < 0.01 and ovec[2] < 0.1: #Todo remove distance from this
                    return 1    
                else:
                    return 0
        else:
            if len(self.targets) > 0 and max(self.targets) > 0:
                return self.clas.predict(np.concatenate((ovec,avec)))[0]
            else:
                return 0
    
    
class GateFunction(object):
    
    def __init__(self):
        self.classifier = Classifier()
        
        pass
    
    def test(self, o1, o2, action):
        vec = o1.getRelVec(o2)
        return self.classifier.test(vec,action)
        
    def checkChange(self, pre, post):
        dif = post.vec-pre.vec
#        print "dif: ", dif[1:4]
        if np.linalg.norm(dif[1:4]) > 0.0 or abs(dif[4]) > 0.0:
#            print "Change"
            return True, dif
#        print "No change"
        return False, dif
        
        
    def update(self, o1Pre, o1Post, o2, action):
        """
        Parameters
        ----------
        o1Pre: Object
        o1Post: Object
        o2: Object
        action: np.ndarray
        """
        #TODO Causal determination, make hypothesis and test these!
#        vec = o2.getRelativeVec(o1Pre)
        
        vec = o1Pre.getRelVec(o2)
        hasChanged, dif = self.checkChange(o1Pre, o1Post)
        if hasChanged:
            self.classifier.train(vec,action, 1)
            return True, dif
        else:
            self.classifier.train(vec,action, 0)
            return False, dif
            
class Predictor(object):
    
    def __init__(self):
        self.predictors = {}
    
    def predict(self, o1, o2, action):
        if o1.id in self.predictors:
            return o1.predict(self.predictors[o1.id], o2)
        else:
            return o1
    
    def update(self, intState, action, dif):
        if not intState[0] in self.predictors:
            #TODO check for close ones that can be used
            self.predictors[intState[0]] = ITM()
#        print "updating with dif: ", dif[3]
        if dif[3] != 0:
            print "dif in height: ", dif
            raise AttributeError
        self.predictors[intState[0]].train(Node(0, wIn = intState, wOut=dif))


class ModelAction(object):
    
    def __init__(self):
        self.gate = GateFunction()
        self.actuator = None
        self.predictor = Predictor()
        self.curObjects = {}
        
    def predict(self, ws, action):
#        if self.actuator == None:
#            self.actuator = ws.actuator
        newWS = WorldState()
        newWS.actuator = ws.actuator.predict(action)
        for o in ws.objectStates.values():
            if self.gate.test(o, ws.actuator, action):
#                print "predicted change"
                newO = self.predictor.predict(o, ws.actuator, action)
                newWS.objectStates[o.id] = newO
            else:
#                print "predicted no change"
                o.vec[5:] = 0.0
                newWS.objectStates[o.id] = o
        return newWS
        
    def update(self, curWS, action):
        
        for o in curWS.objectStates.values():
            #TODO extent to more objects
            if o.id in self.curObjects:
                hasChanged, dif = self.gate.update(self.curObjects[o.id], o, self.actuator, action)
                if hasChanged:
                    self.predictor.update(o.getRelVec(self.actuator), action, dif)
                self.curObjects[o.id].update(curWS.objectStates[o.id])
            else:
                self.curObjects[o.id] = o
                
        if self.actuator == None:
            self.actuator = curWS.actuator
        self.actuator.update(curWS.actuator, action)
            
    