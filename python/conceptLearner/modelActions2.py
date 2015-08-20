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

HARDCODEDGATE = True
HARDCODEDACTUATOR = True

class Object(object):
    
    def __init__(self):
        self.id = 0
        self.vec = np.array([])
        self.intStates = None
        self.lastVec = np.array([])
        self.predVec= np.array([])
        
        
                
    def getRelVec(self, other):
        """
            Computes the relative (interaction) vector of the other object with respect
            to the own reference frame.
        """
        vec = np.zeros(11)
        vec[0] = self.id
        vec[1] = other.id
        vec[2], vec[3] = common.computeDistanceClosing(self.id, self.vec[1:4],self.vec[5:8], 
                        self.vec[4], other.id, other.vec[1:4], other.vec[5:8], other.vec[4])
        vec[4:7], vec[7:10] = common.relPosVel(self.vec[1:4], self.vec[5:8], self.vec[4], other.vec[1:4], other.vec[5:8])
#        vec[2],vec[3] = common.computeDistanceClosing(self.id, self.vec[1:4],self.vec[1:4]-self.lastVec[1:4], 
#                        self.vec[4], other.id, other.vec[1:4], other.vec[1:4]-other.lastVec[1:4], other.vec[4])
#        vec[4:7], vec[7:10] = common.relPosVel(self.vec[1:4], self.vec[1:4]-self.lastVec[1:4], 
#                                            self.vec[4], other.vec[1:4], other.vec[1:4]-other.lastVec[1:4])
        vec[10] = np.dot(np.linalg.norm(vec[4:7]), np.linalg.norm(vec[7:10]))
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
#        print "object before: ", resO.vec[1:4]
#        print "relVec: ", self.getRelVec(other)
        pred = predictor.predict(self.getRelVec(other))
#        print "prediction for o: {}: {}".format(self.id, pred)
        self.predVec = self.vec + pred
        resO.vec 
        resO.vec = np.round(self.predVec, common.NUMDEC)
#        print "resulting object: ", resO.vec[1:4]
        return resO
        
    def update(self, newO):
        self.lastVec = self.vec
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
#        res = copy.deepcopy(self)
        res = Actuator()
        res.id = self.id
        self.predVec = np.copy(self.vec)
#        res.vec[5:8] = action #Set velocity
        self.predVec[5:8] = action
        #Hardcorded version
        if HARDCODEDACTUATOR:
#            res.vec[1:4] += 0.01*action
            self.predVec[1:4] += 0.01*action
        else:
            #Only predict position
            p = self.predictor.predict(action)
            print "prediction actuator: ", p
            self.predVec[1:4] += p
#            res.vec[1:4] += p
        res.lastVec = np.copy(self.vec)
        res.vec = np.round(self.predVec, common.NUMDEC)
        return res
            
    def update(self, newAc, action):
        self.lastVec = self.vec
#        self.predictor = newAc.predictor
        if HARDCODEDACTUATOR:
            pass
        else:
            pdif = newAc.vec[1:4]-self.vec[1:4]
            self.predictor.train(Node(0, wIn=action, wOut=pdif))
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
                self.actuator.vec = np.copy(o.vec[:8])
                self.actuator.lastVec = np.copy(self.actuator.vec)
                self.actuator.intStates = ws.getInteractionStates("gripper")
            else:
                newO = Object()
                newO.id = o["id"][0]
                newO.vec = np.copy(o.vec[:8])
                newO.lastVec = np.copy(newO.vec)
                newO.intStates = ws.getInteractionStates(oN)
                self.objectStates[newO.id] = newO
                
    
class Classifier(object):
    
    def __init__(self):
#        self.clas = neighbors.KNeighborsClassifier(n_neighbors=2, weights='uniform')
        self.clas = svm.SVC()
        self.inputs = []
        self.targets = []
        
    def train(self, o1vec, avec, label):
        if HARDCODEDGATE:
            pass
        else:
            self.inputs.append(np.concatenate((o1vec,avec)))
            self.targets.append(label)
            if max(self.targets) > 0:
                self.clas.fit(self.inputs, self.targets)
    
    def test(self, ovec, avec):
#        print "closing: {}, dist: {}".format(ovec[3], ovec[2])
        if HARDCODEDGATE:
            if ovec[3] <= -100*ovec[2]:
                print "Change: closing: {}, dist: {}, relVel: {}".format(ovec[3], ovec[2], ovec[7:10])
                return 1
            else:
                if ovec[3] == 0 and np.linalg.norm(ovec[7:10]) < 0.01 and ovec[2] < 0.05: #Todo remove distance from this
                    print "Change: closing: {}, dist: {}, relVel: {}".format(ovec[3], ovec[2], ovec[7:10])
                    return 1    
                else:
                    print "no Change: closing: {}, dist: {}, relVel: {}".format(ovec[3], ovec[2], ovec[7:10])
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
        #TODO convert dif to local coordinate frame! 
        #Since ITM input vector is relative to object, the output should be relative as well
#        print "dif: ", dif[1:4]
        if np.linalg.norm(dif[1:4]) > 0.0 or abs(dif[4]) > 0.0:
#            print "Change"
            return True, dif
#        print "No change"
        return False, dif
        
        
    def update(self, o1Pre, o1Post, o2Pre, action):
        """
        TODO Consider removing action here, since o2Pre has already experienced
        the results of action.
        Parameters
        ----------
        o1Pre: Object
        o1Post: Object
        o2Pre: Object
        action: np.ndarray
        """
        #TODO Causal determination, make hypothesis and test these!
#        vec = o2.getRelativeVec(o1Pre)
        
        vec = o1Pre.getRelVec(o2Pre)
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
            
    def getAction(self, targetId, dif):
        if targetId in self.predictors:
            return self.predictors[targetId].getAction(dif)
    
    def update(self, intState, action, dif):
        if not intState[0] in self.predictors:
            #TODO check for close ones that can be used
            self.predictors[intState[0]] = ITM()
#        print "updating with dif: ", dif[3]
        if max(dif[1:4]) >0.1 or min(dif[1:4]) < -0.1:
            print "dif to big: ", dif
            raise AttributeError
#        if np.linalg.norm(dif[1:5]) < 0.002:
#            print "dif to small", dif
#            raise AttributeError
        self.predictors[intState[0]].train(Node(0, wIn = intState, wOut=dif))


class ModelAction(object):
    
    def __init__(self):
        self.gate = GateFunction()
        self.actuator = None
        self.predictor = Predictor()
        self.curObjects = {}
        self.target = None
        
        
    def setTarget(self, target):
        """
            Sets a target that is to be reached.
            Target is an object (maybe partially described)
            Parameters
            ----------
            target : Object
        """
        self.target = target
        
    def getAction(self):
        """
            Returns an action, that is to be performed, trying to get closer to the
            target if one is set.
            
            Returns: np.ndarray
                Action vector for the actuator
        """
        if self.target is None:
            return np.array([0.0,0.0,0.0])
        else:
            if self.isTargetReached():
                return np.array([0.0,0.0,0.0])
            else:
                # Determine Actuator position that allows action in good direction
                action, pre, out = self.predictor.getAction(target.id, target.vec-self.curObjects[target.id].vec)
                # Bring actuator into position so that it influences targetobject
                # Select action to move target object towards target
                # Work in open loop: Compare result of previous action with expected result
                pass
            
        pass
        
        
    def predict(self, ws, action):
#        if self.actuator == None:
#            self.actuator = ws.actuator
        newWS = WorldState()
        newWS.actuator = self.actuator.predict(action)
        for o in ws.objectStates.values():
            print "object lastvec: ", len(o.lastVec)
            if self.gate.test(o, newWS.actuator, action): #TODO Check of newWS.actuator is correct here and if action can be removed!
#                print "predicted change"
                newO = self.predictor.predict(o, newWS.actuator, action)
                newWS.objectStates[o.id] = newO
            else:
#                print "predicted no change"
                o.vec[5:] = 0.0
                newWS.objectStates[o.id] = o
#            print "old object: {}, new object: {}".format(o.vec[1:4], newWS.objectStates[o.id].vec[1:4])
        return newWS
        
    def resetObjects(self, curWS):
        for o in curWS.objectStates.values():
            if o.id in self.curObjects:
                self.curObjects[o.id].update(o)
            else:
                self.curObjects[o.id] = o
                
        if self.actuator == None:
            self.actuator = curWS.actuator
        self.actuator.update(curWS.actuator, np.array([0.0,0.0,0.0]))
        
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
            
    