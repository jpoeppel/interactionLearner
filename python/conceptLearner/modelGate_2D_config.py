#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:27:54 2015

Reworked gate model to only use 2D

Action model that separates actuators from simple objects.
All object changes can only be induced by actuators!

For now (12.8) assume, that object's static properties can only be changed directly
by actuators (no sliding etc after contact is broken)

Idea for multiple objects: For gate function: Start from actuator and only consider 
objects that are changed to have a potential influence on other objects.


Config conform
@author: jpoeppel
"""

import numpy as np
from numpy import round as npround


import common
#from topoMaps import ITM
#from network import Node
from itm import ITM
import copy

from configuration import config
from inverseModel import MetaNetwork




class Object(object):
    
    def __init__(self):
        self.id = 0
        self.vec = np.array([])
        self.lastVec = np.array([])
        self.predVec= np.array([])
        
        
    def getRelVec(self, other):
        """
            Computes the relative (interaction) vector of the other object with respect
            to the own reference frame.
        """
        vec = np.zeros(8)
        vec[0] = self.id
        vec[1] = other.id
        #replaced computeDistanceClosing with generalDistClosing
        if config.USE_DYNS:
            vec[2], vec[3] = common.generalDistClosing(self.id, self.vec[0:2],self.vec[3:5], 
                        self.vec[2], other.id, other.vec[0:2], other.vec[3:5], other.vec[2])
            vec[4:6], vec[6:8], vec[8:10] = common.relPosVel(self.vec[0:3], self.vec[4:7], self.vec[3], other.vec[0:3], other.vec[4:7])
        else:
            vec[2], vec[3] = common.generalDistClosing(self.id, self.vec[0:2],self.vec[0:2]-self.lastVec[0:2], 
                        self.vec[2], other.id, other.vec[0:2], other.vec[0:2]-other.lastVec[0:2], other.vec[2])
#            vec[4:6], vec[6:8], vec[8:10] = common.relPosVel(self.vec[0:2], self.vec[0:2]-self.lastVec[0:2], self.vec[2], other.vec[0:2], other.vec[0:2]-other.lastVec[0:2])  
            vec[4:6], tmp, vec[6:8] = common.relPosVel(self.vec[0:2], self.vec[0:2]-self.lastVec[0:2], self.vec[2], other.vec[0:2], other.vec[0:2]-other.lastVec[0:2])                    
        return vec
        
    def getRelObjectVec(self, other):
        vec = np.zeros(len(self.vec))
        vec[0:2] = common.relPos(self.vec[0:2], self.vec[2], other.vec[0:2])
        vec[2] = other.vec[2]-self.vec[2]
        return vec
        
    def getGlobalPosVel(self, localPos, localVel):
        return common.globalPosVel(self.vec[0:2], self.vec[2], localPos, localVel)
                
    def getLocalChangeVec(self, post):
        res = np.copy(post.vec)
        res -= self.vec
        if config.USE_DYNS:
            res[0:2], res[3:5] = common.relPosVelChange(self.vec[2], res[0:2], res[3:5])
        else:
            res[0:2], v =  common.relPosVelChange(self.vec[2], res[0:2], np.zeros(2))
        return res

    def predict(self, predictor, other):
        resO = copy.deepcopy(self)
        pred = predictor.test(self.getRelVec(other), testMode=config.predictorTestMode)
#        pred = predictor.test(self.getRelVec(other))
        if config.USE_DYNS:
            pred[0:2], pred[3:5] = common.globalPosVelChange(self.vec[2], pred[0:2], pred[3:5])
        else:
            pred[0:2], v = common.globalPosVelChange(self.vec[2], pred[0:2], np.zeros(2))
#        print "prediction for o: {}: {}".format(self.id, pred)
        self.predVec = self.vec + pred*config.predictionBoost 
        resO.vec = self.predVec #np.round(self.predVec, config.NUMDEC)
        resO.lastVec = np.copy(self.vec)
        return resO
        
        
    def update(self, newO):
        self.lastVec = np.copy(self.vec)
        self.vec = np.copy(newO.vec)
        
    def circle(self, otherObject, direction = None):
        """
            Function to return an action that would circle the other object around 
            itself.
            
            Parameters
            ----------
            otherObject: Object
            
            returns: np.ndarray
                Action vector for the actuator for the next step
        """
        
        relVec = self.getRelVec(otherObject)
        dist = relVec[2]
        posDif = otherObject.vec[0:2] - self.vec[0:2]
        if dist < 0.04:
            return 0.4*posDif/np.linalg.norm(posDif)
        elif dist > 0.06:
            return -0.4*posDif/np.linalg.norm(posDif)
        else:
            tangent = np.array([-posDif[1], posDif[0]])
            if direction != None:
                angAct = np.arctan2(relVec[5],relVec[4])
                angTarget = np.arctan2(direction[1],direction[0])
                angDif = angTarget+np.pi - (angAct+np.pi)
                print "angDif: ", angDif
                if angDif > 0:
                    if abs(angDif) < np.pi:
                        print "circling pos"
                        return 0.4*tangent/np.linalg.norm(tangent)
                    else:
                        print "circling neg"
                        return -0.4*tangent/np.linalg.norm(tangent)
                else:
                    if abs(angDif) < np.pi:
                        print "circling neg"
                        return -0.4*tangent/np.linalg.norm(tangent)
                    else:
                        print "circling pos"
                        return 0.4*tangent/np.linalg.norm(tangent)
            else:
                return 0.4*tangent/np.linalg.norm(tangent)
        
        return np.array([0.0,0.0])
        
    def __repr__(self):
        return "{}".format(self.id)
        
    def getKeyPoints(self):
        WIDTH = {27: 0.25, 15: 0.25, 8: 0.025} #Width from the middle point
        DEPTH = {27: 0.25, 15: 0.05, 8: 0.025} #Height from the middle point
        p1x = WIDTH[self.id]
        p2x = -p1x
        p1y = DEPTH[self.id]
        p2y = -p1y
        ang = self.vec[2]
        c = np.cos(ang)
        s = np.sin(ang)
        p1xn = p1x*c -p1y*s + self.vec[0]
        p1yn = p1x*s + p1y*c + self.vec[1]
        p2xn = p2x*c - p2y*s + self.vec[0]
        p2yn = p2x*s + p2y*c + self.vec[1]
        return np.array([np.copy(self.vec[0:2]), np.array([p1xn,p1yn]), np.array([p2xn,p2yn])])
        
        
    @classmethod
    def parse(cls, m):
        res = cls()
        res.id = m.id 
        if config.USE_DYNS:
            res.vec = np.zeros(5)
#            res.vec[0] = m.id
            res.vec[0] = npround(m.pose.position.x, config.NUMDEC) #posX
            res.vec[1] = npround(m.pose.position.y, config.NUMDEC) #posY
            res.vec[2] = npround(common.quaternionToEuler(np.array([m.pose.orientation.x,m.pose.orientation.y,
                                                m.pose.orientation.z,m.pose.orientation.w])), config.NUMDEC)[2] #ori
            res.vec[3] = npround(m.linVel.x, config.NUMDEC) #linVelX
            res.vec[4] = npround(m.linVel.y, config.NUMDEC) #linVelY
        
#            res.vec[8] = npround(m.angVel.z, NUMDEC) #angVel
        else:
            res.vec = np.zeros(3)
#            res.vec[0] = m.id
            res.vec[0] = npround(m.pose.position.x, config.NUMDEC) #posX
            res.vec[1] = npround(m.pose.position.y, config.NUMDEC) #posY
            res.vec[2] = npround(common.quaternionToEuler(np.array([m.pose.orientation.x,m.pose.orientation.y,
                                                m.pose.orientation.z,m.pose.orientation.w])), config.NUMDEC)[2] #ori
        res.lastVec = np.copy(res.vec)
        return res

class Actuator(Object):
    
    def __init__(self):
        Object.__init__(self)
        self.predictor = ITM()
        if config.USE_DYNS:
            self.vec = np.zeros(5)
        else:
            self.vec = np.zeros(3)
        pass
    
    def predict(self, action):
#        res = copy.deepcopy(self)
        res = Actuator()
        res.id = self.id
        self.predVec = np.copy(self.vec)
        #Hardcorded version
        if config.HARDCODEDACTUATOR:
#            res.vec[1:4] += 0.01*action
            self.predVec[0:2] += 0.01*action
        else:
            #Only predict position
            p = self.predictor.test(action, testMode=config.actuatorTestMode)
            print "predicting actuator change: ", p
            self.predVec += p
#            res.vec[1:4] += p
        res.lastVec = np.copy(self.vec)
        res.vec = np.round(self.predVec, config.NUMDEC)
        res.predictor = self.predictor
        return res
            
    def update(self, newAc, action, training = True):
        self.lastVec = self.vec
#        self.predictor = newAc.predictor
        if training:
            if config.HARDCODEDACTUATOR:
                pass
            else:
                pdif = newAc.vec-self.vec
#                self.predictor.train(Node(0, wIn=action, wOut=pdif))
                self.predictor.update(action, pdif, 
                                      etaIn= config.actuatorEtaIn, 
                                      etaOut=config.actuatorEtaOut, 
                                      etaA=config.actuatorEtaA, 
                                      testMode=config.actuatorTestMode)
        self.vec = np.copy(newAc.vec)
        
    @classmethod
    def parse(cls, protoModel):
        res = super(Actuator, cls).parse(protoModel)
        res.vec[2] = 0 # Fix orientation
#        res.vec[8] = 0.0 #Fix angular velocity
        return res
    
class WorldState(object):
    
    def __init__(self):
        self.actuator = Actuator()
        self.objectStates = {}
        
    def parseModels(self, models):
        for m in models:
#            print "name: ", m.name
#            print "id: ", m.id
            if m.name == "ground_plane" or "wall" in m.name or "Shadow" in m.name:
                continue
            else:
                if m.name == "gripper":
                    ac = Actuator.parse(m)
                    self.actuator = ac
                else:
                    tmp = Object.parse(m)               
                    self.objectStates[tmp.id] = tmp
        
    def parse(self, gzWS):
        self.parseModels(gzWS.model_v.models)                
    
class Classifier(object):
    
    def __init__(self):
        self.clas = ITM()
        self.isTrained = False
        
    def train(self, o1vec, avec, label): #Consider removing avec here since it is not used, action already in the o1vec
        if config.HARDCODEDGATE:
            pass
        else:
#            print "training gate with: ", o1vec[config.gateMask]
            self.clas.update(o1vec[config.gateMask], np.array([label]), 
                             etaIn=config.gateClassifierEtaIn, 
                             etaOut=config.gateClassifierEtaOut, 
                             etaA=config.gateClassifierEtaA, 
                             testMode=config.gateClassifierTestMode)
            self.isTrained = True
    
    def test(self, ovec, avec):
        
        if config.HARDCODEDGATE:
            if ovec[3] <= -ovec[2]:
#                print "Change: closing: {}, dist: {}, relVel: {}".format(ovec[3], ovec[2], ovec[6:8])
                return 1
            else:
                if ovec[3] == 0 and np.linalg.norm(ovec[6:8]) < 0.001 and ovec[2] < 0.05: #Todo remove distance from this
#                    print "Change: closing: {}, dist: {}, relVel: {}".format(ovec[3], ovec[2], ovec[6:8])
                    return 1    
                else:
#                    print "no Change: closing: {}, dist: {}, relVel: {}".format(ovec[3], ovec[2], ovec[6:8])
                    return 0
        else:
#            print "testing with gate itm"
            if self.isTrained:
#                print "Gate number of nodes: ", len(self.clas.nodes)
                pred = int(self.clas.test(ovec[config.gateMask], testMode=config.gateClassifierTestMode)[0])
#                print "gate prediction: ", pred
                return pred
            else:
                return 0
    
    
class GateFunction(object):
    
    def __init__(self):
        self.classifier = Classifier()
        self.classifier2 = {}
        
        pass
    
    def test(self, o1, o2, action):
        vec = o1.getRelVec(o2)
        if o1.id in self.classifier2:
            return self.classifier2[o1.id].test(vec,action)
        else:
            return 0
#        print "testing gate with relVec: ", vec[config.gateMask]
#        return self.classifier.test(vec, action)
        
    def checkChange(self, pre, post):
        dif = pre.getLocalChangeVec(post)
        if np.linalg.norm(dif) > 0.0:    
            return True, dif
        return False, dif
        
        
    def update(self, o1Pre, o1Post, o2Post, action):
        """
        Parameters
        ----------
        o1Pre: Object
        o1Post: Object
        o2Post: Object
        action: np.ndarray
        """
        #TODO For multiple objects: Causal determination, make hypothesis and test these!
        vec = o1Pre.getRelVec(o2Post)
        hasChanged, dif = self.checkChange(o1Pre, o1Post)

        if not o1Pre.id in self.classifier2:
            self.classifier2[o1Pre.id] = Classifier()
        self.classifier2[o1Pre.id].train(vec, action, int(hasChanged))
        
#        self.classifier.train(vec, action, int(hasChanged))
        if hasChanged:
            return True, dif
        else:
            return False, dif

        
            
class Predictor(object):
    
    def __init__(self):
        self.predictors = {}
        self.inverseModel = {}
    
    def predict(self, o1, o2, action):
        if o1.id in self.predictors:
            print "num nodes in local itm: ", len(self.predictors[o1.id].nodes)
            return o1.predict(self.predictors[o1.id], o2)
        else:
            return o1
            
    def getAction(self,targetId, dif):
        if targetId in self.inverseModel:
            return self.inverseModel[targetId].getPreconditions(dif)
        else:
            print "target not found"
            return None
            
    def getExplorationPreconditions(self, objectId):
        if objectId in self.inverseModel:
            return self.inverseModel[objectId].tobeNamed()
        else:
            print "No inverse model for objectId {}".format(objectId)
    
    def update(self, intState, action, dif): #TODO: consider removing action here since it is not used
        if not intState[0] in self.predictors:
            #TODO check for close ones that can be used
            self.predictors[intState[0]] = ITM()
            self.inverseModel[intState[0]] = MetaNetwork()
        if np.linalg.norm(dif) == 0.0:
            print "training with zero dif: ", dif
            raise NotImplementedError
        self.predictors[intState[0]].update(intState, dif, 
                                            etaIn= config.predictorEtaIn, 
                                            etaOut= config.predictorEtaOut, 
                                            etaA= config.predictorEtaA, 
                                            testMode=config.predictorTestMode)
        self.inverseModel[intState[0]].train(intState, dif)


class ModelGate(object):
    
    def __init__(self):
        self.gate = GateFunction()
        self.actuator = None
        self.predictor = Predictor()
        self.curObjects = {}
        self.target = None
        self.training = True #Determines if the model should be trained on updates or
                            # just update it's objects features
        
    def getITMInformation(self):
        if config.HARDCODEDGATE:
            gateString = "Gate has been hardcoded.\n"
        else:
            gateString = "Gate ITM: UpdateCalls: {}, Insertions: {}, final Number of nodes: {}\n"\
                        .format(self.gate.classifier.clas.updateCalls,
                                self.gate.classifier.clas.inserts, 
                                len(self.gate.classifier.clas.nodes))
        if config.HARDCODEDACTUATOR:
            actString = "Actuator has been hardcoded.\n"
        else:
            actString = "Actuator ITM: UpdateCalls: {}, Insertions: {}, final Number of nodes: {}\n"\
                        .format(self.actuator.predictor.updateCalls,
                                self.actuator.predictor.inserts, 
                                len(self.actuator.predictor.nodes))
        predString = ""
        for oId, predItm in self.predictor.predictors.items():
            predString += "Object predictor ITM for object {}: UpdateCalls: {}, Insertions: {}, final Number of nodes: {}\n"\
                        .format(oId, predItm.updateCalls,
                                predItm.inserts, len(predItm.nodes))
        return actString + gateString + predString
        
        
    def setTarget(self, target):
        """
            Sets a target that is to be reached.
            Target is an object (maybe partially described)
            Parameters
            ----------
            target : Object
        """
        self.target = target
        
    def isTargetReached(self):
        targetO = self.curObjects[self.target.id]
#        print "targetO.vec: ", targetO.vec
        difVec = targetO.vec-self.target.vec
        norm = np.linalg.norm(difVec)
#        print "dif norm: ", norm
        if norm < 0.01:
            return True
        return False
        
    def explore(self):
        """
            Returns an action in order to increase the knowledge of the model.
            
            Returns: np.ndarray
                Action vector for the actuator
        """
        # Get promising pre conditions
        # Fullfilly pre conditions
        # Perform action from precondtions (idially if it triggers gate/find action that triggers gate)
        # (If successfull ->) get next set of preconditions for different attribute?
        for oId in self.curObjects.keys():
            preCons = self.predictor.getExplorationPreconditions(oId)
            
        if preCons == None:
            print "No features found that need to be improved"
            return np.zeros(2)
        else:
            targetO = self.curObjects[oId]
            relTargetPos = preCons[4:6]
            print "rel target pos: ", relTargetPos
            relTargetVel = preCons[8:10]
            print "relTargetVel: ", relTargetVel
            
            pos, vel = targetO.getGlobalPosVel(relTargetPos, relTargetVel)
            print "target vel: ", vel
            print "target pos: ", pos
            print "cur pos: ", self.actuator.vec[0:2]
            difPos = pos-self.actuator.vec[0:2]
            print "difpos norm: ", np.linalg.norm(difPos)
            relVec = targetO.getRelVec(self.actuator)
            
#                relVec = targetO.getRelObjectVec(self.actuator)
            relPos = relVec[4:6]
            # Determine Actuator position that allows action in good direction
            wrongSides = relPos*relTargetPos < 0
            if np.any(wrongSides):
                if max(abs(relTargetPos[wrongSides]-relPos[wrongSides])) > 0.05:
                 # Bring actuator into position so that it influences targetobject
                    print "try circlying"
                    return targetO.circle(self.actuator)
                    
            if np.linalg.norm(difPos) > 0.01:
                action = 0.5*difPos/np.linalg.norm(difPos)
                print "difpos action: ", action
                tmpAc = self.actuator.predict(action)
                return action

            print "using vel"
            normVel = np.linalg.norm(vel)
            if normVel == 0.0:
                pass
            else:
                vel = 0.5*vel/normVel
            tmpAc = self.actuator.predict(vel)
            if not self.gate.test(targetO, tmpAc, vel):
                print "looking for different vel"
                for i in xrange(len(vel)):
                    tmpVel = relTargetVel
                    tmpVel[i] *= -1
                    pos, tmpVel = targetO.getGlobalPosVel(relTargetPos, relTargetVel)
                    normVel = np.linalg.norm(tmpVel)
                    if normVel == 0.0:
                        pass
                    else:
                        tmpVel = 0.5*tmpVel/normVel
                    tmpAc = self.actuator.predict(tmpVel)
                    
                    if self.gate.test(targetO, tmpAc, tmpVel):
                        vel = tmpVel
                        print "found new vel: ", vel
                        break
            return vel
                

        
    def getAction(self):
        """
            Returns an action, that is to be performed, trying to get closer to the
            target if one is set.
            
            Returns: np.ndarray
                Action vector for the actuator
        """
        if self.target is None:
#            return self.explore()
            return np.array([0.0,0.0])
        else:
            if self.isTargetReached():
                print "target reached"
                self.target = None
                return np.array([0.0,0.0])
            else:
                targetO = self.curObjects[self.target.id]
                # Determine difference vector, the object should follow
#                print "global dif vec: ", self.target.vec-targetO.vec
                difVec = targetO.getLocalChangeVec(self.target)
#                difNorm = np.linalg.norm(difVec)
#                print "difVec: ", difVec[:5]
                pre = self.predictor.getAction(self.target.id, difVec)
                if pre == None:
                    return self.explore()
                relTargetPos = pre[4:6]
#                print "rel target pos: ", relTargetPos
                relTargetVel = pre[6:8]
#                print "relTargetVel: ", relTargetVel
                
                pos, vel = targetO.getGlobalPosVel(relTargetPos, relTargetVel)
#                print "target vel: ", vel
#                print "target pos: ", pos
#                print "cur pos: ", self.actuator.vec[0:2]
                difPos = pos-self.actuator.vec[0:2]
#                print "difpos norm: ", np.linalg.norm(difPos)
                relVec = targetO.getRelVec(self.actuator)
                
#                relVec = targetO.getRelObjectVec(self.actuator)
                relPos = relVec[4:6]
#                print "relPos: ", relPos
                # Determine Actuator position that allows action in good direction
#                wrongSides = relPos*relTargetPos < 0
#                if np.any(wrongSides):
#                    if max(abs(relTargetPos[wrongSides]-relPos[wrongSides])) > 0.05:
#                     # Bring actuator into position so that it influences targetobject
#                        print "try circlying"
#                        return targetO.circle(self.actuator, 2*relTargetPos-relPos)
                if np.linalg.norm(difPos) > 0.1:
#                    print "circling, too far"
                    return targetO.circle(self.actuator, relTargetPos)
                if np.linalg.norm(difPos) > 0.01:
                    difPosAction = 0.3*difPos/np.linalg.norm(difPos)
#                    print "difpos action: ", difPosAction
                    tmpAc = self.actuator.predict(difPosAction)
                    if not self.gate.test(targetO, tmpAc, difPosAction):
#                        print "doing difpos"
                        return difPosAction
                    else:
                        predRes = self.predictor.predict(targetO, tmpAc, difPosAction)
                        if np.linalg.norm(predRes.vec-self.target.vec) > np.linalg.norm(targetO.vec-self.target.vec):
#                            print "circlying since can't do difpos"
                            return targetO.circle(self.actuator, relTargetPos)
                        else:
#                            print "doing difpos anyways"
                            return difPosAction

#                print "using vel"
                normVel = np.linalg.norm(vel)
                if normVel == 0.0 or (normVel > 0.01 and normVel < 0.2):
                    return vel
                else:
                    return 0.3*vel/normVel
#                
#                #TODO!
#                # Work in open loop: Compare result of previous action with expected result
#                pass
            
        pass
    
    def predict(self, ws, action):
        #TODO Remove ws from here since it is not needed at all. Not true if online learning is tested in new prediction task
        newWS = WorldState()
#        newWS.actuator = self.actuator.predict(action)
        #Necessary when using the newly parsed ws since otherwise a not learned itm is used
        if self.actuator != None:
            ws.actuator.predictor = self.actuator.predictor 
        #Alternatively do this, arguably nicer
#        newWS.actuator = self.actuator.predict(action)
        newWS.actuator = ws.actuator.predict(action)
        for o in ws.objectStates.values():
#        for o in self.curObjects.values():
            if self.gate.test(o, newWS.actuator, action):
                newO = self.predictor.predict(o, newWS.actuator, action)
                newWS.objectStates[o.id] = newO
                
            else:
                if config.USE_DYNS:
                    #Stop dynamics
                    o.vec[3:] = 0.0
                    
                newWS.objectStates[o.id] = o
        return newWS
        
        
    def resetObjects(self, curWS):
        for o in curWS.objectStates.values():
            if o.id in self.curObjects:
                self.curObjects[o.id].update(o)
            else:
                self.curObjects[o.id] = o
                
        if self.actuator == None:
            self.actuator = curWS.actuator
        self.actuator.update(curWS.actuator, np.array([0.0,0.0]), training = False)
        
    def update(self, curWS, action):
        if self.actuator == None:
            self.actuator = curWS.actuator
        self.actuator.update(curWS.actuator, action, self.training)
        for o in curWS.objectStates.values():
            #TODO extent to more objects
            if o.id in self.curObjects:
#                hasChanged, dif = self.gate.update(self.curObjects[o.id], o, self.actuator, action)
                if self.training:
                    hasChanged, dif = self.gate.update(self.curObjects[o.id], o, self.actuator, action)
                    if hasChanged:
                        self.predictor.update(self.curObjects[o.id].getRelVec(self.actuator), action, dif)
                self.curObjects[o.id].update(curWS.objectStates[o.id])
            else:
                self.curObjects[o.id] = o
        
        
            
    