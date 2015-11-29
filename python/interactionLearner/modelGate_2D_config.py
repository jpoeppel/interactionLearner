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
import copy

import common
from aitm import AITM
from configuration import config
from inverseModel import MetaNetwork




class Object(object):
    """
        Class representing an object in the environment.
    """
    
    def __init__(self):
        self.id = 0
        self.vec = np.array([])
        self.lastVec = np.array([])
        self.predVec= np.array([])
        
        
    def getRelVec(self, other):
        """
            Computes the relative (interaction) vector of the other object with respect
            to the own reference frame.
            
            Parameters
            ----------
            other : Object
                The secondary object with which the relative interaction vector is to be computed
                
            Returns
            -------
                np.ndarray(8)
                Relative interaction vector with the current object as reference
        """
        vec = np.zeros(8)
        vec[0] = self.id
        vec[1] = other.id
        if config.USE_DYNS:
            vec[2], vec[3] = common.generalDistClosing(self.id, self.vec[0:2],self.vec[3:5], 
                        self.vec[2], other.id, other.vec[0:2], other.vec[3:5], other.vec[2])
            vec[4:6], vec[6:8], vec[8:10] = common.relPosVel(self.vec[0:3], self.vec[4:7], 
                                                    self.vec[3], other.vec[0:3], other.vec[4:7])
        else:
            vec[2], vec[3] = common.generalDistClosing(self.id, self.vec[0:2],
                                self.vec[0:2]-self.lastVec[0:2], self.vec[2], other.id, 
                                other.vec[0:2], other.vec[0:2]-other.lastVec[0:2], other.vec[2])
            vec[4:6], tmp, vec[6:8] = common.relPosVel(self.vec[0:2], 
                                        self.vec[0:2]-self.lastVec[0:2], 
                                        self.vec[2], other.vec[0:2], 
                                        other.vec[0:2]-other.lastVec[0:2])                    
        return vec
        
    def getRelObjectVec(self, other):
        """
            Transforms the vector of the other object so that is represented with respect to the
            current object's coordinate frame
            
            Parameters
            ---------
            other : Object
                The object whose vector is to be transformed
                
            Returns
            -------
                np.ndarray(3)
                Transformed object vector of the other object
        """
        vec = np.zeros(len(self.vec))
        vec[0:2] = common.relPos(self.vec[0:2], self.vec[2], other.vec[0:2])
        vec[2] = other.vec[2]-self.vec[2]
        return vec
        
    def getGlobalPosVel(self, localPos, localVel):
        """
            Computes the global position and velocity based on a local position and velocity
            that are given with respect to the current object's reference frame
            
            Parameters
            ----------
            localPos : np.ndarray(2)
                Position vector in local coordinates
            localVel: np.ndarra(2)
                Velocity vector in local coordinates
                
            Returns
            -------
            globPos: np.array(2)/np.array(3)
                Global position 
            globVel: np.array(2)/np.array(3)
                Global velocity
        """
        return common.globalPosVel(self.vec[0:2], self.vec[2], localPos, localVel)
                
    def getLocalChangeVec(self, post):
        """
            Computes the change vector relative to the current object's reference frame
            
            Parameters
            ---------
            post : Object
                The changed object
                
            Returns
            -------
                np.ndarray(3)
                Change vector in local coordinates
        """
        res = np.copy(post.vec)
        res -= self.vec
        if config.USE_DYNS:
            res[0:2], res[3:5] = common.relPosVelChange(self.vec[2], res[0:2], res[3:5])
        else:
            res[0:2], v =  common.relPosVelChange(self.vec[2], res[0:2], np.zeros(2))
        return res

    def predict(self, predictor, other):
        """
            Uses the provided predictor in order to make predictions about the next state after
            an interaction with the other object.
            
            Parameters
            ----------
            predictor : AITM
                Trained predictor for the current object
            other : Object
                The object that is interacting with the current object
            
            Returns
            -------
                Object
                Predicted object
        """
        resO = copy.deepcopy(self)
        pred = predictor.test(self.getRelVec(other), testMode=config.predictorTestMode)
        if config.USE_DYNS:
            pred[0:2], pred[3:5] = common.globalPosVelChange(self.vec[2], pred[0:2], pred[3:5])
        else:
            pred[0:2], v = common.globalPosVelChange(self.vec[2], pred[0:2], np.zeros(2))
        self.predVec = self.vec + pred*config.predictionBoost 
        resO.vec = self.predVec
        resO.lastVec = np.copy(self.vec)
        return resO
        
        
    def update(self, newO):
        """
            Updates the current object to the new object state
            
            Parameters
            ----------
            newO : Object
                Object containing the new state of the current object
        """
        self.lastVec = np.copy(self.vec)
        self.vec = np.copy(newO.vec)
        
    def circle(self, otherObject, direction = None):
        """
            Function to return an action that would circle the other object around 
            itself.
            
            Parameters
            ----------
            otherObject: Object
                Influencing object. Usually the actuator
            direction : np.ndarray, optional
                Relative direction of the object towards the target position
            
            Returns
            -------
                np.ndarray
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
                if angDif > 0:
                    if abs(angDif) < np.pi:
                        return 0.4*tangent/np.linalg.norm(tangent)
                    else:
                        return -0.4*tangent/np.linalg.norm(tangent)
                else:
                    if abs(angDif) < np.pi:
                        return -0.4*tangent/np.linalg.norm(tangent)
                    else:
                        return 0.4*tangent/np.linalg.norm(tangent)
            else:
                return 0.4*tangent/np.linalg.norm(tangent)
        return np.array([0.0,0.0])
        
    def __repr__(self):
        """
            Provides string representation of the object. Currently only the identifer is returned.
            
            Returns
            -------
                String
                String representation of the current object
        """
        return "{}".format(self.id)
        
    def getKeyPoints(self):
        """
            Computes the position of two edges of the object.
            
            Returns
            -------
                np.ndarray(3x2)
                Matrix containing the 2d positions of the center as well as the two keypoints.
        """
        WIDTH = {27: 0.15, 15: 0.25, 8: 0.025} #Width from the middle point
        DEPTH = {27: 0.15, 15: 0.05, 8: 0.025} #Height from the middle point
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
        """
            Classmethod that parses a protobuf modelstate to get an Object
            
            Paramters
            ---------
            m : Protobuf.ModelState
            
            Returns
            -------
                Object
                The parsed object
        """
        res = cls()
        res.id = m.id 
        if config.USE_DYNS:
            res.vec = np.zeros(5)
            res.vec[0] = npround(m.pose.position.x, config.NUMDEC) #posX
            res.vec[1] = npround(m.pose.position.y, config.NUMDEC) #posY
            res.vec[2] = npround(common.quaternionToEuler(np.array([m.pose.orientation.x,
                                    m.pose.orientation.y, m.pose.orientation.z,
                                    m.pose.orientation.w])), config.NUMDEC)[2] #ori
            res.vec[3] = npround(m.linVel.x, config.NUMDEC) #linVelX
            res.vec[4] = npround(m.linVel.y, config.NUMDEC) #linVelY
        else:
            res.vec = np.zeros(3)
            res.vec[0] = npround(m.pose.position.x, config.NUMDEC) #posX
            res.vec[1] = npround(m.pose.position.y, config.NUMDEC) #posY
            res.vec[2] = npround(common.quaternionToEuler(np.array([m.pose.orientation.x,
                                    m.pose.orientation.y, m.pose.orientation.z,
                                    m.pose.orientation.w])), config.NUMDEC)[2] #ori
        res.lastVec = np.copy(res.vec)
        return res

class Actuator(Object):
    """
        Dedicated representation of the actuator as special kind of object.
    """
    
    def __init__(self):
        Object.__init__(self)
        self.predictor = AITM()
        if config.USE_DYNS:
            self.vec = np.zeros(5)
        else:
            self.vec = np.zeros(3)
        pass
    
    def predict(self, action):
        """
            Predict the next actuator state given an action primitive.
            
            Parameters
            ----------
            action : np.ndarray(2)
                Action primitive used
            
            Returns
            -------
                Actuator
                Predicted actuator representation.
        """
        res = Actuator()
        res.id = self.id
        self.predVec = np.copy(self.vec)
        #Hardcorded version
        if config.HARDCODEDACTUATOR:
            self.predVec[0:2] += 0.01*action
        else:
            #Only predict position
            p = self.predictor.test(action, testMode=config.actuatorTestMode)
            self.predVec += p
        res.lastVec = np.copy(self.vec)
        res.vec = np.round(self.predVec, config.NUMDEC)
        res.predictor = self.predictor
        return res
            
    def update(self, newAc, action, training = True):
        """
            Updates the actuator. Always updates the current state of the actuator. If training
            is given, also updates the local forward model.
            
            Parameters
            ----------
            newAc : Actuator
                New actuator state whose information is taken
            action : np.ndarray(2)
                Action primitive used to reach the current state
            training : bool
                Updates the local forward model if true, otherwise only the state is updated
        """
        self.lastVec = self.vec
        if training:
            if config.HARDCODEDACTUATOR:
                pass
            else:
                pdif = newAc.vec-self.vec
                self.predictor.update(action, pdif, 
                                      etaIn= config.actuatorEtaIn, 
                                      etaOut=config.actuatorEtaOut, 
                                      etaA=config.actuatorEtaA, 
                                      testMode=config.actuatorTestMode)
        self.vec = np.copy(newAc.vec)
        
    @classmethod
    def parse(cls, protoModel):
        """
            Classmethod to parse an actuator from a protobuf ModelState
            
            Parameters
            ----------
            protoModel : Protobuf.ModelState
                Protobuf message containing the object's information
                
            Returns
            -------
                Actuator
                Extracted actuator object.
        """
        res = super(Actuator, cls).parse(protoModel)
        res.vec[2] = 0 # Fix orientation
        return res
    
class WorldState(object):
    """
        Container class for all the object states.
    """
    
    def __init__(self):
        self.actuator = Actuator()
        self.objectStates = {}
        
    def parseModels(self, models):
        """
            Parses a protobuf ModelState_V 
            
            Parameters
            ----------
            models : Protobuf.ModelState_V
                Vector message containing all the ModelStates 
        """
        for m in models:
            if m.name == "ground_plane" or "wall" in m.name or "Shadow" in m.name:
                continue
            else:
                if m.name == "actuator":
                    ac = Actuator.parse(m)
                    self.actuator = ac
                else:
                    tmp = Object.parse(m)               
                    self.objectStates[tmp.id] = tmp
        
    def parse(self, gzWS):
        """
            Parses a protobuf WorldState message
            
            Parameters
            ---------
            gzWS : Protobuf.WorldState
                Worldstate message provided by the simulation
        """
        self.parseModels(gzWS.model_v.models)                
    
class Classifier(object):
    """
        Wrapper class for the ATIM as classifier to allow hardcoded gating function.

    """
    
    def __init__(self):
        self.clas = AITM()
        self.isTrained = False
        
    def train(self, inVec, label): 
        """
            Trains the classifier. If hardcoded, no training is done. Otherwise the used AITM is
            updated.
            
            Parameters
            ----------
            inVec: np.ndarray
                Input vector
            label: int
                Desired output label for the input
                
        """
        if config.HARDCODEDGATE:
            pass
        else:
            self.clas.update(inVec[config.gateMask], np.array([label]), 
                             etaIn=config.gateClassifierEtaIn, 
                             etaOut=config.gateClassifierEtaOut, 
                             etaA=config.gateClassifierEtaA, 
                             testMode=config.gateClassifierTestMode)
            self.isTrained = True
    
    def test(self, inVec):
        """
            Tests the classifier with the input vector.
            
            Parameters
            ----------
            inVec : np.ndarray
                Input vector that is to be tested
            
            Returns
            -------
                int
                The predicted class label for the input
        """
        
        if config.HARDCODEDGATE:
            if inVec[3] <= -inVec[2]:
                return 1
            else:
                #Todo remove distance from this
                if inVec[3] == 0 and np.linalg.norm(inVec[6:8]) < 0.001 and inVec[2] < 0.05: 
                    return 1    
                else:
                    return 0
        else:
            if self.isTrained:
                pred = int(self.clas.test(inVec[config.gateMask], testMode=config.gateClassifierTestMode)[0])
                return pred
            else:
                return 0
    
    
class GateFunction(object):
    """
        Implementation of the gating function that checks for changes and tries to learn when
        an object influences another.
    """
    
    def __init__(self):
        self.classifier = Classifier()
        pass
    
    def test(self, o1, o2):
        """
            Queries to classifier if o2 influences o1.
            
            Parameters
            ----------
            o1 : Object
                Object that might be influenced
            o2 : Object
                The acting object that might influence
                
            Returns
            -------
                int (0,1)
                1 If an influence is predicted, 0 otherwise
        """
        vec = o1.getRelVec(o2)
        return self.classifier.test(vec)
        
    def checkChange(self, pre, post):
        """
            Computes and checks the local change vector in order to determine if the object changed
            in the last update step.
            
            Parameters
            ---------
            pre : Object
                Object state before the update
            post : Object
                Object state after the update
                
            Returns
            -------
            changed : bool
                True if the object changed, False otherwise
            dif : np.ndarray
                The local difference vector                            
        """
        dif = pre.getLocalChangeVec(post)
        if np.linalg.norm(dif) > 0.0:    
            return True, dif
        return False, dif
        
        
    def update(self, o1Pre, o1Post, o2Post):
        """
            Updates the gating function. Trains the classifier with the appropriate label depending
            on whether the object changed or not.
            Parameters
            ----------
            o1Pre: Object
                Object state before the update
            o1Post: Object
                Object state after the update
            o2Post: Object
                Object state of the potentially influencing object after the update
        """
        #TODO For multiple objects: Causal determination, make hypothesis and test these!
        vec = o1Pre.getRelVec(o2Post)
        hasChanged, dif = self.checkChange(o1Pre, o1Post)
        
        self.classifier.train(vec, int(hasChanged))
        if hasChanged:
            return True, dif
        else:
            return False, dif

        
            
class Predictor(object):
    """
        Implementation of the predictor. Contains local forward and inverse models for each
        object group.
    """
    
    def __init__(self):
        self.predictors = {}
        self.inverseModel = {}
    
    def predict(self, o1, o2):
        """
            Makes predictions about the next state of o1 given the actuating object o2
            
            Paramters
            ---------
            o1 : Object
                Object whose next state is to be predicted
            o2 : Object
                Influencing object
                
            Returns
                Object
                Predicted object state of o1
        """        
        
        if o1.id in self.predictors:
            return o1.predict(self.predictors[o1.id], o2)
        else:
            return o1
            
    def getAction(self,targetId, dif):
        """
            Queries to local inverse model for the object group specified by the targetId for
            preconditions.
            
            Parameters
            ----------
            targetId : int
                Identifier of the object group of the target object
            difs : np.ndarray
                The current difference vector
                
            Returns
            -------
            preconditions: np.ndarray
                Preconditions suited to reduce the distance if target known. None otherwise.
        """
        if targetId in self.inverseModel:
            return self.inverseModel[targetId].getPreconditions(dif)
        else:
            print "target not found"
            return None
            
    def getExplorationPreconditions(self, objectId):
        """
            EXPERIMENTAL
            Queries the inverse model responsible for the given objectId for preconditions that 
            increase the model's knowledge.
            
            Parameters
            ----------
            objectId : int
                Identifier of the object that is to explored.
                
            Returns
            -------
            preconditions : np.ndarray
                Preconditions that might increase the knowledge about the given object
        """
        if objectId in self.inverseModel:
            return self.inverseModel[objectId].getPreconsToTry()
        else:
            print "No inverse model for objectId {}".format(objectId)
    
    def update(self, objectId, intVec, dif):
        """
            Updates the local forward model responsible for the given objectId.
            
            Parameters
            ---------
            objectId : int
                Identifier of the object whose object group is to be updated
            intVec : np.ndarray
                Relative interaction features that influenced the object
            dif : np.ndarray
                Difference/Change vector of the object
        """
        if not objectId in self.predictors:
            #TODO check for close ones that can be used
            self.predictors[objectId] = AITM()
            self.inverseModel[objectId] = MetaNetwork()
            
        self.predictors[objectId].update(intVec, dif, 
                                            etaIn= config.predictorEtaIn, 
                                            etaOut= config.predictorEtaOut, 
                                            etaA= config.predictorEtaA, 
                                            testMode=config.predictorTestMode)
        self.inverseModel[objectId].train(intVec, dif)


class ModelGate(object):
    """
        Implementation of the object state with gating function model.
    """
    
    def __init__(self):
        self.gate = GateFunction()
        self.actuator = None
        self.predictor = Predictor()
        self.curObjects = {}
        self.target = None
        self.training = True #Determines if the model should be trained on updates or
                            # just update it's objects features
        
    def getITMInformation(self):
        """
            Helper function to record the state of the included AITMs.
            
            Returns
            -------
                String
                Information about the number of nodes and update calls of each AITM
        """
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
            actString = "Actuator ITM: UpdateCalls: {}, Insertions: {}, \
                        final Number of nodes: {}\n"\
                        .format(self.actuator.predictor.updateCalls,
                                self.actuator.predictor.inserts, 
                                len(self.actuator.predictor.nodes))
        predString = ""
        for oId, predItm in self.predictor.predictors.items():
            predString += "Object predictor ITM for object {}: UpdateCalls: {}, \
                            Insertions: {}, final Number of nodes: {}\n"\
                            .format(oId, predItm.updateCalls, predItm.inserts, len(predItm.nodes))
        return actString + gateString + predString
        
        
    def setTarget(self, target):
        """
            Sets a target that is to be reached.
            Target is an object
            
            Parameters
            ----------
            target : Object
                Object state that is to be reached.
        """
        self.target = target
        
    def isTargetReached(self):
        """
            Helper function to check if the current target has already been reached.
            
            Returns
            ------
                bool
                True if the target has been reached, False otherwise
        """
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
            EXPERIMENTAL!!
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
            relTargetVel = preCons[6:8]
            
            pos, vel = targetO.getGlobalPosVel(relTargetPos, relTargetVel)
            difPos = pos-self.actuator.vec[0:2]
            
            if np.linalg.norm(difPos) > 0.1:
                return targetO.circle(self.actuator, relTargetPos)
            if np.linalg.norm(difPos) > 0.01:
                difPosAction = 0.3*difPos/np.linalg.norm(difPos)
                tmpAc = self.actuator.predict(difPosAction)
                if not self.gate.test(targetO, tmpAc):
#                        print "doing difpos"
                    return difPosAction
                else:
                    predRes = self.predictor.predict(targetO, tmpAc)
                    if np.linalg.norm(predRes.vec-self.target.vec) > np.linalg.norm(targetO.vec-self.target.vec):
                        return targetO.circle(self.actuator, relTargetPos)
                    else:
                        return difPosAction

            normVel = np.linalg.norm(vel)
            if normVel == 0.0 or (normVel > 0.01 and normVel < 0.2):
                return vel
            else:
                return 0.3*vel/normVel
                

        
    def getAction(self):
        """
            Returns an action, that is to be performed, trying to get closer to the
            target if one is set. If no target is set, returns the 0 action.
            
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
                difVec = targetO.getLocalChangeVec(self.target)
                pre = self.predictor.getAction(self.target.id, difVec)
                if pre == None:
#                    return self.explore()
                    randAction = (targetO.vec[:2]-self.actuator.vec[:2]) + (np.random.rand(2)-0.5)
                    norm = np.linalg.norm(randAction)
                    if norm > 0:
                        return 0.3*randAction/norm
                    else:
                        return randAction
                relTargetPos = pre[4:6]
                relTargetVel = pre[6:8]
                pos, vel = targetO.getGlobalPosVel(relTargetPos, relTargetVel)
                difPos = pos-self.actuator.vec[0:2]
                
                if np.linalg.norm(difPos) > 0.1:
                    return targetO.circle(self.actuator, relTargetPos)
                if np.linalg.norm(difPos) > 0.01:
                    difPosAction = 0.3*difPos/np.linalg.norm(difPos)
                    tmpAc = self.actuator.predict(difPosAction)
                    if not self.gate.test(targetO, tmpAc):
                        return difPosAction
                    else:
                        predRes = self.predictor.predict(targetO, tmpAc)
                        if np.linalg.norm(predRes.vec-self.target.vec) > \
                                np.linalg.norm(targetO.vec-self.target.vec):
                            return targetO.circle(self.actuator, relTargetPos)
                        else:
                            return difPosAction

                normVel = np.linalg.norm(vel)
                if normVel == 0.0 or (normVel > 0.01 and normVel < 0.2):
                    return vel
                else:
                    return 0.3*vel/normVel
#                #TODO!
#                # Work in open loop: Compare result of previous action with expected result
#                pass
    
    def predict(self, ws, action):
        """
            Predicts the next worldstate given a current worldstate and an action primitive.
            
            Parameters
            ----------
            ws : WorldState
                Current worldstate
            action : np.ndarray(2)
                Action primitive that is to be used.
                
            Returns
            -------
                WorldState
                Worldstate containing the predicted states of all objects in ws
        """
        newWS = WorldState()
        #Necessary when using the newly parsed ws since otherwise a not learned itm is used
        if self.actuator != None:
            ws.actuator.predictor = self.actuator.predictor 
        #Alternatively do this, arguably nicer
#        newWS.actuator = self.actuator.predict(action)
        newWS.actuator = ws.actuator.predict(action)
        for o in ws.objectStates.values():
#        for o in self.curObjects.values():
            if self.gate.test(o, newWS.actuator):
                newO = self.predictor.predict(o, newWS.actuator)
                newWS.objectStates[o.id] = newO
                
            else:
                if config.USE_DYNS:
                    #Stop dynamics
                    o.vec[3:] = 0.0
                    
                newWS.objectStates[o.id] = o
        return newWS
        
        
    def resetObjects(self, curWS):
        """
            Updates the known object positions. Required for resetting a run, without loosing
            what is already learned.
            
            Parameters
            ---------
            curWS : WorldState
                Current worldstate whose object states represent the new values
        """
        for o in curWS.objectStates.values():
            if o.id in self.curObjects:
                self.curObjects[o.id].update(o)
            else:
                self.curObjects[o.id] = o
                
        if self.actuator == None:
            self.actuator = curWS.actuator
        self.actuator.update(curWS.actuator, np.array([0.0,0.0]), training = False)
        
    def update(self, curWS, action):
        """
            Updates the model. First updates the actuator, afterwards the gating function and
            any object predictor if necessary.
            
            Parameters
            ----------
            curWS : WorldState
                Current worldState which is used to update
            action : np.ndarray(2)
                Action primitive used to produce the current worldstate
        """
        if self.actuator == None:
            self.actuator = curWS.actuator
        self.actuator.update(curWS.actuator, action, self.training)
        for o in curWS.objectStates.values():
            #TODO extent to object-object interaction, i.e. use predicted objects as potential
            #new actuators
            if o.id in self.curObjects:
                if self.training:
                    hasChanged, dif = self.gate.update(self.curObjects[o.id], o, self.actuator)
                    if hasChanged:
                        relVec = self.curObjects[o.id].getRelVec(self.actuator)
                        self.predictor.update(o.id, relVec, dif)
                self.curObjects[o.id].update(curWS.objectStates[o.id])
            else:
                self.curObjects[o.id] = o
        
        
            
    