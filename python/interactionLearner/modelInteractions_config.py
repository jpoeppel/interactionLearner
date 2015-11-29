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
from numpy import copy as npcopy
from numpy import round as npround
import common
from aitm import AITM
import copy
from configuration import config
from inverseModel import MetaNetwork


class Object(object):
    """
        Class representing an object in the environment.
        No really required for the interaction state model, but makes handling more similar to the
        gating model and allows easier parsing of the interaction states.
    """
    
    def __init__(self):
        self.id = 0
        if config.USE_DYNS:
            self.vec = np.zeros(6)
            self.lastVec = np.zeros(6)
        else:
            self.vec = np.zeros(3)
            self.lastVec = np.zeros(3)
    
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
            res.vec = np.zeros(6)
            res.vec[0] = npround(m.pose.position.x, config.NUMDEC) #posX
            res.vec[1] = npround(m.pose.position.y, config.NUMDEC) #posY
            res.vec[2] = npround(common.quaternionToEuler(np.array([m.pose.orientation.x,m.pose.orientation.y,
                                                m.pose.orientation.z,m.pose.orientation.w])), config.NUMDEC)[2] #ori
            res.vec[3] = npround(m.linVel.x, config.NUMDEC) #linVelX
            res.vec[4] = npround(m.linVel.y, config.NUMDEC) #linVelY
            res.vec[5] = npround(m.angVel.z, config.NUMDEC) #angVel
        else:
            res.vec = np.zeros(3)
            res.vec[0] = npround(m.pose.position.x, config.NUMDEC) #posX
            res.vec[1] = npround(m.pose.position.y, config.NUMDEC) #posY
            res.vec[2] = npround(common.quaternionToEuler(np.array([m.pose.orientation.x,m.pose.orientation.y,
                                                m.pose.orientation.z,m.pose.orientation.w])), config.NUMDEC)[2] #ori
        res.lastVec = npcopy(res.vec)
        return res
        
    @classmethod
    def fromInteractionState(cls, intState):
        """
            Classmethod to extract the two object states from an interaction state.
            
            Parameter
            ----------
            intState : InteractionState
                Interactionstate from where the object states are to be extracted.
                
            Returns
            -------
            o1 : Object
                Reference object from the interaction state
            o2 : Object
                Second object from the interaction state
                
        """
        #TODO cannot be used with dynamics as is. Assumes 2nd object to be the actuator
        o1 = cls()
        o2 = cls()
        o1.id = int(intState.vec[0])
        o2.id = int(intState.vec[1])
        p1 = np.ones(3)
        p1[:2] = intState.vec[2:4]
        p1 = np.dot(intState.trans, p1)
        o1.vec[:2] = p1[:2]
        o1.vec[2] = intState.vec[4] + intState.ori
        p2 = np.ones(3)
        p2[:2] = intState.vec[5:7]
        p2 = np.dot(intState.trans, p2)
        o2.vec[:2] = p2[:2]
        o2.vec[2] = 0.0
        return o1, o2
        
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
        return np.array([npcopy(self.vec[0:2]), np.array([p1xn,p1yn]), np.array([p2xn,p2yn])])
    
class InteractionState(object):
    """
        Implementation of the interaction state.
    """
    
    def __init__(self):
        self.id = ""
        self.vec = np.zeros(8)
        self.lastVec = np.zeros(8)
        self.trans = None
        self.invTrans = None
        self.ori = 0.0
        pass
    
    def update(self, newState):
        """
            Updates the interaction state with new information
            
            Parameters
            ----------
            newState : InteractionState
                Interaction state whose values should be adopted.
        """
        self.lastVec = npcopy(self.vec)
        self.vec = npcopy(newState.vec)
        self.ori = newState.ori
        self.trans = npcopy(newState.trans)
        self.invTrans = npcopy(newState.invTrans)
    
    @classmethod
    def fromObjectStates(cls, o1, o2):
        """
            Classmethod to create an interaction state from two object states.
            
            Parameters
            ----------
            o1 : Object
                Reference object for the interaction state
            o2 : Object
                Secondary object for the interaction state
                
            Returns
            -------
                InteractionState
            
        """
        #TODO Does currently not use dynamics
        res = cls()
        res.id = str(o1.id) + "," + str(o2.id)
        
        res.vec[0] = o1.id
        res.vec[1] = o2.id
#        res.vec[2:5] = 0.0 #Does not need to be set since it will be zero in local coordinate system
        res.vec[5:7] = common.relPos(o1.vec[0:2], o1.vec[2], o2.vec[0:2])
        res.vec[7] = o2.vec[2]-o1.vec[2] 
        res.lastVec = npcopy(res.vec)
        res.trans = common.eulerPosToTransformation(o1.vec[2],o1.vec[0:2])
        res.invTrans = common.invertTransMatrix(res.trans)
        res.ori = o1.vec[2]
        return res

        
    def circle(self, toTarget = None):
        """
            Function to return an action that would circle the other object around 
            itself. 
            
            Parameters
            ----------
            toTarget : np.ndarray, optional
                Relative direction of the object towards the target position
            
            Returns
            -------
                np.ndarray
                Action vector for the actuator for the next step
        """
        o1,o2 = Object.fromInteractionState(self)
        dist = common.generalDist(o1.id, o1.vec[0:2], o1.vec[2], o2.id, o2.vec[:2], o2.vec[2])
        relPos = self.vec[5:7]
        globRelPos = np.dot(self.trans[:-1,:-1], relPos)
        if dist < 0.04:
            return 0.4*globRelPos/np.linalg.norm(globRelPos)
        elif dist > 0.06:
            return -0.4*globRelPos/np.linalg.norm(globRelPos)
        tangent = np.array([-globRelPos[1], globRelPos[0]])
        if toTarget != None:
            angAct = np.arctan2(relPos[1],relPos[0])
            angTarget = np.arctan2(toTarget[1],toTarget[0])
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
            
            return -0.4*tangent/np.linalg.norm(tangent)
        else:
            return 0.4*tangent/np.linalg.norm(tangent)
    
class WorldState(object):
    """
        Container class for all the object and interaction states.
    """
    
    def __init__(self):
        self.objectStates = {}
        self.interactionStates = {}
        
        self.actuator = None
        
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
                tmp = Object.parse(m)               
                self.objectStates[tmp.id] = tmp
                if tmp.id == 8:
                    self.actuator = tmp
                
    def parseInteractions(self):
        """
            Parses the known object states to interaction states. The actuator is used as secondary
            object for all interaction states.
        """
        for n1, os1 in self.objectStates.items():
            if n1 != 8:
                for n2, os2 in self.objectStates.items():
                    if n1 != n2:
                        intState = InteractionState.fromObjectStates(os1,os2)
                        self.interactionStates[intState.id] = intState
        
    def parse(self, gzWS):
        """
            Parses a protobuf WorldState message
            
            Parameters
            ---------
            gzWS : Protobuf.WorldState
                Worldstate message provided by the simulation
        """
        self.parseModels(gzWS.model_v.models)   
        self.parseInteractions()
        
    def finalize(self):
        """
            Recomputes the object states from the interaction states.
            In case more than one interactionState is given, only the
            last object state is kept. Will break with multiple objects!!!
        """
        for intState in self.interactionStates.values():
            o1, o2 = Object.fromInteractionState(intState)
            self.objectStates[o1.id] = o1
            self.objectStates[o2.id] = o2
            
        self.parseInteractions()
    
    
class Episode(object):
    """
        Implementation of the Episode container.
        Stores past experiences and computes the difference vectors.
    """    
    
    
    def __init__(self, pre, action, post):
        """
            Constructor which sets up the episode. Performs transformation of the post state and
            computes the difference vector.
            
            Parameters
            ----------
            pre : InteractionState
                Interactionstate before the action
            action : np.ndarray(2)
                Action primitive that caused the change from pre to post
            post : InteractionState
                Interactionstate after the action
        """
        self.preState = pre
        self.action = action
        self.postState = post
        #Transform post to coordinate system of pre to compute proper difs
        postVec = npcopy(post.vec)
        oldPos = np.ones(3)
        oldPos[:2] = postVec[2:4]
        postVec[2:4] = np.dot(np.dot(pre.invTrans,post.trans), oldPos)[:2]
        postVec[4] += post.ori-pre.ori
        oldAltPos = np.ones(3)
        oldAltPos[:2] = postVec[5:7]
        postVec[5:7] = np.dot(np.dot(pre.invTrans,post.trans), oldAltPos)[:2]
        
        self.difs = postVec-pre.vec
        
    def getChangingFeatures(self):
        """
            Computes the set of changing features.
            
            Returns
            -------
                np.ndarray
                List of indices for the feature dimensions that changed
        """
        return np.where(abs(self.difs)>config.episodeDifThr)[0]

        
    
class AbstractCollection(object):
    """
        Implementation of the Abstract Collection. Contains the local forward model.
        Past episodes are stored for potential local optimizations.
    """
    
    def __init__(self, identifier, changingFeatures):
        """
            Abstract Collection (AC) Constructor
            Parameters
            ----------
            identifier : int
                Identifier for this AC
            changingFeatures: np.ndarray
                List of feature indices that this AC is responsible for
        """
        self.id = identifier
        self.predictor = AITM()
        self.inverseModel = MetaNetwork()
        self.changingFeatures = npcopy(changingFeatures)
        self.storedEpisodes = []
    
    def update(self, episode):
        """
            Updates the Abstract Collection with another episode
            
            Parameters
            ----------
            episode : Episode
                Current episode from which the AC should learn.
        """
        #Translation can be ignored since we are dealing with velocity
        transAction = np.dot(episode.preState.invTrans[:-1,:-1], episode.action) 
        vec = np.concatenate((episode.preState.vec, transAction))
        self.predictor.update(vec, episode.difs[self.changingFeatures], 
                              etaIn=config.aCEtaIn, etaOut=config.aCEtaOut, 
                              etaA=config.aCEtaA, testMode=config.aCTestMode)
        self.storedEpisodes.append(episode)
        self.inverseModel.train(vec, episode.difs[self.changingFeatures])
    
    def predict(self, intState, action):
        """
            Predicts the state of the given interactionstate after the given action is performed.
            Action will to be transformed to local coordinate system.
            
            Parameters
            ---------
            intState : InteractionState
                InteractionState whose next state is to be predicted
            action : np.ndarray
                Action vector that is to be used
            
            Returns
            -------
                InteractionState
        """
        res = InteractionState()
        res.id = intState.id
        res.trans = npcopy(intState.trans)
        res.invTrans = npcopy(intState.invTrans)
        res.ori = intState.ori
        res.lastVec = npcopy(intState.vec)
        res.vec = npcopy(intState.vec)
        transAction = np.dot(intState.invTrans[:-1,:-1], action)
        vec = np.concatenate((intState.vec, transAction))
        prediction = self.predictor.test(vec, testMode=config.aCTestMode)
        res.vec[self.changingFeatures] += prediction
        return res
        
    def getAction(self, difs):
        """
            Queries the local inverse model if used for preconditions that contain 
            action primitives.
            
            Parameters
            ----------
            difs : np.ndarray
                The difference vector that is to be reduced by the preconditions
            
            Returns
            ------
                np.ndarray
                Preconditions returned by the inverse model.
        """
        return self.inverseModel.getPreconditions(difs)

    
class ACSelector(object):
    """
        Implementation of the Abstract Collection Selector. Basically a wrapper for the
        AITM as classifier.
    """
    
    def __init__(self):
        self.classifier = AITM()
        self.isTrained = False
        
    def update(self, intState, action, respACId):
        """
            Updates the classifier. The action is first transformed to the local coordinate system
            of the given interaction state.
            
            Parameters
            ----------
            intState : InteractionState
                InteractionState that produced changes that respACId is responsible for with the
                given action
            action : np.ndarray
                Action primitive that produced the changes
            respACId : int
                Identifier of the AC that is responsible for the changes produced by intState 
                and action
        """
        transAction = np.dot(intState.invTrans[:-1,:-1], action) 
        vec = np.concatenate((intState.vec[config.aCSelectorMask], transAction))
        self.classifier.update(vec, np.array([respACId]), 
                               etaIn=config.aCSelectorEtaIn, 
                               etaOut=config.aCSelectorEtaOut, 
                               etaA=config.aCSelectorEtaA, 
                               testMode=config.aCSelectorTestMode)        
        self.isTrained = True
    
    def test(self, intState, action):
        """
            Queries the classifier for the AC id. Action is first transformed to the local 
            coordinate system of the interactionstate.
            
            Parameters
            ---------
            intState : InteractionState
                Current interactionstate.
            action : np.ndarray
                Current action primtive that is to be used.
            
            Returns
                int
                Identifier of the AC that is most likely responsible for the changes produced by 
                action.
        """
        transAction = np.dot(intState.invTrans[:-1,:-1], action) 
        if self.isTrained:
            vec = np.concatenate((intState.vec[config.aCSelectorMask], transAction))
            #Only use winner to make prediction, interpolation does not really
            #make sense when having more then 2 classes
            return int(self.classifier.test(vec, testMode=config.aCSelectorTestMode)) 
        else:
            return None


class ModelInteraction(object):
    """
        Implementation of the interaction state model.
    """

    def __init__(self):
        self.acSelector = ACSelector()
        self.abstractCollections = {}
        self.curInteractionStates = {}
        self.featuresACMapping = {}
        self.training = True
        self.target = None
        self.targetFeatures = []
        self.inverseModel = MetaNetwork()
        pass    
    
    
    def getITMInformation(self):
        """
            Helper function to record the state of the included AITMs.
            
            Returns
            -------
                String
                Information about the number of nodes and update calls of each AITM
        """
        acSelString = "acSelector ITM: UpdateCalls: {}, Insertions: {}, final Number of nodes: {}\n"\
                                    .format(self.acSelector.classifier.updateCalls,
                                            self.acSelector.classifier.inserts, 
                                            len(self.acSelector.classifier.nodes))
        acString = ""
        for ac in self.abstractCollections.values():
            acString += "Abstract collection for features: {}, ITM: UpdateCalls: {}, \
                         Insertions: {}, final Number of nodes: {}\n"\
                        .format(ac.changingFeatures, ac.predictor.updateCalls, 
                                ac.predictor.inserts, len(ac.predictor.nodes))
        return acSelString + acString
        
    def setTarget(self, target):
        """
            Sets a target that is to be reached. Constructs an InteractionState as target from
            the given Object state.
            
            Parameters
            ----------
            target : Object
                Object state of the target that is to be reached
        """
        #Create target interaction state from object
        if target.id == 8:
            raise NotImplementedError("Currently only non acutator objects can be set as targets.")
        dummyObject = Object()
        dummyObject.id = 8
        self.target = InteractionState.fromObjectStates(target, dummyObject)
        self.targetFeatures = [2,3,4]
        
    def isTargetReached(self):
        """
            Helper function to check if the current target has already been reached.
            
            Returns
            ------
                bool
                True if the target has been reached, False otherwise
        """
        targetInteraction = self.curInteractionStates[self.target.id]
        episode = Episode(targetInteraction, np.zeros(2), self.target)
        if np.linalg.norm(episode.difs[self.targetFeatures]) < 0.01:
            return True
        return False
        
    def getAction(self):
        """
            Returns an action, that is to be performed, trying to get closer to the
            target if one is set. If no target is set, returns the 0 action.
            
            Returns: np.ndarray
                Action vector for the actuator
        """
        if self.target is None:
            return np.array([0.0,0.0])
        else:
            if self.isTargetReached():
                self.target = None
                return np.zeros(2)
            else:
                targetInteraction = self.curInteractionStates[self.target.id]
                #Create desired episode with zero action, in order to get required difference vector
                desiredEpisode = Episode(targetInteraction, np.zeros(2), self.target)
                #Only consider target differences
                
                targetDifs = np.zeros(len(desiredEpisode.difs))
                targetDifs[self.targetFeatures] = desiredEpisode.difs[self.targetFeatures]

                #Use global inverse model to avoid having to select the correct one
                preConditions = self.inverseModel.getPreconditions(targetDifs)
                
                if preConditions == None:
                    randAction = np.dot(targetInteraction.trans[:2,:2],-targetInteraction.vec[5:7]) + \
                                        (np.random.rand(2)-0.5)
                    norm = np.linalg.norm(randAction)
                    if norm > 0:
                        return 0.3*randAction/norm
                    else:
                        return randAction
                relTargetPos = preConditions[5:7]
                globTargetPos = np.ones(3)
                globTargetPos[:2] = np.copy(relTargetPos)
                globTargetPos = np.dot(targetInteraction.trans, globTargetPos)[:2]
                relAction = preConditions[8:10]
                curRelPos = targetInteraction.vec[5:7]
                globCurPos = np.ones(3)
                globCurPos[:2] = np.copy(curRelPos)
                globCurPos = np.dot(targetInteraction.trans, globCurPos)[:2]
                
                difPos = globTargetPos-globCurPos
                globAction = np.dot(targetInteraction.trans[:2,:2],relAction)
                if np.linalg.norm(difPos) > 0.1:
                    return targetInteraction.circle(relTargetPos)
                if np.linalg.norm(difPos) > 0.01:
                    return 0.3*difPos/np.linalg.norm(difPos)                
                if np.linalg.norm(globAction) == 0.0:
                    return globAction
                return  0.3*globAction/np.linalg.norm(globAction)
                pass
    
    def update(self, curWorldState, usedAction):
        """
            Updates the model. For each interaction state in the new worldState an episode
            is created to compute the difference vector. This is then used together with the old
            interactionstate to train the ACS and the ACs.
            
            Parameters
            ----------
            curWS : WorldState
                Current worldState which is used to update
            usedAction : np.ndarray(2)
                Action primitive used to produce the current worldstate
        """
        
        for intId, intState in curWorldState.interactionStates.items():
            if intId in self.curInteractionStates:
                #When training, update the ACs and the selector
                if self.training:
                    newEpisode = Episode(self.curInteractionStates[intId], usedAction, intState)
                    changingFeatures = newEpisode.getChangingFeatures()
                    featuresString = ",".join(map(str,changingFeatures))
                    #Create AC if not already known
                    if not featuresString in self.featuresACMapping:
                        newAC = AbstractCollection(len(self.abstractCollections),changingFeatures)
                        self.abstractCollections[newAC.id] = newAC
                        self.featuresACMapping[featuresString] = newAC.id
                        
                    self.abstractCollections[self.featuresACMapping[featuresString]].update(newEpisode)
                    self.acSelector.update(self.curInteractionStates[intId], usedAction, self.featuresACMapping[featuresString])
                    transAction = np.dot(self.curInteractionStates[intId].invTrans[:-1,:-1], usedAction) 
                    vec = np.concatenate((self.curInteractionStates[intId].vec, transAction))
                    self.inverseModel.train(vec, newEpisode.difs)
                    
                self.curInteractionStates[intId].update(intState)
            else:
                self.curInteractionStates[intId] = intState
        
    
    def predict(self, curWorldState, curAction):
        """
            Predicts the next worldstate given a current worldstate and an action primitive.
            Predicted worlsState is finalized at the end in order to extract the predicted object
            states.
            
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
        for intId, intState in curWorldState.interactionStates.items():
            acID = self.acSelector.test(intState, curAction)
            if acID == None:
                newIntState = copy.deepcopy(intState)
            else:
                newIntState = self.abstractCollections[acID].predict(intState, curAction)
            newWS.interactionStates[intId] = newIntState
        
        newWS.finalize()
        return newWS
        
        
        
    def resetObjects(self, curWS):
        """
            Updates the known object positions. Required for resetting a run, without loosing
            what is already learned. All interactionstates are updated.
            
            Parameters
            ---------
            curWS : WorldState
                Current worldstate whose object states represent the new values
        """
        self.curInteractionStates = {}
        for i in curWS.interactionStates.values():
            if i.id in self.curInteractionStates:
                self.curInteractionStates[i.id].update(i)
            else:
                self.curInteractionStates[i.id] = i