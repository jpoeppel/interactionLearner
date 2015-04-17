#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:34:43 2015
TODOS:
* Implement possibility to set target!!!
* Implement active learning!!!!
* Block has action like gripper -> wrong?
* action influence appears not to be learned too well for pos
* Check the case selection when finding best cases, it appears some are quite bad.
* The model should remove unnecessary attributes (like spos) itself
* Split relevantScoringKeys and relevantTrainingKeys?
@author: jpoeppel
"""

import numpy as np
from metrics import similarities
from metrics import differences
from common import GAZEBOCMDS as GZCMD
from sklearn.gaussian_process import GaussianProcess
from topoMaps import ITM
from network import Node
import copy
import math

THRESHOLD = 0.01
NUMDEC = 5
MAXCASESCORE = 16
MAXSTATESCORE = 14
#PREDICTIONTHRESHOLD = 0.5
PREDICTIONTHRESHOLD = MAXSTATESCORE - 0.2

class State(dict):
    """
        Base class representing the state of something. Will be used to
        derive specific state classes from.
    """
    
    def __init__(self):
        pass
    
    def score(self, otherState):
        assert isinstance(otherState, State), "{} is not a State object".format(otherState)
        s = MAXSTATESCORE
        for k in self.relevantKeys():
            s -= differences[k](self[k], otherState[k]) #* weights[k] 
        return s
        
    def relevantKeys(self):
        return self.keys()
        
    def relevantItems(self):
        r = []
        for k in self.relevantKeys():
            r.append((k,self[k]))
        return r
        
    def toVec(self, const = {}):
        r = np.array([])
        for k in self.relevantKeys():
            if k not in const.keys():
#            if k != "spos":
                if isinstance(self[k], np.ndarray):
                    r = np.concatenate((r,self[k]))
                elif not isinstance(self[k], unicode):
                    r = np.concatenate((r,[self[k]]))
        return r
        
        
class ObjectState(State):
    """
        State class used to represent object states.
        Holds information about object position, orientation, type, id, name 
        and other properties.
    """
    
    def __init__(self):
        self.update({"id": -1, "name": "", "type": -1, "pos": np.zeros(3), 
                         "orientation": np.zeros(4), "linVel": np.zeros(3), 
                         "angVel": np.zeros(3)})
        
class InteractionState(State):
    
    def __init__(self, intId, o1):
        self.update({"intId": intId, "sid":o1["id"], "sname": o1["name"], 
                     "stype": o1["type"], "spos":o1["pos"], 
                     "sori": o1["orientation"], "slinVel": o1["linVel"], 
                     "sangVel": o1["angVel"], "dist": 0, "dir": np.zeros(3),
                     "contact": 0, "oid": -1, "oname": "", "otype": 0, 
                     "dori": np.zeros(4), "dlinVel": np.zeros(3), "dangVel":np.zeros(3)})
                     
    def relevantKeys(self):
        keys = self.keys()
        keys.remove("intId")
        keys.remove("sname")
#        keys.remove("spos")
#        keys.remove("stype")
#        keys.remove("sori")
#        keys.remove("sangVel")
#        keys.remove("oid")
        keys.remove("oname")
#        keys.remove("dlinVel")
#        keys.remove("dori")
#        keys.remove("dangVel")
#        keys.remove("otype")
        return keys
                     
    def fill(self, o2):
        assert isinstance(o2, ObjectState), "{} (o2) is not an ObjectState!".format(o2)
        self["dist"] = np.linalg.norm(self["spos"]-o2["pos"])
        self["dir"] = o2["pos"]-self["spos"]
        self["oid"] = o2["id"]
        self["oname"] = o2["name"]
        self["otype"] = o2["type"]
        self["dori"] = o2["orientation"]-self["sori"] # TODO Fix THIS IS WRONG!!!
        self["dlinVel"] = o2["linVel"] - self["slinVel"]
        self["dangVel"] = o2["angVel"] - self["sangVel"]
    
class Action(State):
    
    def __init__(self, cmd=GZCMD["NOTHING"], direction=np.array([0.0,0.0,0.0])):
        self.update({"cmd":cmd, "mvDir": direction})
        
class WorldState(object):
    
    def __init__(self):
        self.objectStates = {}
        self.interactionStates = {}
        self.numIntStates = 0
        self.predictionCases = {}

    def addInteractionState(self, intState, usedCase = None):
#        print "adding interactionState: ", intState["intId"]
        assert isinstance(intState, InteractionState), "{} (intState) is not an InteractionState object.".format(intState)
        self.interactionStates[intState["intId"]] = intState
        self.numIntStates += 1        
        self.predictionCases[intState["intId"]] = usedCase
    
    def parseModels(self, models):
        for m in models:
            if m.name == "ground_plane" or "wall" in m.name or "Shadow" in m.name:
                #Do not model the ground plane or walls for now
                continue
            else:
                tmp = ObjectState()
                tmp["pos"] = np.round(np.array([m.pose.position.x,m.pose.position.y,m.pose.position.z]), NUMDEC)
                tmp["orientation"]  = np.round(np.array([m.pose.orientation.x,m.pose.orientation.y,
                                            m.pose.orientation.z,m.pose.orientation.w]), NUMDEC)
                tmp["linVel"] = np.round(np.array([m.linVel.x,m.linVel.y,m.linVel.z]), NUMDEC)
                tmp["angVel"] = np.round(np.array([m.angVel.x,m.angVel.y,m.angVel.z]), NUMDEC)
                tmp["name"] = m.name
                tmp["id"] = m.id
                tmp["type"] = m.type
                self.objectStates[m.id] = tmp
                
    def parseInteractions(self, ws):
        tmpList = self.objectStates.values()
        for o1 in self.objectStates.values():
            intState = InteractionState(self.numIntStates, o1)
            for o2 in tmpList:
                if not np.array_equal(o1,o2):                    
                    intState.fill(o2)
                    self.addInteractionState(intState)
                
#    def parseContacts(self, contacts):
#        for c in contacts:
#            if self.objectStates.has_key(c.wrench[0].body_1_id):
#                self.objectStates[c.wrench[0].body_1_id].addContact(c.wrench[0].body_2_id)
    
    def parse(self, gzWS):
        self.parseModels(gzWS.model_v.models)
        self.parseInteractions(gzWS)
#        self.parseContacts(ws.contacts.contact)
    

class BaseCase(object):
    
    def __init__(self, pre, action, post):
        assert isinstance(pre, State), "{} is not a State object.".format(pre)
        assert isinstance(post, State), "{} is not a State object.".format(post)
        assert isinstance(action, Action), "{} is not an Action object.".format(action)
        assert (pre.keys()==post.keys()), "Pre and post states have different keys: {}, {}.".format(pre.keys(), post.keys())
        self.preState = pre
        self.postState = post
        self.action = action
        self.dif = {}
        self.abstCase = None
        for k in pre.relevantKeys():
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
        
        
    def getListOfConstants(self):
        #TODO make more efficient by storing these values
        r = []
        for k in self.dif.keys():
            if np.linalg.norm(self.dif[k]) <= THRESHOLD:
                r.append((k,self.preState[k]))
        return r
    
    def predict(self, state,action, attrib):
        return self.dif[attrib]
        
    def score(self, state, action):
        s = MAXCASESCORE
        for k in state.relevantKeys():
            if self.preState.has_key(k):
                bestScore = 1
                
                tmpScore = differences[k](state[k], self.preState[k])
                if tmpScore < bestScore:
                    bestScore = tmpScore
                s -= bestScore
        for k in action.relevantKeys():
            bestScore = 1
            tmpScore = differences[k](action[k], self.action[k])
            if tmpScore < bestScore:
                bestScore = tmpScore
            s -= bestScore
        return s
        
class AbstractCase(object):
    
    def __init__(self, case):
        assert isinstance(case, BaseCase), "case is not a BaseCase object."
        self.refCases = []
        self.avgPrediction = 0.0
        self.numPredictions = 0
        self.name = ""
        self.variables = [] #List of attributes that changed 
        self.attribs = {} # Dictionary holding the attributs:[values,] pairs for the not changing attribs of the references
        self.predictors = {}
        self.variables.extend(case.getListOfAttribs())
        self.preCons = {}
        self.gaussians = {}
        for k in self.variables:
            self.predictors[k] = ITM()
        self.addRef(case)
        
    def predict(self, state, action):
#        print "predicting for variables: ", self.variables
    
        resultState = copy.deepcopy(state)
        if len(self.refCases) > 1:
#            print "resultState intId: ", resultState["intId"]
            for k in self.variables:
                resultState[k] = state[k] + self.predictors[k].predict(np.concatenate((state.toVec(self.preCons),action.toVec(self.preCons))))
#                if k == "spos":
#                    print "prediction: ", self.predictors[k].predict(np.concatenate((state.toVec(self.preCons),action.toVec(self.preCons))))
#            print "resultState intId after: ", resultState["intId"]
            
        else:
#            print "predicting with only one ref"
            for k in self.variables:
                resultState[k] = state[k] + self.refCases[0].predict(state, action, k)
#            prediction= self.refCases[0].predict(state,action)
#            prediction["intId"] = state["intId"]
        return resultState
            
            
    def score(self, state, action):
#        s = MAXCASESCORE
        s = 0
        for k,v in state.relevantItems() + action.relevantItems():
            
#            if s > 16:
#                print "mu: {}, cov: {}, det:{}, inv: {}".format(mu, cov, det, inv)
#            det = np.linalg.det(cov)
#            print "cov: ", cov
#            print "mu: ", mu
#            print "det: ", det
#            print "inv: ", inv
#            print "v: ", v
#            print "result: ", math.exp(-0.5*np.dot(np.dot((v-mu),inv),(v-mu).T))
            
            
#            if self.preCons.has_key(k):
##                bestScore = 1
#                if differences[k](self.preCons[k], v) <= 0.05:
#                    s+= 1
#            else:
#            else:
            if hasattr(v, "__len__"):
                dim = len(v)
            else:
                dim = 1
            mu, cov, det, inv = self.gaussians[k]
#                print "det: ", det
            norm = 1.0/(math.sqrt((2*math.pi)**dim * det))
#            if norm > 1:
##                print "cov: ", cov
##                print "norm still too big: ", norm
#                norm = 1
            
            s += norm * math.exp(-0.5*(np.dot(np.dot((v-mu),inv),(v-mu).T)))
#            if tmp > 1:
               
#                #For keys that are not preconditions, add average value, so that not simply
#                # the cases with the most preconditions win
#                s += 0.75
#                for v in self.attribs[k]:
#                    tmpScore = metrics[k](state[k], v)#Forgot action!!
#                    if tmpScore < bestScore:
#                        bestScore = tmpScore
#                s -= bestScore
#        print "score for case with list {}: {}".format(self.variables, s)
        return s
    
    def updatePredictionScore(self, score):
        self.numPredictions += 1
        self.avgPrediction += (score-self.avgPrediction)/self.numPredictions
        
    def addRef(self, ref):
        ref.abstCase = self
#        for k,v in ref.getListOfConstants():
#            if self.attribs.has_key(k):
#                if np.array_equal()
#                if any(np.array_equal(v,x) for x in self.attribs[k]):
#                    self.attribs[k].append(v)
#            else:
#                self.attribs[k] = [v]
#        for k,v in ref.preState.relevantItems() + ref.action.relevantItems():
#            if self.preCons.has_key(k):
#                #If the previous value is roughly the same as the new one, keep it. Else say it can be arbitrary?
#                if differences[k](self.preCons[k], v) > 0.05: #THRESHOLD:
#                    #Remove pre-condition
#                    del self.preCons[k]           
#                    self.retrain()
#            else:
#                #Only add preconditions for the first case
#                if len(self.refCases) == 0:
#                    self.preCons[k] = v
            
         
        self.refCases.append(ref)
        self.updatePredictorsITM(ref)
        
        self.updateGaussians(ref)
#        self.updatePredictorsGP()
        
    def updateGaussians(self, ref):
        for k,v in ref.preState.relevantItems() + ref.action.relevantItems():
            if hasattr(v, "__len__"):
                dim = len(v)
            else:
                dim = 1
                
            if not self.gaussians.has_key(k):
                self.gaussians[k] = (np.array(v)[np.newaxis], np.identity(dim), 1, np.identity(dim))
            else:
                numData = len(self.refCases)
                muO, covO, detO, invO = self.gaussians[k]
                mu = (1-1.0/numData)*muO + 1.0/numData*v
                cov = (1-1.0/numData)*(covO+1.0/numData*np.dot((v-muO).T,(v-muO)))
#                inv = numData/(numData-1)*(invO-(invO*np.dot(np.dot((v-muO).T,v-muO),invO))/(numData + np.dot(np.dot(v-muO,invO),(v-muO).T)))
                inv = np.linalg.inv(cov)
                self.gaussians[k] = (mu, cov, np.linalg.det(cov), inv )

            
    def getData(self, attrib):
        
        numCases = len(self.refCases)
        if self.refCases[0].preState.has_key(attrib):
            if hasattr(self.refCases[0].preState[attrib], "__len__"):
                dim = len(self.refCases[0].preState[attrib])
            else:
                dim = 1
            data = np.zeros((numCases,dim))
            for i in range(numCases):
                data[i,:] = self.refCases[i].preState[attrib]
        elif self.refCases[0].action.has_key(attrib):
            if hasattr(self.refCases[0].action[attrib], "__len__"):
                dim = len(self.refCases[0].action[attrib])
            else:
                dim = 1
            data = np.zeros((numCases,dim))
            for i in range(numCases):
                data[i,:] = self.refCases[i].action[attrib]
        else:
            raise TypeError("Invalid attribute: ", attrib)
            
        return data
        
        
    def retrain(self):
        for k in self.variables:
            self.predictors[k] = ITM()
        for c in self.refCases:
            self.updatePredictorsITM(c)
#        self.updatePredictorsGP()
        
    def updatePredictorsITM(self, case):
        for k in self.variables:
            self.predictors[k].train(self.toNode(case, k))
            
    def toNode(self, case, attrib):
        node = Node(0, wIn=np.concatenate((case.preState.toVec(self.preCons),case.action.toVec(self.preCons))),
                    wOut=case.postState[attrib]-case.preState[attrib])
        return node
        
    def updatePredictorsGP(self):
        if len(self.refCases) > 1:
            for k in self.variables:
                self.predictors[k] = GaussianProcess(corr='cubic')
                data, labels = self.getTrainingData(k)
                self.predictors[k].fit(data, labels)
                
    def getTrainingData(self, attrib):
        inputs = []
        outputs = []
        for c in self.refCases:
            inputs.append(np.concatenate((c.preState.toVec(self.preCons),c.action.toVec(self.preCons))))
            outputs.append(c.postState[attrib]- c.preState[attrib])
        return inputs, outputs
    
class ModelCBR(object):
    
    def __init__(self):
        self.cases = []
        self.abstractCases = []
        self.numZeroCase = 0
        self.numCorrectCase = 0
        self.numPredictions = 0
        
    def getAction(self, state):
        pass
    
    def getBestCase(self, state, action):
        bestCase = None
        bestScore = 0.0
        for c in self.abstractCases:
            s = c.score(state, action)
            if s >= bestScore:
                bestCase = c
                bestScore = s
        if isinstance(bestCase, AbstractCase):
            print "bestCase #refs: ", len(bestCase.refCases)
            print "bestCase variables: ", bestCase.variables
            if bestCase.variables == []:
                self.numZeroCase += 1
#            print "bestCase preConditions: ", bestCase.preCons
            return bestCase
        elif isinstance(bestCase, BaseCase):
            if bestCase.abstCase != None:
                print "bestCase #refs: ", len(bestCase.abstCase.refCases)
                print "bestCase avgPrediction: ", bestCase.abstCase.avgPrediction
            return bestCase.abstCase
        else:
            return None
    
    def predictIntState(self, state, action):
        self.numPredictions += 1
        bestCase = self.getBestCase(state, action)
        if bestCase != None:
            return bestCase.predict(state, action), bestCase
        else:
#            print "using old state with id: ", state["intId"]
            return state, bestCase
    
    def predict(self, worldState, action):
        
        predictionWs = WorldState()
        for intState in worldState.interactionStates.values():
            
#            print "predicting for ", intState["intId"]
            prediction, usedCase = self.predictIntState(intState, action)
#            print "predicted intId: ", prediction["intId"]
            predictionWs.addInteractionState(prediction, usedCase)
#        print "resulting prediction: ", predictionWs.interactionStates
        return predictionWs
        
    def updateState(self, state, action, prediction, result, usedCase):
        """
        Parameters
        
        state: InteractionState
        Action: Action
        prediction: InteractionState
        result: Interaction
        usedCase: AbstractCase
        """
        newCase = BaseCase(state, action, result)
        attribList = newCase.getListOfAttribs()
        predictionScore = result.score(prediction)
        print "predictionScore: ", predictionScore
        if usedCase != None and usedCase.variables == attribList:
            usedCase.updatePredictionScore(predictionScore)
            self.numCorrectCase += 1
        if predictionScore < PREDICTIONTHRESHOLD:
            self.cases.append(newCase)
            abstractCase = None
            for ac in self.abstractCases:
                if ac.variables == attribList:
                    abstractCase = ac
                    #TODO consider search for all of them in case we distinguis by certain features
                    break
            if abstractCase != None:
                #If an abstract case is found add the reference
                abstractCase.addRef(newCase)
            else:
                #Create a new abstract case
                self.abstractCases.append(AbstractCase(newCase))
                
            
            
    
    def update(self, state, action, prediction, result):
        for intState in state.interactionStates.keys():
            if not prediction.interactionStates.has_key(intState):
                print "prediction: ", prediction.interactionStates
            if not result.interactionStates.has_key(intState):
                print "result: ", result.interactionStates
            if not prediction.predictionCases.has_key(intState):
                print "prediction.predictionCases: ", prediction.predictionCases
            self.updateState(state.interactionStates[intState], action, prediction.interactionStates[intState], 
                             result.interactionStates[intState], prediction.predictionCases[intState])
        