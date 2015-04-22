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
#from metrics import weights
from common import GAZEBOCMDS as GZCMD
from sklearn.gaussian_process import GaussianProcess
from topoMaps import ITM
from network import Node
import copy
import math
from sets import Set

THRESHOLD = 0.005
NUMDEC = 5
MAXCASESCORE = 16
MAXSTATESCORE = 14
#PREDICTIONTHRESHOLD = 0.5
PREDICTIONTHRESHOLD = MAXSTATESCORE - 0.01


#
import logging
logging.basicConfig()

class State(dict):
    """
        Base class representing the state of something. Will be used to
        derive specific state classes from.
    """
    
    def __init__(self):
        self.relKeys = self.keys()
        self.weights = {}
        for k in self.relKeys:
            self.weights[k] = 1.0
        pass
    
    def score(self, otherState):
        assert isinstance(otherState, State), "{} is not a State object".format(otherState)
        s = MAXSTATESCORE
        for k in self.relKeys:
            s -= self.weights[k]*differences[k](self[k], otherState[k]) #* weights[k] 

        return s
        
    def relevantKeys(self):
        return self.relKeys
        
    def relevantItems(self):
        r = []
        for k in self.relKeys:
            r.append((k,self[k]))
        return r
        
    def toVec(self, const = {}):
        r = np.array([])
        for k in self.relKeys:
            if k not in const.keys():
#            if k != "spos":
                if isinstance(self[k], np.ndarray):
                    r = np.concatenate((r,self[k]))
                elif not isinstance(self[k], unicode):
                    r = np.concatenate((r,[self[k]]))
        return r
        
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        for k, v in self.relevantItems():
            if np.linalg.norm(v-other[k]) > THRESHOLD:
                return False
        
        return True
        
    def __ne__(self, other):
        return not self.__eq__(other)
        
        
class ObjectState(State):
    """
        State class used to represent object states.
        Holds information about object position, orientation, type, id, name 
        and other properties.
    """
    
    def __init__(self):
        State.__init__(self)
        self.update({"id": -1, "name": "", "type": -1, "pos": np.zeros(3), 
                         "orientation": np.zeros(4), "linVel": np.zeros(3), 
                         "angVel": np.zeros(3)})
        self.relKeys = self.keys()
        
class InteractionState(State):
    
    def __init__(self, intId, o1):
        
        self.update({"intId": intId, "sid":o1["id"], "sname": o1["name"], 
                     "stype": o1["type"], "spos":o1["pos"], 
                     "sori": o1["orientation"], "slinVel": o1["linVel"], 
                     "sangVel": o1["angVel"], "dist": 0, "dir": np.zeros(3),
                     "contact": 0, "oid": -1, "oname": "", "otype": 0, 
                     "dori": np.zeros(4), "dlinVel": np.zeros(3), "dangVel":np.zeros(3)})
        #Do not move from here because the keys need to be set before State.init and the relKeys need to be changed afterwards             
        State.__init__(self) 
#        self.relKeys = ["spos", "slinVel"]
        self.relKeys = self.keys()
        self.relKeys.remove("intId")
        self.relKeys.remove("sname")
        self.relKeys.remove("oname")
        
                     
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
        State.__init__(self)
        self.update({"cmd":int(round(cmd)), "mvDir": direction})
        self.relKeys = self.keys()

        
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
#            print "interactionState for o1: ", o1
            intState = InteractionState(self.numIntStates, o1)
#            self.addInteractionState(intState)
            for o2 in tmpList:
                if not np.array_equal(o1,o2):                    
                    intState.fill(o2)
                    self.addInteractionState(intState)
#                    
    def getInteractionState(self, sname):
        for i in self.interactionStates.values():
            if i["sname"] == sname:
                return i
        return None
                
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
            
        
    def getSetOfAttribs(self):
        """
            Returns the list of attributes that changed more than THRESHOLD.
        """
        r = []
        for k in self.dif.keys():
            if np.linalg.norm(self.dif[k]) > THRESHOLD:
                r.append(k)
        return Set(r)
        
        
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
        
    def __eq__(self, other):
        if not isinstance(other, BaseCase):
            return False
        return self.preState == other.preState and self.action == other.action and self.postState == other.postState
        
    def __ne__(self, other):
        return not self.__eq__(other)
        
    def __repr__(self):
        return "Pre: {} \n Action: {} \n Post: {}".format(self.preState,self.action,self.postState)
        
class AbstractCase(object):
    
    def __init__(self, case):
        assert isinstance(case, BaseCase), "case is not a BaseCase object."
        self.refCases = []
        self.avgPrediction = 0.0
        self.numPredictions = 0
        self.name = ""
        self.variables = Set() #List of attributes that changed 
        self.attribs = {} # Dictionary holding the attributs:[values,] pairs for the not changing attribs of the references
        self.predictors = {}
        self.variables.update(case.getSetOfAttribs())
        self.preCons = {}
        self.constants = {}
        self.gaussians = {}
        self.errorGaussians = {}
        self.numErrorCases = 0
        for k in self.variables:
            self.predictors[k] = ITM()
        self.addRef(case)
        
    def predict(self, state, action):
#        print "predicting for variables: ", self.variables
    
        resultState = copy.deepcopy(state)
        if len(self.refCases) > 1:
#            print "resultState intId: ", resultState["intId"]
            for k in self.variables:
                prediction = self.predictors[k].predict(np.concatenate((state.toVec(self.constants),action.toVec(self.constants))))
                if prediction != None:
                    resultState[k] = state[k] + prediction
                else:
                    resultState[k] = state[k] + self.refCases[0].predict(state, action, k)

#                if k == "spos":
#                    print "prediction: ", self.predictors[k].predict(np.concatenate((state.toVec(self.preCons),action.toVec(self.preCons))))
#            print "resultState intId after: ", resultState["intId"]
            
        else:
            print "predicting with only one ref"
            for k in self.variables:
                resultState[k] = state[k] + self.refCases[0].predict(state, action, k)
#            prediction= self.refCases[0].predict(state,action)
#            prediction["intId"] = state["intId"]
        return resultState
        
    def getAction(self, pre, var, weights, dif):
        action = np.zeros(4)
#        numVariables = len(self.variables)
        norm = 0.0
        for k in var:
            norm += weights[k]
#            print "case: {}, predicts: {} for variable K: {}".format(self.variables,self.predictors[k].predictAction(pre.toVec(), dif[k]),k )
            action += weights[k] * self.predictors[k].predictAction(pre.toVec(self.constants), dif[k])
            
        
        action /= norm
        res = Action(cmd = action[0], direction=action[1:])
        return res
            
    def addErrorCase(self, case):
        self.numErrorCases += 1
        self.updateGaussians(self.errorGaussians, self.numErrorCases, case)
            
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
            
#            s += norm * math.exp(-0.5*(np.dot(np.dot((v-mu),inv),(v-mu).T)))
            tmp = norm * math.exp(-0.5*(np.dot(np.dot((v-mu),inv),(v-mu).T)))
            if self.errorGaussians.has_key(k):
                mu, cov, det, inv = self.errorGaussians[k]
    #                print "det: ", det
                norm = 1.0/(math.sqrt((2*math.pi)**dim * det))
#                if norm > 1:
#    #                print "cov: ", cov
#    #                print "norm still too big: ", norm
#                    norm = 1
                
                tmp -=  norm * math.exp(-0.5*(np.dot(np.dot((v-mu),inv),(v-mu).T)))
                
            s += tmp

        return s
    
    def updatePredictionScore(self, score):
        self.numPredictions += 1
        self.avgPrediction += (score-self.avgPrediction)/(float(self.numPredictions))
        if self.avgPrediction <= 0.0001:
            raise Exception("something is going wrong when computing avgPrediction! score: {}, numPred: {}".format(score, self.numPredictions))
        
    def addRef(self, ref):
#        
        for k,v in ref.getListOfConstants():
            if self.constants.has_key(k):
                if np.linalg.norm(v-self.constants[k]) > THRESHOLD:
                    del self.constants[k]
                    self.retrain()
            else:
                if len(self.refCases) == 0:
                    self.constants[k] = v
        
#        for k,v in ref.getSetOfConstants():
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
        
        if ref in self.refCases:
            raise Exception("ref already in refCases")
            
#        equal = True
#        for rc in self.refCases:
#            if equal == True:
#                for k in rc.dif.keys():
#                    if np.linalg.norm(rc.dif[k]-ref.dif[k]) > 0.02:
#                       equal = False
#                       break
#                for k in rc.action.relevantKeys():
#                    if np.linalg.norm(rc.action[k] - ref.action[k]) > 0.02:
#                        equal = False
#                        break
#        if equal == True:
#            print "there is one very similar ref already"
#            import sys
#            sys.exit()
         
        self.refCases.append(ref)
        ref.abstCase = self
        self.updatePredictorsITM(ref)
        
        self.updateGaussians(self.gaussians, len(self.refCases), ref)
        
#        self.updatePredictorsGP()
        
    def updateGaussians(self, gaussians, numData, ref):
        for k,v in ref.preState.relevantItems() + ref.action.relevantItems():
            if hasattr(v, "__len__"):
                dim = len(v)
            else:
                dim = 1
                
            if not gaussians.has_key(k):
                gaussians[k] = (np.array(v)[np.newaxis], np.identity(dim), 1, np.identity(dim))
            else:
#                numData = len(self.refCases)
                muO, covO, detO, invO = gaussians[k]
                mu = (1-1.0/numData)*muO + 1.0/numData*v
                cov = (1-1.0/numData)*(covO+1.0/numData*np.dot((v-muO).T,(v-muO)))
#                inv = numData/(numData-1)*(invO-(invO*np.dot(np.dot((v-muO).T,v-muO),invO))/(numData + np.dot(np.dot(v-muO,invO),(v-muO).T)))
                inv = np.linalg.inv(cov)
                gaussians[k] = (mu, cov, np.linalg.det(cov), inv )

            
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
        node = Node(0, wIn=case.preState.toVec(self.constants), action=case.action.toVec(self.constants),
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
        self.target = None
        
    def getAction(self, state):
        bestAction = None
        bestScore = 0
        if self.target != None:
            gripperInt = state.getInteractionState(self.target["sname"])
            if gripperInt != None:
                variables = Set()
                dif = {}
                for k in gripperInt.relevantKeys():
                    dif[k] = (self.target[k] - gripperInt[k])/10.0
                    if k in self.target.relKeys and np.linalg.norm(dif[k]) > THRESHOLD:
                        variables.add(k)                    
                for ac in self.abstractCases:
                    if variables.issubset(ac.variables):
                        action = ac.getAction(gripperInt,variables, self.target.weights, dif)
                        prediction = ac.predict(gripperInt, action)
                        score = self.target.score(prediction)*ac.avgPrediction
                        print "abstract case: ", ac.variables
                        print "ac avgPrediction: ", ac.avgPrediction
                        print "numPredictions: ", ac.numPredictions
                        print "predicted pos: ", prediction["spos"]
                        print "score to target: {}, for action: {}".format(score, action)                    
                        if score > bestScore:
                            bestScore = score
                            bestAction = action
    #                        bestAction["cmd"] = 1
            if bestAction != None:
                print "using Action: ", bestAction
                return bestAction
        return self.getRandomAction()
            
    def getRandomAction(self):
        print "getting random action"
        rnd = np.random.rand()
        a = Action()
        if rnd < 0.3:
            a["cmd"] = GZCMD["MOVE"]
#            a["dir"] = np.array([1.2,0,0])
            a["mvDir"] = np.random.rand(3)*2-1
        elif rnd < 0.4:
            a["cmd"] = GZCMD["MOVE"]
            a["mvDir"] = np.array([0,0,0])
        else:
            a["cmd"] = GZCMD["NOTHING"]
#        a["mvDir"] *= 2.0
        a["mvDir"][2] = 0
        return a
    
    def setTarget(self, postState = None):
        self.target = postState
    
    def getBestCase(self, state, action):
        bestCase = None
        bestScore = 0.0
        for c in self.abstractCases:
            s = c.score(state, action)
#            print "case: {}  has score: {}".format(c.variables,s)
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
        
        bestCase = self.getBestCase(state, action)
        if bestCase != None:
            return bestCase.predict(state, action), bestCase
        else:
#            print "using old state with id: ", state["intId"]
            return state, bestCase
    
    def predict(self, worldState, action):
        
        predictionWs = WorldState()
        for intState in worldState.interactionStates.values():
            self.numPredictions += 1
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
        attribSet = newCase.getSetOfAttribs()
        predictionScore = result.score(prediction)
        print "predictionScore: ", predictionScore
        if usedCase != None and usedCase.variables == attribSet:
            print "correct case selected"
            usedCase.updatePredictionScore(predictionScore)
            self.numCorrectCase += 1
        elif usedCase != None:
            usedCase.addErrorCase(newCase)
        if predictionScore < PREDICTIONTHRESHOLD:
            print "adding Case"
#            self.cases.append(newCase)
            abstractCase = None
            for ac in self.abstractCases:
                if ac.variables == attribSet:
                    abstractCase = ac
                    #TODO consider search for all of them in case we distinguis by certain features
                    break
            if abstractCase != None:
                #If an abstract case is found add the reference
                try:
                    abstractCase.addRef(newCase)
                except Exception, e:
                    print "case was already present"
#                    print "Responsible abstractCase: ", abstractCase.variables
#                    print "newCase: ", newCase
#                    raise Exception("case already present")
                else:
                    self.cases.append(newCase)
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
        