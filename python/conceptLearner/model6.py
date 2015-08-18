# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:22:29 2015
Try to implement a top-down model...
@author: jpoeppel
"""

from model4 import State, ObjectState, InteractionState, WorldState, BaseCase, Action
from model4 import THRESHOLD, MAXCASESCORE, MAXSTATESCORE, PREDICTIONTHRESHOLD, TARGETTHRESHOLD
from common import GAZEBOCMDS as GZCMD
import model4
import numpy as np
from operator import methodcaller
from sets import Set
import copy
from topoMaps import ITM
from operator import methodcaller, itemgetter
from network import Node
from network import Tree
from sklearn import tree

from metrics import similarities


class AbstractCase(model4.AbstractCase):
    
    def __init__(self, case):
        assert isinstance(case, BaseCase), "case is not a BaseCase object."
        self.id = 0
        self.refCases = []
        self.avgPrediction = 0.0
        self.numPredictions = 0
        self.name = ""
        self.variables = Set() #List of attributes that changed 
        self.variables.update(case.getSetOfAttribs())
        self.constants = {}
        self.gaussians = {}
        self.weights= {}
        self.minima = {}
        self.maxima = {}
        for k in self.variables:
            self.weights[k] = 1.0
        self.addRef(case)
    
    def predict(self, state, action):
        print "self.weights: ", self.weights
        return self.weights
    
    def updateWeights(self, prediction, result):
        
        for k in self.variables:
            error = result[k] - prediction[k]
            self.weights[k] += 0.05 * error
            
#        print "AC: {}, weights: {}".format(self.variables, self.weights)

    def addRef(self, ref):
        
        if ref in self.refCases:
            raise TypeError("ref already in refCases")
#        
#        for k,v in ref.getListOfConstants() + ref.action.relevantItems():
        for k,v in ref.preState.relevantItems() + ref.action.relevantItems():
            if self.constants.has_key(k):
                if np.linalg.norm(v-self.constants[k]) > 0.001:
                    del self.constants[k]
            else:
                if len(self.refCases) == 0:
                    self.constants[k] = v
            if hasattr(v, "__len__"):
                ori = np.zeros(len(v))
            else:
                ori = 0
            distToOri = np.linalg.norm(v-ori)
            if not self.minima.has_key(k) or distToOri < self.minima[k]:
                self.minima[k] = distToOri
            if not self.maxima.has_key(k) or distToOri > self.maxima[k]:
                self.maxima[k] = distToOri
#            if not self.minima.has_key(k) or np.all(v < self.minima[k]):
#                self.minima[k] = v
#            if not self.maxima.has_key(k) or np.all(v > self.maxima[k]):
#                self.maxima[k] = v
        
        self.refCases.append(ref)
        ref.abstCase = self
        self.updateGaussians(self.gaussians, len(self.refCases), ref)     
#        return constChanged
#
class ModelCBR(object):
    
    def __init__(self):
        self.predictors = {}
        self.abstractCases = {}
        self.cases = []
        self.numACs = 0
        self.numPredictions = 0
        self.numCorrectCase = 0
        self.aCTree = Tree()
        self.aCClassifier = None
        self.scaler = None
        #TODO try building a decision tree for AC selection
        
    def getBestCase(self, state, action):
        
#        print "getBestCase with state: {} \n action: {}".format(state, action)
        bestCase = None
        if self.aCClassifier != None:
            x = [np.concatenate((state.toSelVec(),action.toSelVec()))]
#            print "X before scaling: ", x
            if self.scaler != None:
                x = self.scaler.transform(x)
#            print "X after sclaing: ", x
            caseID = int(self.aCClassifier.predict(x)[0])
#            print "CaseID: ", caseID
#            print "Case prob: ", self.aCClassifier.predict_proba(x)
            bestCase = self.abstractCases[caseID]
        else:
#            scoreList = [(c,c.score(state,action)) for c in self.abstractCases]
            scoreList = [(c.abstCase,c.score(state,action)) for c in self.cases]
            
    #        sortedList = sorted(self.abstractCases, key=methodcaller('score', state, action), reverse= True)
            sortedList = sorted(scoreList, key=itemgetter(1), reverse=True) 
    #        self.lastScorelist = [(s, sorted(c.variables), len(c.refCases)) for c,s in sortedList]
    #        if state["sid"] == 15:
#            print "ScoreList: ", [(s, sorted(c.variables), len(c.refCases)) for c,s in sortedList]
            if len(sortedList) > 0:
    #            if sortedList[0][1] == 0 and self.lastAC != None:
    #                bestCase = self.lastAC
    #            else:
                bestCase = sortedList[0][0]
                
        
        if isinstance(bestCase, AbstractCase):
#            print "selected AC: ", bestCase.variables
            if bestCase.variables == []:
                self.numZeroCase += 1

            return bestCase
        else:
            return None
            
    def predictIntState(self, state, action):
        
        bestCase = self.getBestCase(state, action)
        resultState = copy.deepcopy(state)
        if bestCase != None:
            weightDic = bestCase.predict(state, action)
            for k in weightDic:
                prediction = self.predictors[k].predict(np.concatenate((state.toVec(),action.toVec())))
                print "k: {}, pred: {}".format(k, prediction)
                if prediction != None:
                    resultState[k] = state[k] +  prediction #* weightDic[k] 
                else:
                    resultState[k] = state[k] + bestCase.refCases[0].predict(state, action, k)


        if resultState["sname"] == "gripper":
            print "Predicted gripper state: ", resultState
        return resultState, bestCase
    
    def predict(self, worldState, action):
        
        predictionWs = WorldState()
        predictionWs.transM = np.copy(worldState.transM)
        predictionWs.invTrans = np.copy(worldState.invTrans)
        predictionWs.ori = np.copy(worldState.ori)
        transformedAction = copy.deepcopy(action)
        transformedAction.transform(worldState.invTrans)
        for intState in worldState.interactionStates.values():
            self.numPredictions += 1
#            print "predicting for ", intState["intId"]
            prediction, usedCase = self.predictIntState(intState, action)
#            print "predicted intId: ", prediction["intId"]
            predictionWs.addInteractionState(prediction, usedCase)
#        print "resulting prediction: ", predictionWs.interactionStates
        return predictionWs
        
    def updateState(self, state, action, prediction, result, usedCase):
        retrain = False
        newCase = BaseCase(state, action, result)
        attribSet = newCase.getSetOfAttribs()
        for k in attribSet:
            if not self.predictors.has_key(k):
                self.predictors[k] = ITM()
            if similarities[k](result[k], prediction[k]) < 0.95:
                self.predictors[k].train(self.toNode(state,action,result, k))
                
        abstractCase = None
        for ac in self.abstractCases.values():
            if ac.variables == attribSet:
                abstractCase = ac
                #TODO consider search for all of them in case we distinguis by certain features
                break
        predictionScore = result.score(prediction)
        print "Prediction score: ", predictionScore
        if abstractCase != None:
            if predictionScore < PREDICTIONTHRESHOLD:
                abstractCase.updateWeights(prediction, result)
                try:
                    abstractCase.addRef(newCase)
                except TypeError:
                    print "Case already present."
                else:
                    self.cases.append(newCase)
            if usedCase != None:
                if usedCase.variables != attribSet:
                    retrain = True
                    constChanged = False
                    #TODO improve scoring
                    try:
                        print "ADDING REF"
                        constChanged = abstractCase.addRef(newCase)
                        print "adding new ref to AC: {}, new constants: {}".format(abstractCase.variables, abstractCase.constants)
                    except TypeError:
                        print "Case already present."
                    else:
                        self.cases.append(newCase)
                        
                    print "has constChanged? ", constChanged
                    if constChanged:
                        print "updating Tree"
#                        self.aCTree = Tree()
#                        self.aCTree.addElements(self.abstractCases.values())                        
#                        self.aCTree.removeElement(abstractCase)
#                        self.aCTree.addElement(abstractCase, abstractCase.constants, abstractCase.minima, abstractCase.maxima)
                    pass      
                else:
                    print "Correct case was selected!!!!"
                    self.numCorrectCase += 1
            
        else:
            print "Create new AC: ", attribSet
            abstractCase = AbstractCase(newCase)
            abstractCase.id = self.numACs
            self.cases.append(newCase)
            self.numACs += 1
            self.abstractCases[abstractCase.id] = abstractCase
            retrain = True
#            self.aCTree = Tree()
#            self.aCTree.addElements(self.abstractCases.values())
#
        if retrain:
            self.retrainACClassifier()


    def retrainACClassifier(self):
        print "Retraining!"
        if len(self.abstractCases) > 1:
            nFeature = np.size(np.concatenate((self.cases[0].preState.toSelVec(),self.cases[0].action.toSelVec())))
            X = np.zeros((len(self.cases),nFeature))
            Y = np.zeros(len(self.cases))
            for i in range(len(self.cases)):
                X[i,:] = np.concatenate((self.cases[i].preState.toSelVec(),self.cases[i].action.toSelVec()))
                Y[i] = self.cases[i].abstCase.id
#            self.scaler = preprocessing.StandardScaler(with_mean = False, with_std=True).fit(X)
#            self.scaler = preprocessing.MinMaxScaler().fit(X)
#            self.scaler = preprocessing.Normalizer().fit(X)
#            self.aCClassifier = svm.SVC(kernel='rbf', C=1, gamma=0.1)
#            self.aCClassifier = SGDClassifier(loss='log', penalty="l2")
            self.aCClassifier = tree.DecisionTreeClassifier(criterion="gini", class_weight='auto')#, min_samples_leaf=5) max_leaf_nodes=len(self.abstractCases))#, max_features='auto')
#            self.aCClassifier = RandomForestClassifier()
#            self.aCClassifier = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=4), n_estimators=50)
#            self.aCClassifier.fit(self.scaler.transform(X),Y)
            self.aCClassifier.fit(X,Y)
#                

            
    def toNode(self, state, action, result, attrib):
        node = Node(0, wIn=state.toVec(), action=action.toVec(),
                    wOut=result[attrib]-state[attrib])
        return node
            
    def update(self, worldState, action, prediction, result):
        transformedAction = copy.deepcopy(action)
        transformedAction.transform(worldState.invTrans)
        for intState in worldState.interactionStates.keys():
            self.updateState(worldState.interactionStates[intState], action, prediction.interactionStates[intState], 
                             result.interactionStates[intState], prediction.predictionCases[intState])