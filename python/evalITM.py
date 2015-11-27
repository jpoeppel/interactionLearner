# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:32:37 2015

@author: jpoeppel
"""

import numpy as np
import os
import re
import matplotlib.pyplot as plt

def natSort(s, _nsr=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsr, s)]
    
    
directory = "../evalData/"


fileList = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]

def readITMInformatin():
    res = {}
    for fn in sorted(fileList, key=natSort):
        if fn.startswith(".") or "_config" in fn or not "ITMInformation" in fn:
            continue        
        
        if not "gate" in fn:
            continue
        
        if not "Mode2_Configuration_8" in fn:
            continue
        
        if not "Sigma005" in fn:
            continue
        
        if not "E20Fold" in fn:
            continue

        nameOnly = fn.replace('.txt','')
        parts = nameOnly.split("_")
        modelName = parts[0]
        targetConfig = int(parts[parts.index("TrainRuns")-1])
        currentConfig = int(parts[parts.index("Configuration")+1])   
        if "_E" in fn:
            experimentSuffix = parts[-1]
        else:
            experimentSuffix = ""
        experimentName = modelName+"_C_"+str(currentConfig)+"_Target_"+str(targetConfig)+experimentSuffix
        
        actUpdates = []
        actNodes = []
        gateUpdates = []
        gateNodes = []
        predUpdates = []
        predNodes = []
        acsUpdates = []
        acsNodes = []
        acUpdates = {}
        acNodes = {}
        numACs = []
        maxAcUpdates = []
        maxAcNodes = []
        curNumAcs = None
        curmaxACUpdates = None
        curMaxACNodes = None
        print "filename: ", directory+fn
        with open(directory+fn, "r") as f:
            for line in f:
                if line.startswith("Actuator ITM:"):
                    line = line.rstrip()
                    line = line.replace(",", "")
                    parts = line.split(" ")
                    actUpdates.append(int(parts[3]))
                    actNodes.append(int(parts[-1]))
                if line.startswith("Gate ITM:"):
                    line = line.rstrip()
                    line = line.replace(",", "")
                    parts = line.split(" ")
                    gateUpdates.append(int(parts[3]))
                    gateNodes.append(int(parts[-1]))
                if line.startswith("Object predictor ITM"):
                    line = line.rstrip()
                    line = line.replace(",", "")
                    parts = line.split(" ")
                    predUpdates.append(int(parts[7]))
                    predNodes.append(int(parts[-1]))
                if line.startswith("####"):
                    if curNumAcs != None:
                        numACs.append(curNumAcs)
                        maxAcUpdates.append(curmaxACUpdates)
                        maxAcNodes.append(curMaxACNodes)
                    curNumAcs = 0
                    curmaxACUpdates = 0
                    curMaxACNodes = 0
                if line.startswith("acSelector ITM:"):
                    line = line.rstrip()
                    line = line.replace(",", "")
                    parts = line.split(" ")
                    acsUpdates.append(int(parts[3]))
                    acsNodes.append(int(parts[-1]))
                if line.startswith("Abstract collection for"):
                    curNumAcs += 1
                    line = line.rstrip()
                    endName = line.find("]")+1
                    acName = line[line.find("["):endName]
                    line = line.replace(",", "")
                    parts = line[endName:].split(" ")
#                    print parts
                    if not acName in acUpdates:
                        acUpdates[acName] = []
                        acNodes[acName] = []
                    if int(parts[3]) > curmaxACUpdates:
                        curmaxACUpdates = int(parts[3])
                    if int(parts[3]) > curMaxACNodes:
                        curMaxACNodes = int(parts[-1])
                    acUpdates[acName].append(int(parts[3]))
                    acNodes[acName].append(int(parts[-1]))
        res[experimentName] = (actUpdates, actNodes, gateUpdates, gateNodes, predUpdates, 
                            predNodes, acsUpdates, acsNodes,acUpdates,acNodes, 
                            numACs, maxAcUpdates, maxAcNodes)
    return res
                    
if __name__ == "__main__":
    res = readITMInformatin()
    yactUp = []
    yactNodes=[] 
    gateUp = []
    gateNodes = [] 
    predUp= [] 
    predNodes = [] 
    for e in sorted(res.keys(), key=natSort):
        print e
        print "actUpdateMean: ", np.mean(res[e][0])
        print "actNodesMean: ", np.mean(res[e][1])
        print "gateUpdateMean: ", np.mean(res[e][2])
        print "gateNodesMean: ", np.mean(res[e][3])
        print "predUpdateMean: ", np.mean(res[e][4])
        print "predNodesMean: ", np.mean(res[e][5])
        
        print "actUpdateMean std: ", np.std(res[e][0])
        print "actNodesMean std: ", np.std(res[e][1])
        print "gateUpdateMean std: ", np.std(res[e][2])
        print "gateNodesMean std: ", np.std(res[e][3])
        print "predUpdateMean std: ", np.std(res[e][4])
        print "predNodesMean std: ", np.std(res[e][5])
        
        
        print "acsUpdatesMean: ", np.mean(res[e][6])
        print "acsNodesMean: ", np.mean(res[e][7])
        print "number of ACs: ", np.mean(res[e][10])
        print "maxACUpdateMean: ", np.mean(res[e][11])
        print "maxACNodesMean: ", np.mean(res[e][12])
        
        print "acsUpdatesMean std: ", np.std(res[e][6])
        print "acsNodesMean std: ", np.std(res[e][7])
        print "number of ACs std: ", np.std(res[e][10])
        print "maxACUpdateMean std: ", np.std(res[e][11])
        print "maxACNodesMean std: ", np.std(res[e][12])
        
        yactUp.append(np.mean(res[e][0]))
        yactNodes.append(np.mean(res[e][1]))
        gateUp.append(np.mean(res[e][2]))
        gateNodes.append(np.mean(res[e][3]))
        predUp.append(np.mean(res[e][4]))
        predNodes.append(np.mean(res[e][5]))
        
    
#    plt.plot([1,2,3,5,10,20,30], yactUp)
#    plt.plot([1,2,3,5,10,20,30], yactNodes)
#    plt.plot([1,2,3,5,10,20,30], gateUp, 'o')
#    plt.plot([1,2,3,5,10,20,30], gateNodes)
#    plt.plot([1,2,3,5,10,20,30], predUp)
#    plt.plot([1,2,3,5,10,20,30], predNodes)
    
    plt.show()