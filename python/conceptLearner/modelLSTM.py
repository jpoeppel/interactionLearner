#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:07:35 2015

@author: jpoeppel
"""

import lstm

class ModelLSTM(object):
    
    def __init__(self):
        self.net = lstm.LSTM1
        

    def predict(self, xs):
        """
        Function to query the model
        
        Parameters
        ----------
        xs: array numSeqElems x inDims
            Input sequence, can be one element
        """
        return self.net.predict(xs)
                
        
    def update(self, xs, ys):
        """
        Function to update the LSTM
        
        Parameters
        ----------
        
        xs: array numSeqElems x inDims
            Input sequence
        ys: array numSeqElems x outDims
            Target output sequence
        
        """
        preds = self.net.train(xs,ys)