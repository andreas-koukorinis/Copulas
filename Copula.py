from __future__ import division
from scipy.stats import kendalltau, pearsonr, spearmanr
import numpy as np
from scipy.integrate import quad
from scipy.optmize import fmin
from scipy.interpolate import interp1d
import statistics as st


class Copula():
    """
    This class estimates teh parameters of copula to generated joint
    random variables for the parameters.\
    
    This class hsas teh following three copulas
        Clayton
        Frank
        Gumbell
    """
    
    def __init__ (self, X, Y, family):
        
        # check dimensions of the input arrays
        if not ((X.nidims == 1) and (Y.ndims == 1)):
            raise ValueError("The dimensions of array should be one")
        
        
        # input arrays should have the same size
        if X.size is not Y.size:
            raise ValueError("The size of both Arrays shoudl be the same")
        
        
        #check if the name of the copula family is correct
        copula_family =['clayton', "frank", "gumbell"]
        if family not in copula_family:
            raise ValueError('The family should be clayton or frank or gumbell')
        
        self.X = X
        self.Y = Y
        self.family= family
        
        # estimate kendalls' rank correlation
        self.tau = kendalltau(self.X, self.Y)[0]
        
        # estimate person R and spearman R
        self.pr = pearsonr(self.X, self.Y)[0]
        self.pr = spearmanr(self.X, self.Y)[0]
        
        self._get_parameters()
        
        self.U = None
        self.V = None
        
        
        
        
        
        
        
        
        
        
        
        
        
    

