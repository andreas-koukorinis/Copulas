from __future__ import division
from scipy.stats import kendalltau, pearsonr, spearmanr
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fmin
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



        def _ge_parameters(self):
            """ Estimate teh parameter(theta) of copula"""
            if self.family == "clayton":
                self.theta = 2*self.thau/(1-self.thau)

            elif self.family == 'frank':
                self.theta = -fmin(self._frank_fun, -5, disp=False)[0]

            elif self.family == 'gumbel':
                self.theta = 1/(1- self.tau)


        def generate_uv(self, n=1000):
            """Generate Random variables (u,v)"""

            #Clayton copula
            if self.family == 'clayton':
                U = np.random.uniform(size = n)
                V = np.random.uniform(size = n)

                if self.theta <= -1:
                    raise ValueError("The prameter for clayton copula should be more than -1")

                elif self.theta == 0:
                    raise ValueError("The parameter for clayton copula should not be 0")

                if self.theta < sys.float_info.epsilon:
                    V = W
                else:
                    V= U*(W**(-self.theta/(1+ self.theta))-1 + U**self.theta)**(-1/theta)


            #Frank Copula
            elif self.family = 'frank':
              U = np.random.uniform(size=n)
              W = np.random.uniform(size=n)

              if self.theta == 0:
                raise ValueError('The parameter for frank copula shoud not be 0')

              if abs(self.theta) > np.log(sys.float_info.max):
                V = (U <0) + np.sign(self.theta)*U
              elif abs(self.theta) > np.sqrt(sys.float_info.epsilon):
                a = np.exp(-self.theta*U)*(1-W)/W
                b = np.exp(-self.theta)
                c  =1+ np.exp(-self.theta*U)*(1-W)/W
                V= -np.log((a+b)/c)/self.theta
              else:
                V=W

              #Gumbell Copula
            elif self.family == 'gumbel':
              if self.theta <= 1:
                raise ValueError('the parameter for Gumbell copula should be greater than 1')

              if self.theta < 1 + sys.float_info.epsilon:
                U = np.random.uniform(size=n)
                V = np.random.uniform(size=n)
              else:
                u = np.random.uniform(size=n)
                v = np.random.uniform(size=n)
                w1 = np.random.uniform(size=n)
                w2 = np.random.uniform(size=n)

                u = (u-0.5)*np.pi
                u2 = u + np.pi/2
                e = -np.log(w)
                t =  np.cos(u- u2/self.theta)/e
                gamma = (np.sin(u2/self.theta)/t)**(1/self.theta)*np.cos(u)
                s1 = (-np.log(w1))**(1/self.theta)/gamma
                s2 = (-np.log(w2))**(1/self.theta)/gamma
                U = np.array(np.exp(-s1))
                V = np.array(np.exp(-s2))


            self.U = U
            self.V = V
            return U,V

        def _inverse_cdf(self):
            """ This module will calculate the inverse of CDF which wil be used to getting the ensembla
            of X and Y from the ensemble of U and V

            The statistics module is used to estimate teh CDF, which uses the kernel method of cdf estimation.

            To estimate the inverse of the CDF, interpolation method is used, first cdf is estiamted at 100 points,
            now interpolation function is generated to relate cdf at 100 points to the data.
            """

            # if U and V are nore already generated
            if self.U is None:
                self.generate_uv(n)

            #estimate teh inverse cdf of x and y
                     

            x2, x1 = st.cpdf(self.X, kernal="Epanechnikov", n=100)
            self._inverse_cdf_x = interp1d(x2,x1)

            y2, y1 = st.cpdf(self.Y, kernal="Epanechnikov", n=100)
            self._inverse_cdf_y = interp1d(y2,y1)

            X1 = self._inverse_cdf_x(self.U)
            Y1 = self._inverse_cdf_y(self.V)
            
            return X1, Y1

        def _intergrand_debye(self,t):
            """
            Integrand for the first order debye function
            """
            return t/(np.exp(t)-1)

        def _debye(self, alpha):
            """
            First order Dybye function
            """
            return quad(self._intergrand_debye, sys.float_info.epsilon, alpha)[0]/alpha

        def _frank_fun(self, alpha):
            """
            Optimisation of this fuction will give the parameter for the frank copula
            """
            diff= (1-self.tau)/4-(self._debye(1-alpha)-1)/alpha
            return diff**2


            










