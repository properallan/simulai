# (C) Copyright IBM Corp. 2019, 2020, 2021, 2022.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#           http://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import warnings

from typing import List, Union, Tuple, Optional

import os

import numpy as np

MPI_GLOBAL_AVAILABILITY = True

try:
    from mpi4py import MPI
except:
    MPI_GLOBAL_AVAILABILITY = False
    warnings.warn(f"Trying to import MPI in {__file__}.")
    warnings.warn(
        "mpi4py is not installed. If you want to execute MPI jobs, we recommend you install it."
    )

from scipy.linalg import block_diag

from scipy.integrate import odeint

from simulai.math.integration import RKF78

def differential(x):
    dx = np.zeros_like(x)
    dx[1:-1, :] = (x[2:, :] - x[:-2, :])/2.
    dx[0,:]    = (x[1,:] - x[0,:])
    dx[-1,:]   = (x[-1,:] - x[-2,:])
    return dx

class SINDy:
    """the vanilla sindy of brunton"""
    def __init__(self, feature_library, solver):
        self.feature_library = feature_library
        self.solver = solver

    def fit(self, X, t):
        self.phi = self.phi_j(X)
        self.G = self.G_kj(self.phi, X)
        self.b = self.b_k(X, t)
        
        if self.solver == 'pinv':
            self.w = np.linalg.pinv(self.G) @ self.b
        elif self.solver == 'lstsq':
            self.w = np.linalg.lstsq(self.G, self.b)[0]
        return self.w

    def phi_j(self, X):
        phi = self.feature_library.fit_transform(X)
        return phi

    def G_kj(self, phi, X):
        return block_diag(*[phi]*X.shape[1])

    def b_k(self, X, t):
        dx = differential(X)/differential(t[:,None])
        b_k = dx.flatten(order='F')
        return b_k

    def eval(self, x0, t=None):
        if len(x0.shape) == 1:
            return self._eval(x0, t)
        elif len(x0.shape) == 2:
            return np.array([self._eval(x, t) for x in x0])
        
    def _eval(self, x0, t=None):

        basis = self.feature_library.fit_transform(x0[None,:])
        phi = block_diag(*[basis]*len(x0))
        dy = phi @ self.w
        return dy

    
    def simulate(self, x0, t, integrator='odeint', integrator_args=dict()):
        # compose integration over derivatives
        #x0 = self.scale(x0, self.x_min, self.x_max, self.x_mean)
        self.t_predicted = t
        if integrator.upper() == 'ODEINT':
            odeint_eval = lambda x, t: self.eval(x, t)
            solution = odeint(odeint_eval, x0, t)
        elif integrator.upper() == 'RKF78':
            rkf78 = RKF78(**integrator_args)
            solution, t = rkf78.solve(self.eval, x0, t)
        #solution = self.descale(solution, self.x_min, self.x_max, self.x_mean)
        self.x_predicted = solution
        return solution, t

    # Setting up model parameters
    def set(self, **kwargs):
        """Setting up extra parameters (as regularization terms)

        :param kwargs: dictionary containing extra parameters
        :type kwargs:dict
        :return: nothing
        """

        for key, value in kwargs.items():
            setattr(self, key, value)