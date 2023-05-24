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

from ._sindy import SINDy

MPI_GLOBAL_AVAILABILITY = True

try:
    from mpi4py import MPI
except:
    MPI_GLOBAL_AVAILABILITY = False
    warnings.warn(f"Trying to import MPI in {__file__}.")
    warnings.warn(
        "mpi4py is not installed. If you want to execute MPI jobs, we recommend you install it."
    )

import numpy as np
from scipy.optimize import brentq
import numpy as np
import itertools
import operator
from scipy.linalg import lstsq
from numpy import matlib as mb
import scipy 
from scipy.integrate import solve_ivp
#from kneed import KneeLocator
from scipy.integrate import odeint


#from .integration import RKF78
#from .utilities import differentiate
#from .lyapunov import LyapunovEstimator
#from .plot import plot_latent

import matplotlib.pyplot as plt

class WeakSINDy(SINDy):
    """
    Reimplementation of SINDy class to account for waek formulation 
    and pysindy features

     Inputs:
       polys: monomial powers to include in library
       trigs: sine / cosine frequencies to include in library
       scale_theta: normalize columns of theta. Ex: scale_theta = 0 means no normalization, scale_theta = 2 means l2 normalization.
       ld: sequential thresholding parameter 
       gamma: Tikhonoff regularization parameter
    """
    def __init__(
        self, 
        bias_rescale: float = 1,
        feature_library: object = None, 
        solver : Union[str, callable] = "lstsq",
        engine : str = "numpy",
        test_function: object = None,
        ):

        self.polys = np.arange(0, 6)
        self.trigs = []
        self.scale_theta = 0

        self.ld = 0.001
        self.gamma = 10**(-np.inf)
        self.coef = None
        self.multiple_trajectories = False
        self.useGLS = 10**(-12)

        self.feature_library = feature_library
        self.test_function = test_function
        self.solver = solver

        # lyapunov estimator
        self.LE = None

        self.x_predicted = None
        self.t_predicted = None

        self.eval_op = self._eval

    def print(self, weights=None, precision=2, names=None):

        if weights is None:
            weights = self.coef

        #if names is None:
        #    names = [f'x{i}' for i in range(weights.shape[1])]
        vars = [f'x{i}' for i in range(weights.shape[1])]
        if names is not None:
            new_vars = names
        else:
            new_vars = vars
        names = self.feature_library.get_feature_names_out(names)
        
        for i, wi in enumerate(weights.T):
            
            expr = [f'{wij:+.{precision}f} {n.replace(" ","*")}' for n,wij in zip(names,wi) if f'{abs(wij):.{precision}f}' != f'{abs(0.00):.{precision}f}']
            expr = [e if not e.split(' ')[-1].isnumeric() else ''.join(e.split(' ')[:-1]) for e in expr]
            expr = [e.replace('-','- ') if i > 0 else e.removeprefix('+') for i,e in enumerate(expr)]
            expr = [e.replace('+','+ ') if i > 0 else e.removeprefix('+') for i,e in enumerate(expr)]
            expr = ' '.join(expr)
            expr = f'd{vars[i]}_dt = {expr}'
            
            for var, new_var in zip(vars,new_vars):
                expr = expr.replace(var, new_var)
                
            print(expr)

    def build_b(self, Vp, x, RT):

        integral = Vp.dot(x) 
        #for k in range(len(integral)):
        #    integral[k] += self.test_function.basis(k)(-1)*x[-1] - #self.test_function.basis(k)(0)*x[0]

        if self.useGLS > 0:
            b = lstsq(RT, integral)[0]
        else:
            temp = integral
            b = RT.T*temp

        # ridiculous simple
        #b = Vp @ X[:,i]
        return b

    def build_G(self, V, Theta_0, RT):

        integral = V.dot(Theta_0)

        if self.useGLS > 0:
            G = lstsq(RT, integral )[0]        
        else:
            G = np.multiply(integral, RT)
        # ridiculous simple
        # G = V @ Theta_0
        return G
    
    def predict(self, x0):
        if len(x0.shape) == 1:
            solution = self.eval(x0)
        else:
            solution = np.array([self.eval(xi) for xi in x0])
        return solution
    
    def fit(
        self, 
        input_data: np.ndarray = None,
        target_data: np.ndarray = None, 
        ) -> None:
        """
        Fit the model to data.

        :param input_data: the input data (snapshots)
        :type input_data: np.ndarray
        :param target_data: the target data (time)
        :type target_data: np.ndarray
        """

        self.x = input_data
        self.t = target_data

        #X, T = self.get_multiple_trajectories(X, T)

        M = len(self.t)

        Theta_0 = self.build_Theta(self.x)
        #grid = self.test_function.grid(T)

        if self.test_function.sampling == 'uniform':
            V, Vp, grid = self.test_function.build_V_Vp(t=self.t)
            
            if self.useGLS > 0:
                Cov = Vp.dot(Vp.T) + self.useGLS*np.identity(V.shape[0])
                RT = np.linalg.cholesky(Cov)
            else:
                RT = 1/np.linalg.norm(Vp, 2, 1)
                RT = np.reshape(RT, (RT.size, 1))

            G = self.build_G(V, Theta_0, RT)

        def find_max_L(data, L):
            max_freq = np.array([np.argmax( np.fft.rfft(signal) ) for signal in data.T])
            if 0 in max_freq:
                print('insufitient data for multiscale')
                max_freq[max_freq == 0] = 1
            min_freq = np.min(max_freq)
            min_N = np.ceil(2.5*min_freq)
            min_L = int(np.floor(data.shape[0]/min_N))
            normalized_freq = max_freq/min_freq
            Lnew = np.array(np.floor(1/normalized_freq*min_L), dtype=int)
            return Lnew

        if self.test_function.sampling == 'uniform_multiscale':
            Ls = find_max_L(self.x, self.test_function.L)

        n = self.x.shape[1]
        w_sparse = np.zeros((Theta_0.shape[1], n))
        mats = []  
        ts_grids = []  

        if self.scale_theta > 0:
            M_diag = np.linalg.norm(Theta_0, self.scale_theta, 0)
            M_diag = np.reshape(M_diag, (len(M_diag), 1))
        else:
            M_diag = np.array([])

        bs = []
        for i in range(n):
            if self.test_function.sampling == 'uniform_multiscale':
                self.test_function.L = Ls[i]
                V, Vp, grid = self.test_function.build_V_Vp(self.t)

                if self.useGLS > 0:
                    Cov = Vp.dot(Vp.T) + self.useGLS*np.identity(V.shape[0])
                    RT = np.linalg.cholesky(Cov)
                else:
                    RT = 1/np.linalg.norm(Vp, 2, 1)
                    RT = np.reshape(RT, (RT.size, 1))

                G = self.build_G(V, Theta_0, RT)

            elif self.test_function.sampling == 'adaptive':
                V, Vp, grid = self.test_function.build_V_Vp(t=self.t, x=self.x[:,i])

                if self.useGLS > 0:
                    Cov = Vp.dot(Vp.T) + self.useGLS*np.identity(V.shape[0])
                    RT = np.linalg.cholesky(Cov)
                else:
                    RT = 1/np.linalg.norm(Vp, 2, 1)
                    RT = np.reshape(RT, (RT.size, 1))

                G = self.build_G(V, Theta_0, RT)

            mats.append([V, Vp])
            ts_grids.append(grid)

            b = self.build_b( Vp, self.x[:,i], RT)
            
            if self.scale_theta > 0:     
                w_sparse_temp = self.solve_linear_system(
                    np.multiply(G, (1/M_diag.T)), b.T)
                w_sparse[:, i] = np.ndarray.flatten(
                    np.multiply((1/M_diag.T), w_sparse_temp))
            else:
                w_sparse_temp = self.solve_linear_system(G, b.T)
                w_sparse[:, i] = np.ndarray.flatten(w_sparse_temp)
            
            bs.append(b)

        self.coef = w_sparse
        self.mats = mats
        self.ts_grids = ts_grids
        self.G = G
        self.b = b
        #self.bs = np.array(bs)

    def build_Theta(self, X):
        self.Theta_0 = self.feature_library.fit_transform(X)
        return self.Theta_0

    def get_multiple_trajectories(self, xobs, tobs):
        
        if self.multiple_trajectories == True:
            x_values = xobs[0]
            t_values = tobs[0]
            for i in range(1, len(xobs)):
                x_values = np.vstack((x_values, xobs[i]))
                t_values = np.hstack((t_values, tobs[i]))
            xobs = x_values
            tobs = t_values

        return xobs, tobs

    
    def get_uniform_grid(self, t, L, overlap):
        M = len(t)
        # polynomial degree, should vary
        p = self.p
        s = overlap
        n_overlap = int(np.floor(L*(1-np.sqrt(1-s**(1/p)))))
        
        a = 0
        b = L
        grid = [[a, b]]

        while b - n_overlap + L <= M-1:
            a = b - n_overlap
            b = a + L
            grid.append([a, b])

        return np.asarray(grid)

    def integration_vectors_adaptive(self, g, gp, t, a, b, param=[1, 1, 0]):
        if param == None:
            gap = 1
            nrm = np.inf
            ord = 0
        else:
            gap = param[0]
            nrm = param[1]
            ord = param[2]
        
        if a > b:
            a_temp = b
            b = a
            a = a_temp

        t_grid = t[a:b+1:gap] 
        dts = np.diff(t_grid)
        w = 1/2*(np.append(dts, [0]) + np.append([0], dts))

        V_row = g(t_grid, t[a], t[b])*w
        Vp_row = -gp(t_grid, t[a], t[b])*w

        if ord == 0:
            scale_fac = np.linalg.norm(
                np.ndarray.flatten(V_row), nrm)
        elif ord == 1:
            scale_fac = np.linalg.norm(
                np.ndarray.flatten(Vp_row), nrm)
        else:
            scale_fac = np.mean(dts)
        Vp_row = Vp_row/scale_fac
        V_row = V_row/scale_fac
        return V_row, Vp_row

    def integration_vectors(self, g, gp, t, a, b, param=[0, np.inf, 0]):
        if param == None:
            pow = 1
            gap = 1
            nrm = np.inf
            ord = 0
        else:
            pow = param[0]
            nrm = param[1]
            ord = param[2]
            gap = 1
        
        if a > b:
            a_temp = b
            b = a
            a = a_temp


        t_grid = t[a:b+1:gap] 
        dts = np.diff(t_grid)
        w = 1/2*(np.append(dts, [0]) + np.append([0], dts))

        V_row = g(t_grid, t[a], t[b])*w
        Vp_row = -gp(t_grid, t[a], t[b])*w

        if pow != 0:
            if ord == 0:
                scale_fac = np.linalg.norm( np.ndarray.flatten(V_row), nrm)
            elif ord == 1:
                scale_fac = np.linalg.norm( Vp_row, nrm)
            else:
                scale_fac = np.mean(dts)
            V_row = V_row/scale_fac
            Vp_row = Vp_row/scale_fac
        return V_row, Vp_row

    def build_V_Vp(self, t, grid):
        p = self.p
        M = len(t)
        N = len(grid)

        V = np.zeros((N, M))
        Vp = np.zeros((N, M))

        g, gp = self.test_function()
        for k in range(N):
            #g, gp = self.basis_fcn(p, p)
            #g, gp = self.get_basis_function()
            #g = self.g
            #gp = self.gp
            
            a = grid[k][0]
            b = grid[k][1]
            V_row, Vp_row = self.integration_vectors(g, gp, t, a, b)
            V[k, a:b+1] = V_row
            Vp[k, a:b+1] = Vp_row
    
        return V, Vp
    
    def solve_linear_system(self, G, b):
        if isinstance(self.solver, str):
            if self.solver.upper() == 'PINV':
                w = np.linalg.pinv(G) @ b
            elif self.solver.upper() == 'LSTSQ':
                w = np.linalg.lstsq(G, b)[0]
        else:
            self.solver.fit(G, b)
            w = self.solver.coef_
        return w
    """
    def simulate(self, x0, t_span, t_eval):
        #print(self.tags)
        #print(self.coef)

        rows, cols = self.tags.shape
        tol_ode = 10**(-14)
        def rhs(t, x):
            term = np.ones(rows)
            for row in range(rows):
                for col in range(cols): 
                    term[row] = term[row]*x[col]**self.tags[row, col]
            return term.dot(self.coef)

        #print(len(t_eval))
        sol = solve_ivp(fun = rhs, t_eval=t_eval, t_span=t_span, y0=x0, rtol=tol_ode)
        return sol.y.T
    """

    def Uniform_grid(self, t, L, s, param):
        M = len(t)
        #p = int(np.floor(1/8*((L**2*rho**2 - 1) + np.sqrt((L**2*rho**2 - 1)**2 - 8*L**2*rho**2))))
        p = 16

        overlap = int(np.floor(L*(1 - np.sqrt(1 - s**(1/p)))))
        #print("support and overlap", L, overlap)

        # create grid
        grid = []
        a = 0
        b = L
        grid.append([a, b])
        while b - overlap + L <= M-1:
            a = b - overlap
            b = a + L
            grid.append([a, b])

        grid = np.asarray(grid)
        N = len(grid)
        
        V = np.zeros((N, M))
        Vp = np.zeros((N, M))

        for k in range(N):
            g, gp = self.test_function()
            a = grid[k][0]
            b = grid[k][1]
            V_row, Vp_row = self.tf_mat_row(g, gp, t, a, b, param)
            V[k, :] = V_row
            Vp[k, :] = Vp_row

        return V, Vp, grid

    
    def tf_mat_row(self, g, gp, t, t1, tk, param):
        N = len(t)
        if param == None:
            pow = 1
            gap = 1
            nrm = np.inf
            ord = 0
        else:
            pow = param[0]
            nrm = param[1]
            ord = param[2]
            gap = 1

        if t1 > tk:
            tk_temp = tk
            tk = t1
            t1 = tk_temp

        V_row = np.zeros((1, N))
        Vp_row = np.copy(V_row)

        t_grid = t[t1:tk+1:gap]
        dts = np.diff(t_grid)
        w = 1/2*(np.append(dts, [0]) + np.append([0], dts))

        V_row[:, t1:tk+1:gap] = g(t_grid, t[t1], t[tk])*w
        Vp_row[:, t1:tk+1:gap] = -gp(t_grid, t[t1], t[tk])*w
        Vp_row[:, t1] = Vp_row[:, t1] - g(t[t1], t[t1], t[tk])
        Vp_row[:, tk] = Vp_row[:, tk] + g(t[tk], t[t1], t[tk])

        if pow != 0:
            if ord == 0:
                scale_fac = np.linalg.norm(
                    np.ndarray.flatten(V_row[:, t1:tk+1:gap]), nrm)
            elif ord == 1:
                scale_fac = np.linalg.norm(
                    np.ndarray.flatten(Vp_row[:, t1:tk+1:gap]), nrm)
            else:
                scale_fac = np.mean(dts)
            Vp_row = Vp_row/scale_fac
            V_row = V_row/scale_fac
        return V_row, Vp_row

    def _get_adaptive_grid(self, t, xobs, params):
        if params == None:
            index_gap = 16
            K = max(int(np.floor(len(t))/50), 4)
            p = 2
            tau = 1
        else:
            index_gap = params[0]
            K = params[1]
            p = params[2]
            tau = params[3]

        M = len(t)
        #g = self.g
        #gp = self.gp
        g, gp = self.test_function()
        Vp = self.build_Vp_adaptive(g, gp, t, index_gap)

        weak_der = Vp.dot(xobs)
        weak_der = np.append(np.zeros((int(np.floor(index_gap/2)), 1)), weak_der)
        weak_der = np.append(weak_der, np.zeros((int(np.floor(index_gap/2)), 1)))

        Y = np.abs(weak_der)
        Y = np.cumsum(Y)
        Y = Y/Y[-1]

        Y = tau*Y + (1-tau)*np.linspace(Y[0], Y[-1], len(Y)).T

        temp1 = Y[int(np.floor(index_gap/2)) - 1]
        temp2 = Y[int(len(Y) - np.ceil(index_gap/2)) - 1]
        U = np.linspace(temp1, temp2, K+2)

        final_grid = np.zeros((1, K))

        for i in range(K):
            final_grid[0, i] = np.argwhere((Y-U[i+1] >= 0))[0]

        final_grid = np.unique(final_grid)

        return final_grid

    def build_Vp_adaptive(self, g, gp, t, index_gap):
        M = len(t)
        #o, Vp_row = self.AG_tf_mat_row(g, gp, t, 1, 1+index_gap, [1, 1, 0])
        _, Vp_row = self.integration_vectors_adaptive(g, gp, t,1 , 1+index_gap, [1, 1, 0])
        Vp_diags = mb.repmat(Vp_row[0:index_gap+1], M - index_gap, 1)
        Vp = scipy.sparse.diags(Vp_diags.T, np.arange(
            0, index_gap+1), (M-index_gap, M))

        return Vp

    def Adaptive_Grid(self, t, xobs, params=None):
        if params == None:
            index_gap = 16
            K = max(int(np.floor(len(t))/50), 4)
            p = 2
            tau = 1
        else:
            index_gap = params[0]
            K = params[1]
            p = params[2]
            tau = params[3]

        M = len(t)
        g, gp = self.basis_fcn(p, p)

        o, Vp_row = self.AG_tf_mat_row(g, gp, t, 1, 1+index_gap, [1, 1, 0])
        Vp_diags = mb.repmat(Vp_row[:, 0:index_gap+1], M - index_gap, 1)
        Vp = scipy.sparse.diags(Vp_diags.T, np.arange(
            0, index_gap+1), (M-index_gap, M))
        weak_der = Vp.dot(xobs)
        weak_der = np.append(np.zeros((int(np.floor(index_gap/2)), 1)), weak_der)
        weak_der = np.append(weak_der, np.zeros((int(np.floor(index_gap/2)), 1)))

        Y = np.abs(weak_der)
        Y = np.cumsum(Y)
        Y = Y/Y[-1]

        Y = tau*Y + (1-tau)*np.linspace(Y[0], Y[-1], len(Y)).T

        temp1 = Y[int(np.floor(index_gap/2)) - 1]
        temp2 = Y[int(len(Y) - np.ceil(index_gap/2)) - 1]
        U = np.linspace(temp1, temp2, K+2)

        final_grid = np.zeros((1, K))

        for i in range(K):
            final_grid[0, i] = np.argwhere((Y-U[i+1] >= 0))[0]

        final_grid = np.unique(final_grid)
        #print("length grid", len(final_grid))
        return final_grid #y

    def get_adaptive_grid(self, t, xobs, params=None):
        if params == None:
            index_gap = 16
            K = max(int(np.floor(len(t))/50), 4)
            p = 2
            tau = 1
        else:
            index_gap = params[0]
            K = params[1]
            p = params[2]
            tau = params[3]

        M = len(t)
        g, gp = self.basis_fcn(p, p)
        #g = self.basis_function(p,p)
        #gp = differentiate(g)
        o, Vp_row = self.AG_tf_mat_row(g, gp, t, 1, 1+index_gap, [1, 1, 0])
        Vp_diags = mb.repmat(Vp_row[:, 0:index_gap+1], M - index_gap, 1)
        Vp = scipy.sparse.diags(Vp_diags.T, np.arange(
            0, index_gap+1), (M-index_gap, M))
        weak_der = Vp.dot(xobs)
        weak_der = np.append(np.zeros((int(np.floor(index_gap/2)), 1)), weak_der)
        weak_der = np.append(weak_der, np.zeros((int(np.floor(index_gap/2)), 1)))

        Y = np.abs(weak_der)
        Y = np.cumsum(Y)
        Y = Y/Y[-1]

        Y = tau*Y + (1-tau)*np.linspace(Y[0], Y[-1], len(Y)).T

        temp1 = Y[int(np.floor(index_gap/2)) - 1]
        temp2 = Y[int(len(Y) - np.ceil(index_gap/2)) - 1]
        U = np.linspace(temp1, temp2, K+2)

        final_grid = np.zeros((1, K))

        for i in range(K):
            final_grid[0, i] = np.argwhere((Y-U[i+1] >= 0))[0]

        final_grid = np.unique(final_grid)
        #print("length grid", len(final_grid))
        return final_grid #y


    def AG_tf_mat_row(self, g, gp, t, t1, tk, param=None):
        N = len(t)

        if param == None:
            gap = 1
            nrm = np.inf
            ord = 0
        else:
            gap = param[0]
            nrm = param[1]
            ord = param[2]

        if t1 > tk:
            tk_temp = tk
            tk = t1
            t1 = tk_temp

        V_row = np.zeros((1, N))
        Vp_row = np.copy(V_row)

        #print(t1, tk, gap)
        t_grid = t[t1:tk+1:gap]

        dts = np.diff(t_grid)
        w = 1/2*(np.append(dts, [0]) + np.append([0], dts))

        V_row[:, t1:tk+1:gap] = g(t_grid, t[t1], t[tk])*w
        Vp_row[:, t1:tk+1:gap] = -gp(t_grid, t[t1], t[tk])*w
        Vp_row[:, t1] = Vp_row[:, t1] - g(t[t1], t[t1], t[tk])
        Vp_row[:, tk] = Vp_row[:, tk] + g(t[tk], t[t1], t[tk])

        if ord == 0:
            scale_fac = np.linalg.norm(
                np.ndarray.flatten(V_row[:, t1:tk+1:gap]), nrm)
        elif ord == 1:
            scale_fac = np.linalg.norm(
                np.ndarray.flatten(Vp_row[:, t1:tk+1:gap]), nrm)
        else:
            scale_fac = np.mean(dts)
        Vp_row = Vp_row/scale_fac
        V_row = V_row/scale_fac
        return V_row, Vp_row

    def build_V_Vp_adaptive(self, t, centers, r_whm, param=None):
        if param == None:
            param = [1, 2, 1]

        N = len(t)
        M = len(centers)
        V = np.zeros((M, N))
        Vp = np.zeros((M, N))
        ab_grid = np.zeros((M, 2))
        ps = np.zeros((M, 1))
        p, a, b = self.test_fcn_param(r_whm, t[int(centers[0]-1)], t)

        a = int(a)
        b = int(b)

        if b-a < 10:
            center = (a+b)/2
            a = int(max(0, np.floor(center-5)))
            b = int(min(np.ceil(center+5), len(t)))

        #g = self.basis_function(p,p)
        #gp = differentiate(g)
        g, gp = self.basis_fcn(p, p)
        V_row, Vp_row = self.tf_mat_row(g, gp, t, a, b, param)

        V[0, :] = V_row
        Vp[0, :] = Vp_row
        ab_grid[0, :] = np.array([a, b])
        ps[0] = p

        for k in range(1, M):
            cent_shift = int(centers[k] - centers[k-1])
            b_temp = min(b + cent_shift, len(t))

            if a > 0 and b_temp < len(t):
                a = a + cent_shift
                b = b_temp
                V_row = np.roll(V_row, cent_shift)
                Vp_row = np.roll(Vp_row, cent_shift)
            else:
                p, a, b = self.test_fcn_param(
                    r_whm, t[int(centers[k]-1)], t)
                a = int(a)
                b = int(b)
                if b-a < 10:
                    center = (a+b)/2
                    b = int(min(np.ceil(center+5), len(t)))
                    a = int(max(0, np.floor(center-5)))
                #g = self.basis_function(p,p)
                #gp = differentiate(g)
                g, gp = self.basis_fcn(p, p)
                V_row, Vp_row = self.tf_mat_row(g, gp, t, a, b, param)

            V[k, :] = V_row
            Vp[k, :] = Vp_row

            ab_grid[k, :] = np.array([a, b])
            ps[k] = p
        return V, Vp, ab_grid, # ps

    def _build_V_Vp_adaptive(self, t, centers, r_whm, param):
        if param == None:
            param = [1, 2, 1]

        N = len(t)
        M = len(centers)
        V = np.zeros((M, N))
        Vp = np.zeros((M, N))
        ab_grid = np.zeros((M, 2))

        ps = np.zeros((M, 1))
        p, a, b = self.test_fcn_param(r_whm, t[int(centers[0]-1)], t)

        a = int(a)
        b = int(b)

        if b-a < 10:
            center = (a+b)/2
            a = int(max(0, np.floor(center-5)))
            b = int(min(np.ceil(center+5), len(t)))

        #g = self.basis_function(p,p)
        #gp = differentiate(g)
        g, gp = self.basis_fcn(p, p)
        V_row, Vp_row = self.integration_vectors(g, gp, t, a, b, param)

        V[0, a:b+1] = V_row
        Vp[0, a:b+1] = Vp_row
        ab_grid[0, :] = np.array([a, b])
        ps[0] = p

        for k in range(1, M):
            cent_shift = int(centers[k] - centers[k-1])
            b_temp = min(b + cent_shift, len(t))

            if a > 0 and b_temp < len(t):
                a = a + cent_shift
                b = b_temp
                V_row = np.roll(V_row, cent_shift)
                Vp_row = np.roll(Vp_row, cent_shift)
            else:
                p, a, b = self.test_fcn_param(
                    r_whm, t[int(centers[k]-1)], t)
                a = int(a)
                b = int(b)
                if b-a < 10:
                    center = (a+b)/2
                    b = int(min(np.ceil(center+5), len(t)))
                    a = int(max(0, np.floor(center-5)))
                #g = self.basis_function(p,p)
                #gp = differentiate(g)
                g, gp = self.basis_fcn(p, p)
                V_row, Vp_row = self.tf_mat_row(g, gp, t, a, b, param)

                V[k, :] = V_row
                Vp[k, :] = Vp_row

                ab_grid[k, :] = np.array([a, b])
        return V, Vp, ab_grid

    def VVp_build_adaptive_whm(self, t, centers, r_whm, param=None):
        if param == None:
            param = [1, 2, 1]

        N = len(t)
        M = len(centers)
        V = np.zeros((M, N))
        Vp = np.zeros((M, N))
        ab_grid = np.zeros((M, 2))
        ps = np.zeros((M, 1))
        p, a, b = self.test_fcn_param(r_whm, t[int(centers[0]-1)], t)

        a = int(a)
        b = int(b)

        if b-a < 10:
            center = (a+b)/2
            a = int(max(0, np.floor(center-5)))
            b = int(min(np.ceil(center+5), len(t)))

        g, gp = self.basis_fcn(p, p)
        V_row, Vp_row = self.tf_mat_row(g, gp, t, a, b, param)

        V[0, :] = V_row
        Vp[0, :] = Vp_row
        ab_grid[0, :] = np.array([a, b])
        ps[0] = p

        for k in range(1, M):
            cent_shift = int(centers[k] - centers[k-1])
            b_temp = min(b + cent_shift, len(t))

            if a > 0 and b_temp < len(t):
                a = a + cent_shift
                b = b_temp
                V_row = np.roll(V_row, cent_shift)
                Vp_row = np.roll(Vp_row, cent_shift)
            else:
                p, a, b = self.test_fcn_param(
                    r_whm, t[int(centers[k]-1)], t)
                a = int(a)
                b = int(b)
                if b-a < 10:
                    center = (a+b)/2
                    b = int(min(np.ceil(center+5), len(t)))
                    a = int(max(0, np.floor(center-5)))
                g, gp = self.basis_fcn(p, p)
                V_row, Vp_row = self.tf_mat_row(g, gp, t, a, b, param)

            V[k, :] = V_row
            Vp[k, :] = Vp_row

            ab_grid[k, :] = np.array([a, b])
            ps[k] = p
        return V, Vp, ab_grid, # ps


    def sparsifyDynamics(self, Theta, dXdt, n, M=None):
        if M is None:
            M = np.ones((Theta.shape[1], 1))

        if self.gamma == 0:
            Theta_reg = Theta
            dXdt_reg = np.reshape(dXdt, (dXdt.size, 1))
        else:
            nn = Theta.shape[1]
            Theta_reg = np.vstack((Theta, self.gamma*np.identity(nn)))
            dXdt = np.reshape(dXdt, (dXdt.size, 1))
            dXdt_reg_temp = np.vstack((dXdt, self.gamma*np.zeros((nn, n))))
            dXdt_reg = np.reshape(dXdt_reg_temp, (dXdt_reg_temp.size, 1))
            #print(nn)
        
        #print("theta", Theta_reg.shape)
        #print("dXdt_reg", dXdt_reg.shape)

        Xi = M*(lstsq(Theta_reg, dXdt_reg)[0])

        for i in range(10):
            smallinds = (abs(Xi) < self.ld)
            while np.argwhere(np.ndarray.flatten(smallinds)).size == Xi.size:
                self.ld = self.ld/2
                smallinds = (abs(Xi) < self.ld)
            Xi[smallinds] = 0
        for ind in range(n):
            biginds = ~smallinds[:, ind]
            temp = dXdt_reg[:, ind]
            temp = np.reshape(temp, (temp.size, 1))
            Xi[biginds, ind] = np.ndarray.flatten(
                M[biginds]*(lstsq(Theta_reg[:, biginds], temp)[0]))
        #residual = np.linalg.norm((Theta_reg.dot(Xi)) - dXdt_reg)
        return Xi

    def buildTheta(self, xobs):
        theta_0, tags = self.poolDatagen(xobs)
        if self.scale_theta > 0:
            M_diag = np.linalg.norm(theta_0, self.scale_theta, 0)
            M_diag = np.reshape(M_diag, (len(M_diag), 1))
            return theta_0, tags, M_diag
        else:
            M_diag = np.array([])
            return theta_0, tags, M_diag

    def poolDatagen(self, xobs):
        # generate monomials
        n, d = xobs.shape
        if len(self.polys) != 0:
            P = self.polys[-1]
        else:
            P = 0
        rhs_functions = {}
        def f(t, x): return np.prod(np.power(list(t), list(x)))
        powers = []
        for p in range(1, P+1):
            size = d + p - 1
            for indices in itertools.combinations(range(size), d-1):
                starts = [0] + [index+1 for index in indices]
                stops = indices + (size,)
                powers.append(tuple(map(operator.sub, stops, starts)))
        for power in powers:
            rhs_functions[power] = [lambda t, x=power: f(t, x), power]

        theta_0 = np.ones((n, 1))
        #print(powers)

        tags = np.array(powers)
        #print('tags', tags)
        # plug in
        for k in rhs_functions.keys():
            func = rhs_functions[k][0]
            new_column = np.zeros((n, 1))
            for i in range(n):
                new_column[i] = func(xobs[i, :])
            theta_0 = np.hstack([theta_0, new_column])

        # trigs:
        for i in range(len(self.trigs)):
            trig_inds = np.array([-self.trigs[i]*1j*np.ones(d), self.trigs[i]*1j*np.ones(d)])
            sin_col = np.zeros((n, 1))
            cos_col = np.zeros((n, 1))
            for m in range(n):
                sin_col[m] = np.sin(self.trigs[i]*xobs[m, :])
                cos_col[m] = np.cos(self.trigs[i]*xobs[m, :])
            theta_0 = np.hstack([theta_0, sin_col, cos_col])
            tags = np.vstack([tags, trig_inds])

        tags = np.vstack([np.zeros((1, d)), tags])
        # print(tags)
        return theta_0, tags
    
    

    def test_fcn_param(self, r, c, t, p=None):
        if self.tau_p < 0:
            self.tau_p = - self.tau_p
        else:
            p = self.tau_p
            self.tau_p = 16
        dt = t[1]-t[0]
        r_whm = r*dt
        A = np.log2(10)*self.tau_p
        def gg(s): return -s**2*((1-(r_whm/s)**2)**A-1)
        def hh(s): return (s-dt)**2
        def ff(s): return hh(s)-gg(s)

        s = brentq(ff, r_whm, r_whm*np.sqrt(A)+dt)

        if p == None:
            p = min(np.ceil(max(-1/np.log2(1-(r_whm/s)**2), 1)), 200)

        a = np.argwhere((t >= (c-s)))
        if len(a) != 0:
            a = a[0]
        else:
            a = []

        if c+s > t[-1]:
            b = len(t)-1
        else:
            b = np.argwhere((t >= (c+s)))[0]
        return p, a, b

    def eval(self, input_data: np.ndarray = None, **kwargs) -> np.ndarray:
        """Evaluate the model at the given input data.

        Parameters
        ----------
        input_data : np.ndarray, optional
            Input data, by default None
        **kwargs
            Additional keyword arguments

        Returns
        -------
        np.ndarray
            Model output
        """

        if len(input_data.shape) == 1:
            solution = self.eval_op(input_data)
        else:
            solution = np.array([self.eval_op(xi) for xi in input_data])
        return solution

    def _eval(self, input_data: np.ndarray = None) -> np.ndarray:
        basis = self.feature_library.fit_transform(input_data[None,:])
        w = np.array(self.coef)
        dy = (basis @ w)
        return dy.flatten()

    def simulate(self, x0, t, integrator='odeint', integrator_args=dict()):
        # compose integration over derivatives
        #x0 = self.scale(x0, self.x_min, self.x_max, self.x_mean)
        self.t_predicted = t
        if integrator.upper() == 'ODEINT':
            odeint_eval = lambda x, t: self.eval(x)
            solution = odeint(odeint_eval, x0, t)
        elif integrator.upper() == 'RKF78':
            rkf78 = RKF78(**integrator_args)
            solution, t = rkf78.solve(self.eval, x0, t)
        #solution = self.descale(solution, self.x_min, self.x_max, self.x_mean)
        self.x_predicted = solution
        return solution, t

    def get_max_lyapunov_exponents(self, integrator='rkf78', integrator_args=dict()):
        if integrator.upper() == 'ODEINT':
            rhs = lambda x, t: self.eval(x)
            integrator = odeint    
        elif integrator.upper() == 'RKF78':
            rhs = self.eval
            integrator = RKF78(**integrator_args)

        LE = LyapunovEstimator(integrator, rhs)
        LE.get_max_lyapunov_exponents(self.x_predicted, self.t_predicted)
        self.LE = LE
        return LE.max_Lyapunov_exponents
    
    @property
    def max_lyapunov_exponent(self):
        if self.LE is not None:
            return np.max(self.LE.max_lyapunov_exponents)