import sympy as sp
from simulai.math.integration import BRKF78, LSODA, RK4
from scipy import integrate
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import legendre
import numpy as np
from typing import Union
from pysindy import BaseOptimizer
class SSINDy:
    """
    Base Class for spectral SINDy (vanilla finite differences)
    """
    def __init__(
            self, 
            basis_function: object = PolynomialFeatures(degree=2), 
            solver: Union[str, BaseOptimizer] = 'pinv'
        ):
        
        self.basis_function = basis_function
        self.solver = solver

        self.predict = self.eval
        self.K = None

    def _basis_function(self, input_data):
        if len(input_data.shape) == 1:
            input_data = input_data[None]
        return self.basis_function.fit_transform(input_data)

    def build_G(self, input_data):
        return self._basis_function(input_data)
    
    def build_b(self, input_data, time):
        return np.gradient(input_data, time, axis=0)

    def fit(self, input_data, time):
        self.b = self.build_b(input_data, time)
        self.G = self.build_G(input_data)

        self.weights = self.solve_linear_system(self.G, self.b)
        
    def solve_linear_system(self, G, b):
        if isinstance(self.solver, str):
            #print(f'solver {self.solver}')
            if self.solver.upper() == 'PINV':
                w = np.linalg.pinv(G) @ b
            elif self.solver.upper() == 'LSTSQ':
                w = np.linalg.lstsq(G, b, rcond=None)[0]
        elif isinstance(self.solver, BaseOptimizer):
            print(f'solver {self.solver}')
            self.solver.fit(G, b)
            w = self.solver.coef_.T
        return w

    def eval(self, input_data, time=None):
        return (self.weights.T @ self._basis_function(input_data).T).T

    def simulate(self, initial_state, time, integrator='BRKF78'):
        integrator = eval(integrator)(right_operator=self)
        return integrator.run(initial_state, time)

    def print(self, weights=None, precision=2, names=None):

        if weights is None:
            weights = self.weights

        #if names is None:
        #    names = [f'x{i}' for i in range(weights.shape[1])]
        vars = [f'x{i}' for i in range(weights.shape[1])]
        if names is not None:
            new_vars = names
        else:
            new_vars = vars
        names = self.basis_function.get_feature_names_out(names)
        
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



class WeakSSINDy(SSINDy):
    """
    Weak Spectral SINDy

    """
    def __init__(self, 
            basis_function: object = PolynomialFeatures(degree=2),
            test_function: object = legendre,
            K : int = 10, 
            solver: Union[str, BaseOptimizer] = 'pinv'):
        
        self.basis_function = basis_function
        self.solver = solver
        self.test_function = test_function
        self.K = K
        

    def _test_function(self, input_data, K):
        Omega = np.linspace(-1,1, input_data.shape[0])
        Omega = input_data
        test = np.array([
            self.test_function(i)(Omega) for i in range(K) ])
        return test

    
    def _test_function_deriv(self, input_data, K):
        input_data = np.linspace(-1,1, input_data.shape[0])
        test = np.array([
            self.test_function(i).deriv()(input_data) for i in range(K) ])
        return test

    def build_G(self, input_data, time, K):
        Omega = np.linspace(-1,1, time.shape[0])
        test_functions = self._test_function(Omega, K)
        basis_functions = self._basis_function(input_data)
        J = self._basis_function(input_data[0]).shape[1]
        
        G = np.zeros((K,J))
        for k in range(K):
            for j in range(J):
                G[k,j] = integrate.simpson( basis_functions[:,j] * test_functions[k], time)#

        #        G[k,j] = integrate.simpson( y = basis_j* self.test_function(k)(Omega), x = time)
        return G
        #return ( self._test_function(time, K) @ self._basis_function(input_data))

    def fit(self, input_data, time):
        self.b = self.build_b(input_data, time, self.K)
        self.G = self.build_G(input_data, time, self.K)

        self.weights = self.solve_linear_system(self.G, self.b)
    
    def build_b(self, input_data, time, K):
        #Omega = time
        Omega = np.linspace(-1,1, time.shape[0])
        test_functions_deriv = self._test_function_deriv(Omega, K)
        test_functions = self._test_function(Omega, K)
        
        D = input_data.shape[1]
        b = np.zeros((K, D))
        for k in range(K):
            for d in range(D):
                
                #b[k, d] = - integrate.simpson( input_data[:,d] * test_functions_deriv[k], time) + input_data[-1,d]*test_functions[k][-1] - input_data[0,d]*test_functions[k][0]
 
                # working - but not clever(derivative of input_data)
                #b[k, d] = integrate.simpson( np.gradient(input_data[:,d], time) * test_functions[k], time) 


                # working - but need to improve the derivative computation of test functions
                #test_functions_derivative = np.gradient(test_functions[k], Omega) 
                test_functions_derivative = test_functions_deriv[k]
                b[k, d] = - integrate.simpson( input_data[:,d] * test_functions_derivative, Omega) + input_data[-1,d]*test_functions[k][-1] - input_data[0,d]*test_functions[k][0]
                

                #b[k, d] = - integrate.simpson( y = x_d * self.test_function(k).deriv()(Omega), x = time) + x_d[-1]*self.test_function(k)(Omega)[-1] - x_d[0]*self.test_function(k)(Omega)[0]
        return b
 
        
        #return  - (self._test_function_deriv(time, K) @ input_data) + (self._test_function(time, K)[:,-1,None] @ input_data[-1][None,:]) -(self._test_function(time, K)[:,0,None] @ input_data[0][None,:])

class SpectralWSINDy(WeakSSINDy):
    """New Implementation using Numpy instead of Sympy"""
    def __ini__(self, basis_function, test_function, solver):
        
        self.basis_function = basis_function
        self.test_function = test_function
        self.solver = solver

    def _test_function(self, input_data, K):
        test = np.array([
            self.test_function.basis(i)(input_data) for i in range(K) ])
        return test

    def _test_function_deriv(self, input_data, K):
        test = np.array([
            self.test_function.basis(i).deriv()(input_data) for i in range(K) ])
        return test

    def build_b(self, input_data, time):
        K = self.test_function.K

        Omega = np.linspace(-1,1, time.shape[0])
        test_functions_deriv = self._test_function_deriv(Omega, K)
        test_functions = self._test_function(Omega, K)

        D = input_data.shape[1]
        b = np.zeros((K, D))
        for k in range(K): 
            #test_function = self.test_function.fit(Omega, input_data[:,d], deg=K)
            for d in range(D):
                b[k, d] = - integrate.simpson( 
                    #input_data[:,d] * test_function.basis(k).deriv()(Omega)
                    #+ input_data[-1,d]*test_function.basis(k)(Omega[-1]) \
                    #- input_data[0,d]*test_function.basis(k)(Omega[0])
                    input_data[:,d] * test_functions_deriv[k] , Omega ) \
                    + input_data[-1,d]*test_functions[k][-1] \
                    - input_data[0,d]*test_functions[k][0]
        return b
    
    def build_G(self, input_data, time):
        K = self.test_function.K

        Omega = np.linspace(-1,1, time.shape[0])
        test_functions = self._test_function(Omega, K)
        basis_functions = self._basis_function(input_data)
        
        J = self._basis_function(input_data[0]).shape[1]
        G = np.zeros((K,J))

        for k in range(K):
            #test_function = self.test_function.fit(Omega, basis_functions[:,j], deg=K)
            for j in range(J):
                #G[k,j] = integrate.simpson( basis_functions[:,j] * test_function.basis(k)(Omega), time)
                G[k,j] = integrate.simpson( basis_functions[:,j] * test_functions[k], time)

        return G
    
    def fit(self, input_data, time):
        self.b = self.build_b(input_data, time)
        self.G = self.build_G(input_data, time)

        self.weights = self.solve_linear_system(self.G, self.b)

class MultiSpectralWSINDy(WeakSSINDy):
    """New Implementation using Numpy instead of Sympy"""
    def __ini__(self, basis_function, test_function, solver):
        
        self.basis_function = basis_function
        self.test_function = test_function
        self.solver = solver

    def build_b(self, input_data, time):
        D = input_data.shape[1]
        K = self.test_function.K
        b = np.zeros((K, D))
        Omega = np.linspace(-1,1, time.shape[0])

        test_function_deriv_k = [self.test_function.basis(k).deriv()(Omega) for k in range(K)]
        test_function_k = [self.test_function.basis(k)(Omega) for k in range(K)]


        for k in range(K): 
            #test_function = self.test_function.fit(Omega, input_data[:,d], deg=K)
            for d in range(D):
                b[k, d] = - integrate.simpson( 
                    #input_data[:,d] * test_function.basis(k).deriv()(Omega)
                    #+ input_data[-1,d]*test_function.basis(k)(Omega[-1]) \
                    #- input_data[0,d]*test_function.basis(k)(Omega[0])
                    input_data[:,d] * test_function_deriv_k[k] , Omega ) \
                    + input_data[-1,d]*test_function_k[k][-1] \
                    - input_data[0,d]*test_function_k[k][0]
        return b
    
    def build_G(self, input_data, time):
        basis_functions = self._basis_function(input_data)
        Omega = np.linspace(-1,1, time.shape[0])

        J = self._basis_function(input_data[0]).shape[1]
        K = self.test_function.K
        G = np.zeros((K,J))

        test_function_k = [self.test_function.basis(k)(Omega) for k in range(K)]


        for k in range(K):
            #test_function = self.test_function.fit(Omega, basis_functions[:,j], deg=K)
            for j in range(J):
                #G[k,j] = integrate.simpson( basis_functions[:,j] * test_function.basis(k)(Omega), time)
                G[k,j] = integrate.simpson( basis_functions[:,j] * test_function_k[k], time)

        return G
    
    def fit(self, input_data, time):
        self.b = self.build_b(input_data, time)
        self.G = self.build_G(input_data, time)

        self.weights = self.solve_linear_system(self.G, self.b)