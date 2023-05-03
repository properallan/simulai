import numpy as np
from scipy.linalg import lstsq
from numpy import matlib as mb
from scipy import interpolate
import scipy
from scipy.optimize import brentq

import numba

@numba.njit
def shift(arr, num, fill_value=np.nan):
    if num >= 0:
        return np.concatenate((np.full(num, fill_value), arr[:-num]))
    else:
        return np.concatenate((arr[-num:], np.full(-num, fill_value)))

@numba.njit
def shift5_numba(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def cut_boundaries(array, L):
    L = int(L)
    ret = array
    if L > 0:
        ret = array[...,L:-L]
    return ret

def extrapolate_boundaries(array, L):
    L = int(L)
    f = interpolate.interp1d(np.arange(len(array)), array, fill_value = "extrapolate")
    pre = f(np.arange(-L,0))
    post = f(np.arange(len(array), len(array)+L))
    return np.concatenate((pre, array, post))

def refine_grid(grid, new_pts):
    L = grid[0,1]-grid[0,0]
    new_a = np.array(np.linspace(grid[0,0], grid[1,0], new_pts+2)[1:-1], dtype=np.int32)
    new_b = new_a + L
    append_grid = np.array((new_a,new_b)).T

    new_grid = np.concatenate((grid[0:1], append_grid, grid[1:]))

    return new_grid

class TestFunction:
    def __init__(self, p:int=16, q:int=16):
        self.p = p
        self.q = q

    def __call__(self):
        p = self.p
        q = self.q
        def g(t, t1, tk): return (p > 0)*(q > 0)*(t - t1)**max(p, 0)*(tk - t)**max(q, 0) + (p == 0)*(q == 0)*(1 - 2 *
                                                                                                            np.abs(t - (t1+tk)/2)/(tk - t1)) + (p > 0)*(q < 0)*np.sin(p*np.pi/(tk - t1)*(t - t1)) + (p == -1)*(q == -1)

        def gp(t, t1, tk): return (t-t1)**(max(p-1, 0))*(tk-t)**(max(q-1, 0))*((-p-q)*t+p*tk+q*t1)*(q > 0)*(p > 0) + -2*np.sign(t -
                                                                                                                                (t1+tk)/2)/(tk-t1)*(q == 0)*(p == 0) + p*np.pi/(tk-t1)*np.cos(p*np.pi/(tk-t1)*(t-t1))*(q < 0)*(p > 0) + 0*(p == -1)*(q == -1)

        if p > 0 and q > 0:
            def normalize(t, t1, tk): return (
                t - t1)**max(p, 0)*(tk - t)**max(q, 0)

            def g(t, t1, tk): return ((p > 0)*(q > 0)*(t - t1)**max(p, 0)*(tk - t)**max(q, 0) + (p == 0)*(q == 0)*(1 - 2*np.abs(t - (t1+tk)/2) /
                                                                                                                (tk - t1)) + (p > 0)*(q < 0)*np.sin(p*np.pi/(tk - t1)*(t - t1)) + (p == -1)*(q == -1))/(np.abs(normalize((q*t1+p*tk)/(p+q), t1, tk)))

            def gp(t, t1, tk): return ((t-t1)**(max(p-1, 0))*(tk-t)**(max(q-1, 0))*((-p-q)*t+p*tk+q*t1)*(q > 0)*(p > 0) + -2*np.sign(t-(t1+tk)/2)/(tk-t1)*(q == 0)
                                    * (p == 0) + p*np.pi/(tk-t1)*np.cos(p*np.pi/(tk-t1)*(t-t1))*(q < 0)*(p > 0) + 0*(p == -1)*(q == -1))/(np.abs(normalize((q*t1+p*tk)/(p+q), t1, tk)))

        return g, gp


class AdaptiveTestFunction(TestFunction):
    def __init__(
        self, 
        s, 
        K, 
        p, 
        tau_p, 
        r_whm,
        ghost_cells:int=0, 
        **kwargs):

        self.s = s
        self.K = K
        self.p = p
        self.tau_p =  tau_p
        self.r_whm = r_whm
        self.sampling = 'adaptive'
        self.ghost_cells = ghost_cells

        super().__init__( p=self.p, q=self.p)

    def AG_tf_mat_row(self, g, gp, t, t1, tk, param=None):
        if self.ghost_cells > 0 :
            t = extrapolate_boundaries(t, self.ghost_cells)

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

        if self.ghost_cells == 0:
            return V_row, Vp_row
        else:
            return cut_boundaries(V_row, self.ghost_cells), cut_boundaries(Vp_row, self.ghost_cells)



        #return V_row, Vp_row

    def grid_adaptive(self, x, t, params=None):
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
        self.p = p
        self.q = p
        g, gp = self()
        o, Vp_row = self.AG_tf_mat_row(g, gp, t, 1, 1+index_gap, [1, 1, 0])
        
        #Vp_row = cut_boundaries(Vp_row, self.ghost_cells)
        #M = M - self.ghost_cells*2

        Vp_diags = mb.repmat(Vp_row[:, 0:index_gap+1], M - index_gap, 1)
        Vp = scipy.sparse.diags(Vp_diags.T, np.arange(
            0, index_gap+1), (M-index_gap, M))

        weak_der = Vp.dot(x)
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

    def build_V_Vp(self, t, x=None, params=None):
        V_, Vp_, grid_ = [], [], []
        for i in range(x.shape[1]):
            V, Vp, grid = self._build_V_Vp(t=t, x=x[:,i])
            V_.append(V)
            Vp_.append(Vp)
            grid_.append(grid)
        return np.array(V_), np.array(Vp_), np.array(grid_)
    
    def _build_V_Vp(self, t, x=None, params=None):
        # Adaptive vectors V and Vp are built for each dimension
        if params == None:
            params = [self.s, self.K, self.p, 1]

        grid = self.grid_adaptive(x, t, params=params)

        #if self.ghost_cells != 0:
        #    t = extrapolate_boundaries(t, self.ghost_cells)
        
        V, Vp, ab_grid = self.build_adaptive_whm(t, grid, self.r_whm, [0, np.inf, 0])

        #if self.ghost_cells == 0:
        return V, Vp, ab_grid, # ps
        #else:
        #    return V[:,int(self.ghost_cells):-int(self.ghost_cells)], Vp[:,int(self.ghost_cells):-int(self.ghost_cells)], ab_grid

        #return V, Vp, ad_grid
        #return V[:,int(self.ghost_cells):-int(self.ghost_cells)], Vp[:,int(self.ghost_cells):-int(self.ghost_cells)], grid


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

    def build_adaptive_whm(self, t, centers, r_whm, param=None):
        if param == None:
            param = [1, 2, 1]

        #if self.ghost_cells > 0 :
        #    t = extrapolate_boundaries(t, self.ghost_cells)

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

        self.p = p
        self.q = p
        #g, gp = self.basis_fcn(p, p)
        g, gp = self()
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
                #V_row = shift5_numba(V_row,  -(cent_shift + self.ghost_cells), 0)
                #Vp_row = shift5_numba(Vp_row, -(cent_shift + self.ghost_cells), 0)
                V_row = shift5_numba(V_row,  cent_shift, 0)
                Vp_row = shift5_numba(Vp_row, cent_shift, 0)
                
                #V_row = np.roll(V_row, cent_shift + int(self.ghost_cells/2))
                #Vp_row = np.roll(Vp_row, cent_shift + int(self.ghost_cells/2))
            else:
                p, a, b = self.test_fcn_param(
                    r_whm, t[int(centers[k]-1)], t)
                a = int(a)
                b = int(b)
                if b-a < 10:
                    center = (a+b)/2
                    b = int(min(np.ceil(center+5), len(t)))
                    a = int(max(0, np.floor(center-5)))
                
                self.p = p
                self.q = p
                g, gp = self()
                #g, gp = self.basis_fcn(p, p)
                V_row, Vp_row = self.tf_mat_row(g, gp, t, a, b, param)

            V[k, :] = V_row
            Vp[k, :] = Vp_row

            ab_grid[k, :] = np.array([a, b])
            ps[k] = p
        
        #if self.ghost_cells == 0:
        return V, Vp, ab_grid, # ps
        #else:
        #    return V[:,int(self.ghost_cells):-int(self.ghost_cells)], Vp[:,int(self.ghost_cells):-int(self.ghost_cells)], ab_grid

    def tf_mat_row(self, g, gp, t, t1, tk, param):
        if self.ghost_cells > 0 :
            t = extrapolate_boundaries(t, self.ghost_cells)

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

        if self.ghost_cells == 0:
            return V_row, Vp_row
        else:
            return cut_boundaries(V_row, self.ghost_cells), cut_boundaries(Vp_row, self.ghost_cells)

        
        #return V_row, Vp_row

    
class UniformTestFunction(TestFunction):
    def __init__(self, L:int=100, overlap:float=0.9, ghost_cells:int=0, **kwargs):
        self.L = L
        self.overlap = overlap
        self.sampling = 'uniform'
        self.ghost_cells = ghost_cells
        super().__init__( **kwargs)

    def grid(self, t):
        M = len(t)
        # polynomial degree, should vary
        p = self.p
        s = self.overlap
        L = self.L
        n_overlap = int(np.floor(L*(1-np.sqrt(1-s**(1/p)))))
        
        a = 0
        b = L
        grid = [[a, b]]

        while b - n_overlap + L <= M-1:
            a = b - n_overlap
            b = a + L
            grid.append([a, b])

        grid = np.array(grid)
        #grid = refine_grid(grid, 3)

        return grid
    
    def _build_V_Vp(self, t, x=None):
        V_, Vp_, grid_ = [], [], []
        for i in range(x.shape[1]):
            V, Vp, grid = self._build_V_Vp(t=t, x=x[:,i])
            V_.append(V)
            Vp_.append(Vp)
            grid_.append(grid)
        return np.array(V_), np.array(Vp_), np.array(grid_)

    def build_V_Vp(self, t, x=None):
        if self.ghost_cells != 0:
            t = extrapolate_boundaries(t, self.ghost_cells)
        grid = self.grid(t)
        p = self.p
        M = len(t)
        N = len(grid)

        V = np.zeros((N, M))
        Vp = np.zeros((N, M))

        g, gp = self.__call__()
        for k in range(N):
            a = grid[k][0]
            b = grid[k][1]
            V_row, Vp_row = self.integration_vectors(g, gp, t, a, b)
            #V[k, a:b+1] = V_row
            #Vp[k, a:b+1] = Vp_row

            V[k, :] = V_row
            Vp[k, :] = Vp_row

        if self.ghost_cells == 0:
            return V, Vp, grid
        else:
            return V[:,int(self.ghost_cells):-int(self.ghost_cells)], Vp[:,int(self.ghost_cells):-int(self.ghost_cells)], grid

        #return V, Vp, grid
        #return V[:,int(self.ghost_cells):-int(self.ghost_cells)], Vp[:,int(self.ghost_cells):-int(self.ghost_cells)], grid


    def integration_vectors(self, g, gp, t, a, b, param=[0, np.inf, 0]):
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
        
        if a > b:
            a_temp = b
            b = a
            a = a_temp

        V_row = np.zeros((1, N))
        Vp_row = np.zeros_like(V_row)

        t_grid = t[a:b+1:gap] 
        dts = np.diff(t_grid)
        w = 1/2.*(np.append(dts, [0]) + np.append([0], dts))

        V_row[:, a:b+1:gap] = g(t_grid, t[a], t[b])*w
        Vp_row[:, a:b+1:gap]= -gp(t_grid, t[a], t[b])*w
        Vp_row[:, a] = Vp_row[:, a] - g(t[a], t[a], t[b])
        Vp_row[:, b] = Vp_row[:, b] + g(t[b], t[a], t[b])

        
        #V_row = np.trapz(y=g(t_grid, t[a], t[b]), x=t_grid, dx=w[0])
        #Vp_row = np.trapz(y=-gp(t_grid, t[a], t[b]), x=t_grid, dx=w[0])

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

class UniformMultiscaleTestFunction(UniformTestFunction):
    def __init__(self, L:int=100, overlap:float=0.9, ghost_cells:int=0, **kwargs):
        self.sampling = 'uniform_multiscale'
        super().__init__( L, overlap, ghost_cells, **kwargs)

class rkhsFunction:
    def __init__(self, samples:int=100, **kwargs):
        self.samples = samples

    def __call__(self):
        def g(x, y):
            mu = 20**2/3
            return np.exp(-1/mu*np.linalg.norm(x-y)**2)
        
        def gp(x, y):
            mu = 20**2/3
            return (-2/mu)*(x-y)*np.exp(-1/mu*np.linalg.norm(x-y)**2)
        
        return g, gp

    #def build_V_Vp(self, t):
    #    samples = np.linspace(t[0], t[-1], self.samples)
    #    V = np.zeros((len(samples), len(t)))
    #    Vp = np.zeros((len(samples), len(t)))

    #    dts = np.diff(t)
    #    w = 1/2.*(np.append(dts, [0]) + np.append([0], dts))

    #    for i in range(len(samples)):
    #        for j in range(len(t)):
    #            V[i, j] = self()[0](samples[i], t[j])
    #            Vp[i, j] = -self()[1](samples[i], t[j])
    #        #V[i, :] = V[i, :]*w
    #        #Vp[i, :] = Vp[i, :]*w
    #    
    #    return V, Vp