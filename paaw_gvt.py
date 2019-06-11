"""Output Error Method for the parameter estimation of a ground vibration test
(GVT) of the Performance Adaptive Aeroelastic Wing program (PAAW)."""


import itertools
import os

import numpy as np
import scipy.io
import sympy
import sym2num.model
from scipy import interpolate, signal

from ceacoest import oem, optim
from ceacoest.modelling import symoem, symstats


@symoem.collocate(order=4)
class PaawGvt:
    """Symbolic linear GVT model, modal form."""

    @property
    def variables(self):
        v = super().variables
        v['x'] = ['x1a', 'x1b', 'x2a', 'x2b', 'x3a', 'x3b', 'x4a', 'x4b',
                  'x5a', 'x5b', 'x6a', 'x6b', 'x7a', 'x7b', 'x8a', 'x8b',
                  'x9a', 'x9b', 'x10a', 'x10b', 'x11a', 'x11b', 'x12a', 'x12b']
        v['y'] = ['y1']
        v['u'] = ['de']
        v['p'] = ['sigma1',
                  'sigma2',
                  'sigma3',
                  'sigma4',
                  'sigma5',
                  'sigma6',
                  'sigma7',
                  'sigma8',
                  'sigma9',
                  'sigma10',
                  'sigma11',
                  'sigma12',
                  'omega1',
                  'omega2',
                  'omega3',
                  'omega4',
                  'omega5',
                  'omega6',
                  'omega7',
                  'omega8',
                  'omega9',
                  'omega10',
                  'omega11',
                  'omega12',
                  'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 
                  'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18',
                  'c19', 'c20', 'c21', 'c22', 'c23', 'c24', 
                  'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 
                  'b10', 'b11', 'b12',
                  'y1_std']
        return v
    
    
    @property
    def generate_functions(self):
        return {'g', 'A', 'B', 'C'}
    
    def A(self, p):
        nx = len(self.variables['x'])
        nmode = nx // 2
        sigma = p[:nmode]
        omega = p[nmode:nx]
        A_diag = ([[s, w], [-w, s]] for s,w in zip(sigma, omega))
        return sympy.diag(*A_diag)
    
    def B(self, p):
        nx = len(self.variables['x'])
        nmode = nx // 2
        b = p[2*nx:2*nx+nmode]
        return sympy.Matrix(sympy.flatten(zip(b, [0]*nmode)))

    def C(self, p):
        nx = len(self.variables['x'])
        c = p[nx:2*nx]
        return sympy.Matrix([c])
    
    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
        """ODE function."""
        A = self.A(p)
        B = self.B(p)
        xdot = A * sympy.Matrix(x) + B * sympy.Matrix(u)
        return [*xdot]

    @sym2num.model.collect_symbols
    def g(self, x, u, p, *, s):
        """Measurement log likelihood."""
        C = self.C(p)
        y = C * sympy.Matrix(x)
        return list(y)
    
    @sym2num.model.collect_symbols
    def L(self, y, x, u, p, *, s):
        """Measurement log likelihood."""
        y1_pred, = self.g(x, u, p)
        return symstats.normal_logpdf1(s.y1, y1_pred, s.y1_std)


def load_data():
    module_dir = os.path.dirname(__file__)
    data_file_path = os.path.join(
        module_dir, 'data', 'paaw_gvt_exp7_accel1_filt.txt'
    )
    data = np.loadtxt(data_file_path)
    u = interpolate.interp1d(data[:, -1], data[:, :1], axis=0)
    
    downsample = 20
    y = data[::downsample, 1:2]
    t = data[::downsample, -1]
    return t, u, y


if __name__ == '__main__':
    # Compile and instantiate model
    symb_mdl = PaawGvt()
    GeneratedPaawGvt = sym2num.model.compile_class(symb_mdl)
    model = GeneratedPaawGvt()
    nx = model.nx
    nmode = nx // 2
    
    # Load experiment data
    t, u, y = load_data()

    # Create OEM problem
    problem = oem.Problem(model, t, y, u)
    tc = problem.tc
    uc = problem.u
    
    # Define problem bounds
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    dec_L, dec_U = dec_bounds
    problem.set_decision_item('y1_std', 0.00025, dec_L)
    for k in range(1, nmode + 1):
        problem.set_decision_item(f'sigma{k}', 0, dec_U)
        problem.set_decision_item(f'omega{k}', 6 * 2 * np.pi, dec_L)
        problem.set_decision_item(f'omega{k}', 50 * 2 * np.pi, dec_U)
    
    constr_bounds = np.zeros((2, problem.ncons))
    constr_L, constr_U = constr_bounds
    
    # Define initial guess for decision variables
    dec0 = np.zeros(problem.ndec)
    problem.set_decision_item('y1_std', 1e-5, dec0)
    for k in range(1, nx + 1):
        problem.set_decision_item(f'c{k}', 1, dec0)
    
    omega_guess = [199.06,
                   186.17,
                   178.44,
                   172.64,
                   144.37,
                   121.47,
                   96.515,
                   59.572,
                   58.567,
                   59.029,
                   48.989,
                   48.901]
    for k, omega in enumerate(omega_guess):
        problem.set_decision_item(f'b{k+1}', 1, dec0)
        problem.set_decision_item(f'omega{k+1}', omega, dec0)
        
    # Define problem scaling
    dec_scale = np.ones(problem.ndec)
    constr_scale = np.ones(problem.ncons)
    obj_scale = -1.0e-4
    
    # Run ipopt
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma97')
        nlp.add_num_option('tol', 1e-4)
        nlp.add_num_option('acceptable_tol', 1e-3)
        nlp.add_int_option('acceptable_iter', 5)
        nlp.set_scaling(obj_scale, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)
    
    # Unpack the solution
    opt = problem.variables(decopt)
    xopt = opt['x']
    popt = opt['p']
    yopt = model.g(xopt, uc, popt)
    
    # Build LTI sytem object from estimated model
    Aopt = model.A(popt)
    Bopt = model.B(popt)
    Copt = model.C(popt)
    sys_opt = signal.lti(Aopt, Bopt, Copt, 0.0)
    
    # Calculate the model frequency response
    f = np.logspace(np.log10(6), np.log10(35), 1000)
    w = f * 2 * np.pi
    w, Gopt = signal.freqresp(sys_opt, w)
    bode_mag = 20 * np.log10(abs(Gopt))

    # Save results
    os.makedirs('results', exist_ok=True)
    np.savetxt('results/paaw_gvt_A.txt', Aopt)
    np.savetxt('results/paaw_gvt_B.txt', Bopt)
    np.savetxt('results/paaw_gvt_C.txt', Copt)
    np.savetxt('results/paaw_gvt_xopt.txt', xopt)
    np.savetxt('results/paaw_gvt_yopt.txt', yopt)
    np.savetxt('results/paaw_gvt_opt_bode.txt', np.c_[f, bode_mag])
