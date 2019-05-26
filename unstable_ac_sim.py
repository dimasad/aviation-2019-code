"""Output error method estimation of the longitudinal parameters a simulated
unstable aircraft.

This example corresponds to the test case #9 of the 4th chapter of the book
Flight Vehicle System Identification: A Time-Domain Methodology, Second Edition
by Ravindra V. Jategaonkar, Senior Scientist, Institute of Flight Systems, DLR.
The example is described in detail in Section 9.16.1, with the nominal values 
and estimates from other methods presented in Table 9.1.

The test data for this experiment can be downloaded, free of charge, from 
<https://arc.aiaa.org/doi/suppl/10.2514/4.102790>. To use it, extract 
`unStabAC_sim.asc` from `flt_data.zip` into the `data/` directory.
"""


import functools
import os
import os.path

import numpy as np
import sympy
import sym2num.model
import sym2num.utils
from scipy import integrate, interpolate
from sym2num import var
from sympy import cos, sin

from ceacoest import oem, optim
from ceacoest.modelling import symcol, symoem, symstats


@symoem.collocate(order=2)
class UAcLong:
    """Symbolic unstable aircraft longitudinal model."""
    
    @property
    def generate_functions(self):
        """Iterable of the model functions to generate."""
        return getattr(super(), 'generate_functions', set()) | {'g'}
    
    @sym2num.utils.classproperty
    @functools.lru_cache()
    def variables(cls):
        """Model variables definition."""
        params = ['zw', 'zq', 'zeta', 'mw', 'mq', 'meta',
                  'w_meas_std', 'q_meas_std', 'az_meas_std']
        vars = [var.SymbolArray('x', ['w', 'q']),
                var.SymbolArray('y', ['w_meas', 'q_meas', 'az_meas']),
                var.SymbolArray('u', ['eta']),
                var.SymbolArray('p', params)]
        return var.make_dict(vars)
    
    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
        """ODE function."""
        u0  = 44.5695
        wd = s.zw*s.w + (u0 + s.zq)*s.q + s.zeta*s.eta
        qd = s.mw*s.w + s.mq*s.q + s.meta*s.eta
        return sympy.Array([wd, qd])
    
    @sym2num.model.collect_symbols
    def g(self, x, u, p, *, s):
        """Measurement log likelihood."""
        az = s.zw*s.w + s.zq*s.q + s.zeta*s.eta
        return sympy.Array([s.w, s.q, az])
    
    @sym2num.model.collect_symbols
    def L(self, y, x, u, p, *, s):
        """Measurement log likelihood."""
        az = self.g(x, u, p)[2]
        return sympy.Array(
            symstats.normal_logpdf1(s.w_meas, s.w, s.w_meas_std)
            + symstats.normal_logpdf1(s.q_meas, s.q, s.q_meas_std)
            + symstats.normal_logpdf1(s.az_meas, az, s.az_meas_std)
        )


if __name__ == '__main__':
    # decision variable lower bounds
    lower = {'w_meas_std': 1e-4, 'q_meas_std': 1e-8, 'az_meas_std': 1e-8}
    
    # Compile and instantiate model
    symb_mdl = UAcLong()
    GeneratedUAcLong = sym2num.model.compile_class(symb_mdl)
    model = GeneratedUAcLong()
    
    # Load experiment data
    dirname = os.path.dirname(__file__)
    data = np.loadtxt(os.path.join(dirname, 'data', 'unStabAC_sim.asc'))
    Ts = 0.05 # Sampling time
    t = np.arange(len(data)) * Ts
    y = data[:, [2, 3, 5]]
    u = interpolate.interp1d(t, data[:, [4]], axis=0)
    
    # Create OEM problem
    problem = oem.Problem(model, t, y, u)
    tc = problem.tc
    
    # Set bounds
    constr_bounds = np.zeros((2, problem.ncons))
    dec_L, dec_U = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    for k,v in lower.items():
        problem.set_decision_item(k, v, dec_L)
    
    # Set initial guess
    x0 = interpolate.interp1d(t, y.T[:2])(tc).T
    p0 = np.zeros(model.np)
    p0[-model.ny:] = 1 # guess for measurement standard deviations
    dec0 = np.zeros(problem.ndec)
    problem.set_decision('x', x0, dec0)
    problem.set_decision('p', p0, dec0)
    
    dec_scale = np.ones(problem.ndec)
    
    constr_scale = np.ones(problem.ncons)
    
    with problem.ipopt((dec_L, dec_U), constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('ma57_pre_alloc', 1e3)
        nlp.add_num_option('tol', 1e-6)
        nlp.add_int_option('max_iter', 1000)
        nlp.set_scaling(-1, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)
    xopt = opt['x']
    popt = opt['p']
    yopt = model.g(xopt, problem.u, popt)
    
    os.makedirs('results', exist_ok=True)
    np.savetxt('results/uac_u.txt', np.c_[tc, problem.u])
    np.savetxt('results/uac_xopt.txt', np.c_[tc, xopt])
    np.savetxt('results/uac_popt.txt', popt)
    np.savetxt('results/uac_yopt.txt', np.c_[tc, yopt])
    np.savetxt('results/uac_z.txt', np.c_[t, y])
