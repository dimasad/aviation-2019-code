"""Output error method estimation of the longitudinal parameters of an HFB-320.

The parameters are estimated from several random starting values for the 
nonlinear optimization.

This example corresponds to the test case #4 of the 4th chapter of the book
Flight Vehicle System Identification: A Time-Domain Methodology, Second Edition
by Ravindra V. Jategaonkar, Senior Scientist, Institute of Flight Systems, DLR.

It uses flight test data obtained by the DLR that accompanies the book's
supporting materials (not provided here).
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
class HFB320Long:
    """Symbolic HFB-320 aircraft nonlinear longitudinal model."""
    
    @property
    def generate_functions(self):
        """Iterable of the model functions to generate."""
        return getattr(super(), 'generate_functions', set()) | {'g'}
    
    @sym2num.utils.classproperty
    @functools.lru_cache()
    def variables(cls):
        """Model variables definition."""
        params = [
            'CD0', 'CDV', 'CDa', 'CL0', 'CLV', 'CLa',
            'Cm0', 'CmV', 'Cma', 'Cmq', 'Cmde', 
            'q_bias', 'qdot_bias', 'ax_bias', 'az_bias',
            'V_meas_std', 'alpha_meas_std', 'theta_meas_std', 'q_meas_std',
            'qdot_meas_std', 'ax_meas_std', 'az_meas_std', 
        ]
        consts = [
            'g0', 'Sbym', 'ScbyIy', 'FEIYLT', 'V0', 'mass', 'sigmaT', 'rho',
            'cbarH'
        ]
        y = ['V_meas', 'alpha_meas', 'theta_meas', 'q_meas', 
             'qdot_meas', 'ax_meas', 'az_meas']

        vars = [var.SymbolObject('self', var.SymbolArray('consts', consts)),
                var.SymbolArray('x', ['V', 'alpha', 'theta', 'q']),
                var.SymbolArray('y', y),
                var.SymbolArray('u', ['de', 'T']),
                var.SymbolArray('p', params)]
        return var.make_dict(vars)
    
    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
        """ODE function."""
        qbar = 0.5 * s.rho * s.V ** 2
        gamma = s.theta - s.alpha
        qhat = s.cbarH * s.q / s.V0
        
        CD = s.CD0 + s.CDV * (s.V - s.V0) / s.V0 + s.CDa * s.alpha
        CL = s.CL0 + s.CLV * (s.V - s.V0) / s.V0 + s.CLa * s.alpha
        Cm = (s.Cm0 + s.CmV * (s.V - s.V0) / s.V0 + s.Cma * s.alpha + 
              s.Cmq*qhat + s.Cmde*s.de)
        
        Vd = (-s.Sbym*qbar*CD + s.T*cos(s.alpha + s.sigmaT)/s.mass 
              - s.g0*sin(gamma))
        alphad = (-s.Sbym*qbar/s.V*CL - s.T*sin(s.alpha + s.sigmaT)/(s.mass*s.V)
                  + s.g0*cos(gamma)/s.V + s.q)
        qd = s.ScbyIy*qbar*Cm + s.T*s.FEIYLT
        return sympy.Array([Vd, alphad, s.q, qd])

    @sym2num.model.collect_symbols
    def g(self, x, u, p, *, s):
        """System outputs."""
        qbar = 0.5 * s.rho * s.V ** 2
        qhat = s.cbarH * s.q / s.V0

        CD = s.CD0 + s.CDV * (s.V - s.V0) / s.V0 + s.CDa * s.alpha
        CL = s.CL0 + s.CLV * (s.V - s.V0) / s.V0 + s.CLa * s.alpha
        Cm = (s.Cm0 + s.CmV * (s.V - s.V0) / s.V0 + s.Cma * s.alpha + 
              s.Cmq*qhat + s.Cmde*s.de)
        
        salpha =  sin(s.alpha)
        calpha =  cos(s.alpha)
        CX =  CL*salpha - CD*calpha
        CZ = -CL*calpha - CD*salpha
        
        qdot = s.ScbyIy*qbar*Cm + s.T*s.FEIYLT + s.qdot_bias
        ax = s.Sbym*qbar*CX + s.T*cos(s.sigmaT)/s.mass + s.ax_bias
        az = s.Sbym*qbar*CZ - s.T*sin(s.sigmaT)/s.mass + s.az_bias
        out = [s.V, s.alpha, s.theta, s.q + s.q_bias, qdot, ax, az]
        return sympy.Array(out)
    
    @sym2num.model.collect_symbols
    def L(self, y, x, u, p, *, s):
        """Measurement log likelihood."""
        out = self.g(x, u, p)
        return sympy.Array(
            symstats.normal_logpdf1(s.V_meas, out[0], s.V_meas_std)
            + symstats.normal_logpdf1(s.alpha_meas, out[1], s.alpha_meas_std)
            + symstats.normal_logpdf1(s.theta_meas, out[2], s.theta_meas_std)
            + symstats.normal_logpdf1(s.q_meas, out[3], s.q_meas_std)
            + symstats.normal_logpdf1(s.qdot_meas, out[4], s.qdot_meas_std)
            + symstats.normal_logpdf1(s.ax_meas, out[5], s.ax_meas_std)
            + symstats.normal_logpdf1(s.az_meas, out[6], s.az_meas_std)
        )


if __name__ == '__main__':
    given = {'g0': 9.80665, 'Sbym': 4.0280e-3, 'ScbyIy': 8.0027e-4, 
             'FEIYLT': -7.0153e-6, 'V0': 104.67, 'mass':7472, 'sigmaT':0.0524,
             'rho': 0.7920, 'cbarH': 1.215}
    lower = {'V': 2, 'V_meas_std': 1e-3, 'alpha_meas_std': 1e-4,
             'theta_meas_std': 1e-4, 'q_meas_std': 1e-4, 'qdot_meas_std': 1e-4,
             'ax_meas_std': 1e-4, 'az_meas_std': 1e-4, 'CD0': 0}
    
    # Compile and instantiate model
    symb_mdl = HFB320Long()
    GeneratedHFB320Long = sym2num.model.compile_class(symb_mdl)
    model = GeneratedHFB320Long(**given)
    
    # Load experiment data
    dirname = os.path.dirname(__file__)
    data = np.loadtxt(os.path.join(dirname, 'data', 'hfb320_1_10.asc'))
    Ts = 0.1
    ndata = len(data)
    tdata = np.arange(ndata) * Ts
    ydata = data[:, 4:11]
    udata = data[:, [1,3]]
    u = interpolate.interp1d(tdata, udata, axis=0)
    
    # Prepare problem inputs
    nmp = 0 # number of mesh points in between measurements (>= 0)
    t = np.arange((ndata - 1) * (1 + nmp) + 1) * Ts / (nmp + 1)
    y = np.ma.masked_all((len(t), ydata.shape[1]))
    y[::nmp + 1] = ydata
    u = interpolate.interp1d(tdata, udata, axis=0)
    
    # Simulate single shooting with null parameters
    p0 = np.zeros(model.np)
    def odefun(t, x):
        return model.f(x, u(t), p0)
    xini = y[0, :4]
    tspan = [0, 3]
    sol_ss = integrate.solve_ivp(odefun, tspan, xini, max_step=0.05)

    os.makedirs('results', exist_ok=True)
    np.savetxt('results/hfb_single_shooting.txt', np.c_[sol_ss.t, sol_ss.y.T])
    
    # Simulate single shooting with null parameters
    ms_file = open('results/hfb_multiple_shooting.txt', 'wt')
    for t0 in range(60):
        tspan = [t0, t0+0.99]
        xini = interpolate.interp1d(tdata, ydata.T[:4])(t0)
        sol_ms = integrate.solve_ivp(odefun, tspan, xini, max_step=0.1)
        for t, x in zip(sol_ms.t, sol_ms.y.T):
            print(t, *x, file=ms_file)
        print(t0 + 1.0, *['nan'] * 4, file=ms_file)
    ms_file.close()
    
    raise SystemExit
    # Create OEM problem
    problem = oem.Problem(model, t, y, u)
    tc = problem.tc
    
    # Set bounds
    constr_bounds = np.zeros((2, problem.ncons))
    dec_L, dec_U = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    for k,v in lower.items():
        problem.set_decision_item(k, v, dec_L)
    
    # Set initial guess
    x0 = interpolate.interp1d(tdata, ydata.T[:4])(tc).T
    p0 = np.zeros(model.np)
    p0[-model.ny:] = 1 # guess for measurement standard deviations
    dec0 = np.zeros(problem.ndec)
    problem.set_decision('x', x0, dec0)
    problem.set_decision('p', p0, dec0)
    
    # Set problem scaling
    dec_scale = np.ones(problem.ndec)
    problem.set_decision_item('V', 1e-2, dec_scale)
    problem.set_decision_item('alpha', 20, dec_scale)
    problem.set_decision_item('q', 30, dec_scale)
    problem.set_decision_item('theta', 20, dec_scale)
    problem.set_decision_item('V_meas_std', 1/0.2, dec_scale)
    problem.set_decision_item('alpha_meas_std', 1/0.03, dec_scale)
    problem.set_decision_item('theta_meas_std', 1/0.002, dec_scale)
    problem.set_decision_item('q_meas_std', 1/0.001, dec_scale)
    problem.set_decision_item('qdot_meas_std', 1/0.025, dec_scale)
    problem.set_decision_item('ax_meas_std', 1/0.03, dec_scale)
    problem.set_decision_item('az_meas_std', 1/0.03, dec_scale)

    # Set constraint scaling
    constr_scale = np.ones(problem.ncons)
    problem.set_defect_scale('V', 1e-2, constr_scale)
    problem.set_defect_scale('alpha', 20, constr_scale)
    problem.set_defect_scale('q', 30, constr_scale)
    problem.set_defect_scale('theta', 20, constr_scale)

    # Run estimation starting with zero for the dynamic system parameters
    with problem.ipopt((dec_L, dec_U), constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma97')
        nlp.add_num_option('tol', 1e-6)
        nlp.set_scaling(-1, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)
    
    # Save the results for this initial estimation and the parameter values
    opt = problem.variables(decopt)
    popt_best = opt['p']
    yopt = model.g(opt['x'], problem.u, opt['p'])
    os.makedirs('results', exist_ok=True)
    np.savetxt('results/hfb_u.txt', np.c_[tc, problem.u])
    np.savetxt('results/hfb_xopt.txt', np.c_[tc, opt['x']])
    np.savetxt('results/hfb_popt.txt', opt['p'])
    np.savetxt('results/hfb_yopt.txt', np.c_[tc, yopt])
    np.savetxt('results/hfb_z.txt', np.c_[t, y])
    
    # Repeat for random starting values
    for i in range(1000):
        # Initialize random number generator
        np.random.seed(i)
        
        # Sample the starting value
        problem.set_decision_item("CD0", np.random.uniform(0, 0.5), dec0)
        problem.set_decision_item("CDV", np.random.uniform(-0.5, 0.5), dec0)
        problem.set_decision_item("CDa", np.random.uniform(0, 1), dec0)
        problem.set_decision_item("CL0", np.random.uniform(0, 2), dec0)
        problem.set_decision_item("CLV", np.random.uniform(-2, 2), dec0)
        problem.set_decision_item("CLa", np.random.uniform(0, 10), dec0)
        problem.set_decision_item("Cm0", np.random.uniform(0, 0.5), dec0)
        problem.set_decision_item("CmV", np.random.uniform(0, 0.5), dec0)
        problem.set_decision_item("Cma", np.random.uniform(-5, 1), dec0)
        problem.set_decision_item("Cmq", np.random.uniform(-50, 0), dec0)
        problem.set_decision_item("Cmde", np.random.uniform(-10, 0), dec0)
        p0 = problem.unpack_decision(dec0)['p']
        
        # Create the results folder
        os.makedirs(f'results/hfb_mc_{i:04}', exist_ok=True)

        # Save the starting parameters
        np.savetxt(f'results/hfb_mc_{i:04}/p0.txt', p0)

        # Run the estimation
        with problem.ipopt((dec_L, dec_U), constr_bounds) as nlp:
            nlp.add_str_option('linear_solver', 'ma57')
            nlp.add_num_option('tol', 1e-6)
            nlp.add_num_option('ma57_pre_alloc', 1e3)
            nlp.set_scaling(-1, dec_scale, constr_scale)
            decopt, info = nlp.solve(dec0)
        
        # Save the estimation results
        opt = problem.variables(decopt)
        popt = opt['p']
        perr = np.max(np.abs(popt - popt_best))
        print(f'Parameter error: {perr}')
        np.savetxt(f'results/hfb_mc_{i:04}/popt.txt', popt)
        np.savetxt(f'results/hfb_mc_{i:04}/perr.txt', [perr])
        if perr > 1e-7:
            np.savetxt(f'results/hfb_mc_{i:04}/status_diverged.txt', [1])
