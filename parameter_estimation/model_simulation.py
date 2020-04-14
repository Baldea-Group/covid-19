from __future__ import division
from pyomo.environ import *
from pyomo.dae import *
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import gc
import cplex
import pandas as pd


def simulation_fun(params,t_in, t_end):
    # Time horizon
    m = ConcreteModel()
    m.t_init = Param(initialize=t_in)
    m.t = ContinuousSet(bounds=(m.t_init, t_end))

    alpha_e_val = params['alpha_e']
    alpha_i_val = params['alpha_i']
    kappa_val = params['kappa']
    latent_val = params['latent']
    gamma_val = params['gamma']
    rho_val = params['rho']
    beta_val = params['beta']
    mu_val = params['mu']
    N = params['N']

    # Parameter definitions
    m.alpha_e = Param(initialize=alpha_e_val)
    m.alpha_i = Param(initialize=alpha_i_val)
    m.latent = Param(initialize=latent_val)
    m.gamma_p = Param(initialize=gamma_val)
    m.kappa_p = Param(initialize=kappa_val)
    m.rho_p = Param(initialize=rho_val)
    m.beta_p = Param(initialize=beta_val)
    m.mu_p = Param(initialize=mu_val)

    # State variables
    m.S = Var(m.t)  # Succeptible
    m.E = Var(m.t)  # Exposed
    m.A = Var(m.t)  # Asymptomatic
    m.I = Var(m.t)  # Infected
    m.R = Var(m.t)  # Recovered
    m.P = Var(m.t)  # Dead

    # Derivative varibales
    m.dSdt = DerivativeVar(m.S, wrt=m.t)
    m.dEdt = DerivativeVar(m.E, wrt=m.t)
    m.dAdt = DerivativeVar(m.A, wrt=m.t)
    m.dIdt = DerivativeVar(m.I, wrt=m.t)
    m.dRdt = DerivativeVar(m.R, wrt=m.t)
    m.dPdt = DerivativeVar(m.P, wrt=m.t)

    # Initial conditions

    S0 = params['S0']
    E0 = params['E0']
    A0 = params['A0']
    I0 = params['I0']
    R0 = params['R0']
    P0 = params['P0']

    m.S[m.t_init ].fix(S0)
    m.E[m.t_init ].fix(E0)
    m.A[m.t_init ].fix(A0)
    m.I[m.t_init ].fix(I0)
    m.R[m.t_init ].fix(R0)
    m.P[m.t_init ].fix(P0)

    # Total population
    m.N = Param(initialize=N)

    # Differential equations in the model
    def _diffeq1(m, t):
        return m.dSdt[t] == -m.alpha_e * m.S[t] * m.A[t] - m.alpha_i * m.S[t] * m.I[t] + m.gamma_p * m.R[t]

    m.diffeq1 = Constraint(m.t, rule=_diffeq1)

    def _diffeq2(m, t):
        return m.dEdt[t] == m.alpha_e * m.S[t] * m.A[t] + m.alpha_i * m.S[t] * m.I[t] - m.latent * m.E[t]

    m.diffeq2 = Constraint(m.t, rule=_diffeq2)

    def _diffeq3(m, t):
        return m.dAdt[t] == m.latent * m.E[t] - m.kappa_p * m.A[t] - m.rho_p * m.A[t]

    m.diffeq3 = Constraint(m.t, rule=_diffeq3)

    def _diffeq4(m, t):
        return m.dIdt[t] == m.kappa_p * m.A[t] - m.beta_p * m.I[t] - m.mu_p * m.I[t]

    m.diffeq4 = Constraint(m.t, rule=_diffeq4)

    def _diffeq5(m, t):
        return m.dRdt[t] == m.beta_p * m.I[t] + m.rho_p * m.A[t] - m.gamma_p * m.R[t]

    m.diffeq5 = Constraint(m.t, rule=_diffeq5)

    def _diffeq6(m, t):
        return m.dPdt[t] == m.mu_p * m.I[t]

    m.diffeq6 = Constraint(m.t, rule=_diffeq6)

    # Simulation of DAE system
    sim = Simulator(m, package='scipy')
    t, sol = sim.simulate(integrator='vode', numpoints=500)

    # # plt.plot(t, sol[:, 0], label='S')
    # plt.plot(t, sol[:, 1], label='E')
    # plt.plot(t, sol[:, 2], label='A')
    # plt.plot(t, sol[:, 3], label='I')
    # plt.plot(t, sol[:, 4], label='R')
    # plt.plot(t, sol[:, 5], label='P')
    # plt.xlabel('Time (days)')
    # plt.ylabel('Normalized populations')
    # plt.tick_params(axis='both', direction='in', top=True, right=True)
    # plt.legend(loc='best')
    # # plt.title('Pyomo + scipy')
    # plt.grid()
    # plt.savefig('sim_params_Peng_et_al.png')
    # plt.show()

    return sim, t, sol
