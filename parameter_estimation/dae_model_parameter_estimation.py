'''
Author: Fernando Lejarza (lejarza@utexas.edu)
Affilition: The University of Texas at Austin
Last modified: 04.17.2020 (CT)
'''


from __future__ import division
from pyomo.environ import *
from pyomo.dae import *
import matplotlib.pyplot as plt
import data_read
import model_simulation
import gc


def param_estimation_fun(country_name, n_pwl):
    
    data_I_dict, N = data_read.return_data_ready('I', country_name)
    data_P_dict, N = data_read.return_data_ready('P', country_name)
    data_R_dict, N = data_read.return_data_ready('R', country_name)
    for key in data_I_dict.keys():
        data_I_dict[key] = data_I_dict[key] - data_P_dict[key] - data_R_dict[key]
        
    t_horizon = len(data_I_dict.keys()) - 1

    # Time horizon
    m = ConcreteModel()
    m.t_horizon = Param(initialize=t_horizon)
    m.t = ContinuousSet(bounds=(0.0, m.t_horizon))
    m.t_d = RangeSet(0, t_horizon)
    m.n_pwl = Param(initialize=n_pwl)
    m.t_cons = RangeSet(1, m.n_pwl)

    # Parameter definitions
    m.alpha_e = Var(m.t, initialize=0.02, bounds=(0.05, 0.5))
    m.alpha_e_cons = Var(m.t_cons, initialize=0.03, bounds=(0.05, 0.5))
    m.alpha_e_end = Var(initialize=0.03, bounds=(0.05, 0.5))

    m.alpha_i = Var(m.t, initialize=0.05, bounds=(0.01, 0.3))
    m.alpha_i_cons = Var(m.t_cons, initialize=0.05, bounds=(0.01, 0.3))
    m.alpha_i_end = Var(initialize=0.05, bounds=(0.01, 0.3))

    m.latent = Param(initialize=0.5)
    m.gamma_p = Param(initialize=0)

    m.kappa_p = Var(m.t, initialize=0.25, bounds=(0.1, 0.3))
    m.kappa_cons = Var(m.t_cons, initialize=0.25, bounds=(0.1, 0.3))
    m.kappa_end = Var(initialize=0.25, bounds=(0.1, 0.3))

    m.rho_p = Var(initialize=2 * 0.0066177, bounds=(0.1, 0.1))
    m.beta_p = Var(initialize=0.0066177, bounds=(0.001, 0.05))
    m.mu_p = Var(initialize=0.00290, bounds=(0.0001, 0.05))

    # State variables
    m.S = Var(m.t, bounds=(0, 1))  # Susceptible
    m.E = Var(m.t)  # Exposed
    m.E0 = Var(bounds=(0, 1e-6), initialize= 1e-8)
    m.A = Var(m.t, bounds=(0, 1))  # Asymptomatic
    m.I = Var(m.t, bounds=(0, 1))  # Infected
    m.R = Var(m.t, bounds=(0, 1))  # Recovered
    m.P = Var(m.t, bounds=(0, 1))  # Dead

    # Derivative variables
    m.dSdt = DerivativeVar(m.S, wrt=m.t)
    m.dEdt = DerivativeVar(m.E, wrt=m.t)
    m.dAdt = DerivativeVar(m.A, wrt=m.t)
    m.dIdt = DerivativeVar(m.I, wrt=m.t)
    m.dRdt = DerivativeVar(m.R, wrt=m.t)
    m.dPdt = DerivativeVar(m.P, wrt=m.t)

    # Initial conditions
    m.S[0].fix(1)
    m.E[0].fix(value(m.E0))
    m.A[0].fix(0)
    m.I[0].fix(0)
    m.R[0].fix(0)
    m.P[0].fix(0)


    # Differential equations in the model
    def _diffeq1(m, t):
        return m.dSdt[t] == -m.alpha_e[t] * m.S[t] * m.A[t] - m.alpha_i[t] * m.S[t] * m.I[t] + m.gamma_p * m.R[t]


    m.diffeq1 = Constraint(m.t, rule=_diffeq1)


    def _diffeq2(m, t):
        return m.dEdt[t] == m.alpha_e[t] * m.S[t] * m.A[t] + m.alpha_i[t] * m.S[t] * m.I[t] - m.latent * m.E[t]


    m.diffeq2 = Constraint(m.t, rule=_diffeq2)


    def _diffeq3(m, t):
        return m.dAdt[t] == m.latent * m.E[t] - m.kappa_p[t] * m.A[t] - m.rho_p * m.A[t]


    m.diffeq3 = Constraint(m.t, rule=_diffeq3)


    def _diffeq4(m, t):
        return m.dIdt[t] == m.kappa_p[t] * m.A[t] - m.beta_p * m.I[t] - m.mu_p * m.I[t]


    m.diffeq4 = Constraint(m.t, rule=_diffeq4)


    def _diffeq5(m, t):
        return m.dRdt[t] == m.beta_p * m.I[t] - m.gamma_p * m.R[t] # m.rho_p * m.A[t]


    m.diffeq5 = Constraint(m.t, rule=_diffeq5)


    def _diffeq6(m, t):
        return m.dPdt[t] == m.mu_p * m.I[t]


    m.diffeq6 = Constraint(m.t, rule=_diffeq6)

    # Total population
    m.N = Param(initialize=N)


    def data_I_init(m, t):
        return data_I_dict[t]


    m.data_I = Param(m.t_d, initialize=data_I_init)


    def data_P_init(m, t):
        return data_P_dict[t]


    m.data_P = Param(m.t_d, initialize=data_P_init)


    def data_R_init(m, t):
        return data_R_dict[t]


    m.data_R = Param(m.t_d, initialize=data_R_init)

    # # Simulation of DAE system
    m.var_input = Suffix(direction=Suffix.LOCAL)
    m.var_input[m.kappa_p] = {0: 0.0005, 20: 0.00065, 50: 0.0064, 60: 0.1}
    m.var_input[m.alpha_e] = {0: 0.4, 20: 0.45, 50: 0.3, 60: 0.1}
    m.var_input[m.alpha_i] = {0: 0.005, 20: 0.006, 50: 0.006}

    sim = Simulator(m, package='scipy')
    t_sim, sol = sim.simulate(integrator='vode', numpoints=2000, varying_inputs=m.var_input)

    # #  Discretize model using radau collocation
    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, wrt=m.t, nfe=t_horizon, ncp=2)



    def sum_squares(m):
        return sum((m.N * m.I[t_i] - m.data_I[t_i]) ** 2 + (m.N * m.P[t_i] - m.data_P[t_i]) ** 2 +
                   (m.N * m.R[t_i] - m.data_R[t_i]) ** 2 for t_i in m.t_d)

    m.obj = Objective(rule=sum_squares, sense=minimize)

    m.E[0].fixed = False


    def init_E_rule(m):
        return m.E[0] == m.E0



    # Enforcing piecewise linear parameter values
    def alpha_e_pwl_rule(m, t):
        t_int = m.t_horizon / m.n_pwl
        t_discrete = int(t / t_int) + 1
        if t >= value(m.t_horizon):
            t_discrete = int(t / t_int)
        return m.alpha_e[t] == m.alpha_e_cons[t_discrete]


    m.alpha_e_pwl = Constraint(m.t, rule=alpha_e_pwl_rule)


    def alpha_i_pwl_rule(m, t):
        t_int = m.t_horizon / m.n_pwl
        t_discrete = int(t / t_int) + 1
        if t >= value(m.t_horizon):
            t_discrete = int(t / t_int)
        return m.alpha_i[t] == m.alpha_i_cons[t_discrete]


    m.alpha_i_pwl = Constraint(m.t, rule=alpha_i_pwl_rule)


    def kappa_pwl_rule(m, t):
        t_int = m.t_horizon / m.n_pwl
        t_discrete = int(t / t_int) + 1
        if t >= value(m.t_horizon):
            t_discrete = int(t / t_int)
        return m.kappa_p[t] == m.kappa_cons[t_discrete]


    m.kappa_pwl = Constraint(m.t, rule=kappa_pwl_rule)


    def alpha_e_change_init(m, t):
        if t > 1:
            return m.alpha_e_cons[t] <= m.alpha_e_cons[t - 1]
        else:
            return Constraint.Skip


    m.alpha_e_change = Constraint(m.t_cons, rule=alpha_e_change_init)


    def alpha_i_change_init(m, t):
        if t > 1:
            return m.alpha_i_cons[t] <= m.alpha_i_cons[t - 1]
        else:
            return Constraint.Skip


    m.alpha_i_change = Constraint(m.t_cons, rule=alpha_i_change_init)


    def kappa_change_init(m, t):
        if t > 1:
            return m.kappa_cons[t] >= m.kappa_cons[t - 1]
        else:
            return Constraint.Skip


    m.kappa_change = Constraint(m.t_cons, rule=kappa_change_init)

    solver = SolverFactory('ipopt')
    solver.options['mu_init'] = 1e-5
    solver.options['print_user_options'] = 'yes'
    results = solver.solve(m, tee=True)

    MSE_I = sum((m.N * value(m.I[t_i]) - m.data_I[t_i]) ** 2 for t_i in m.t_d)
    MSE_P = sum((m.N * value(m.P[t_i]) - m.data_P[t_i]) ** 2 for t_i in m.t_d)
    MSE_R = sum((m.N * value(m.R[t_i]) - m.data_R[t_i]) ** 2 for t_i in m.t_d)

    print('\n \n \n \n ')
    print('******************************************************************************')
    print('                    List of parameters obtained for %s               ' % (country_name))
    print('******************************************************************************')
    print('latent:      %1.5f ' % (value(m.latent)))
    print('gamma:       %1.5f ' % (value(m.gamma_p)))
    print('rho:         %1.5f ' % (value(m.rho_p)))
    print('beta:        %1.5f ' % (value(m.beta_p)))
    print('mu:          %1.5f ' % (value(m.mu_p)))
    print('E0:          %1.5f ' % (value(m.N * m.E0)))
    print('******************************************************************************')
    print('RMSE(I) = %8.2f         RMSE(P) = %8.2f         RMSE(R) = %8.2f  ' % (
    MSE_I ** 0.5, MSE_P ** 0.5, MSE_R ** 0.5))
    print('******************************************************************************')

    print('\n \n \n \n ')

    return m

def plotting_fun(m,country_name, t_f):

    data_I_dict, N = data_read.return_data_ready('I', country_name)
    data_P_dict, N = data_read.return_data_ready('P', country_name)
    data_R_dict, N = data_read.return_data_ready('R', country_name)
    
    for key in data_I_dict.keys():
        data_I_dict[key] = data_I_dict[key] - data_P_dict[key] - data_R_dict[key]

    S = []
    E = []
    A = []
    I = []
    R = []
    P = []
    t = []
    kappa_p = []
    alpha_e = []
    alpha_i = []

    for i in sorted(m.t):
        t.append(i)
        S.append(value(m.S[i]))
        E.append(value(m.E[i]))
        A.append(value(m.A[i]))
        I.append(value(m.I[i]))
        R.append(value(m.R[i]))
        P.append(value(m.P[i]))
        kappa_p.append(value(m.kappa_p[i]))
        alpha_e.append(value(m.alpha_e[i]))
        alpha_i.append(value(m.alpha_i[i]))


    alpha_i_dict, alpha_e_dict, kappa_dict = {}, {}, {}
    i = 0
    for j in t:
        alpha_i_dict[j], alpha_e_dict[j], kappa_dict[j] = alpha_i[i], alpha_e[i], kappa_p[i]
        i +=1


    # plt.gca().set_prop_cycle(None)

    # plt.plot(t, S, label='S')
    plt.plot(t, E, label='E')
    plt.plot(t, A, label='A')
    plt.plot(t, I, label='I')
    plt.plot(t, R, label='R')
    plt.plot(t, P, label='P')
    plt.xlabel('Time (days)')
    plt.ylabel('Normalized populations')
    plt.tick_params(axis='both', direction='in', top=True, right=True)
    plt.legend(loc='best')
    plt.grid()
    plt.title('Simmulation for: '+country_name)
    plt.draw()

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(t, alpha_e, 'k', label=r'$\alpha_e(t)$')
    plt.tick_params(axis='both', direction='in', top=True, right=True)
    ax1.grid()
    ax1.legend(loc='best')
    ax2.plot(t, alpha_i, 'r', label=r'$\alpha_i(t)$')
    ax2.tick_params(axis='both', direction='in', top=True, right=True)
    ax2.grid()
    ax2.legend(loc='best')
    ax3.plot(t, kappa_p, 'b', label=r'$\kappa(t)$')
    plt.xlabel('Time (days)')
    ax3.tick_params(axis='both', direction='in', top=True, right=True)
    ax3.grid()
    ax3.legend(loc='best')
    plt.draw()



    # Simulating future time steps and plotting data and predictions
    params = {}

    params['latent'] = value(m.latent)
    params['gamma'] = value(m.gamma_p)
    params['rho'] = value(m.rho_p)
    params['beta'] = value(m.beta_p)
    params['mu'] = value(m.mu_p)

    params['S0'] = S[-1]
    params['E0'] = E[-1]
    params['A0'] = A[-1]
    params['I0'] = I[-1]
    params['R0'] = R[-1]
    params['P0'] = P[-1]

    params['N'] = N
    params['alpha_e'] = alpha_e[-1]
    params['alpha_i'] = alpha_i[-1]
    params['kappa'] = kappa_p[-1]

    sim, t_sim, sol = model_simulation.simulation_fun(params, t[-1], t_f)

    plt.rcParams['lines.markersize'] = 3

    fig = plt.figure()
    time_data = list(data_I_dict.keys())
    inf_data = list(data_I_dict.values())
    plt.scatter(time_data, inf_data, color="black", label='data')
    I_scaled1 = [N * i for i in sol[:, 3]]
    I_scaled2 = [N * i for i in I]
    plt.plot(t_sim, I_scaled1, 'r', label='model sim')
    plt.plot(t, I_scaled2, 'r--', label='model opt')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Time (days)')
    plt.ylabel('Infected people')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title(country_name)
    plt.draw()

    fig = plt.figure()
    time_data = list(data_P_dict.keys())
    dead_data = list(data_P_dict.values())
    plt.scatter(time_data, dead_data, color="black", label='data')
    P_scaled1 = [N * i for i in sol[:, 5]]
    P_scaled2 = [N * i for i in P]
    plt.plot(t_sim, P_scaled1, 'r', label='model sim')
    plt.plot(t, P_scaled2, 'r--', label='model opt')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Time (days)')
    plt.ylabel('Dead people')
    plt.title(country_name)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.draw()

    fig = plt.figure()
    time_data = list(data_R_dict.keys())
    rec_data = list(data_R_dict.values())
    plt.scatter(time_data, rec_data, color="black", label='data')
    R_scaled1 = [N * i for i in sol[:, 4]]
    R_scaled2 = [N * i for i in R]
    plt.plot(t_sim, R_scaled1, 'r', label='model sim')
    plt.plot(t, R_scaled2, 'r--', label='model opt')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Time (days)')
    plt.ylabel('Receovered people')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title(country_name)
    plt.draw()

    plt.show()



gc.collect()
