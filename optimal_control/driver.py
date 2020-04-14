'''
Author: Fernando Lejarza (lejarza@utexas.edu)
Affilition: The University of Texas at Austin
Last modified: 04.13.2020
'''


from __future__ import division
import matplotlib.pyplot as plt
import dae_model_optimization


''' List of model country-sprecific parameters '''
params = {}
params['latent'], params['gamma'], params['rho'], params['beta'] = [0.50000, 0, 0.10000, 0.00670]
params['mu'], params['N'] = [0.00410, 327167434.0]
params['S0'] = 0.9977558755803503
params['E0'] = 0.0003451395725394549
params['A0'] = 0.00037846880968213874
params['I0'] = 337072.0/params['N']
params['R0'] = 17448.0/params['N']
params['P0'] = 9619.0/params['N']

'''Optimal control function.
    Inputs: 
        - params: SEAIRP model parameters 
        - t_init: initial time point 
        - t_f: final time point
        - n_pwl: number of piece-wise linear segments for inputs
        - i_peak: infeacted peak'''
m = dae_model_optimization.policy_opt_fun(params,0,100,20,1e6)
dae_model_optimization.plotting_fun(m)


plt.draw()


