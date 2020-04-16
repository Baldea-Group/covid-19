'''
Author: Fernando Lejarza (lejarza@utexas.edu)
Affilition: The University of Texas at Austin
Last modified: 04.13.2020
'''

import dae_model_parameter_estimation
import gc

'''
The COVID-19 data used for parameter estimation was obtained from:  https://github.com/CSSEGISandData/COVID-19
Additonally, countries populations for up to 2018 was obatined from: https://data.worldbank.org/indicator/sp.pop.totl
The data is located in  and is read directly from the repository './data' 
'''

'''Parameter estimation function.
    Inputs: 
        - country_name: string containing the name of the country (refer the list obtained in data_read.py for the 
                        correct list of countries)  
        - n_pwl: number of piece-wise linear segments for inputs
 '''
m = dae_model_parameter_estimation.param_estimation_fun('US', 15)

'''Plotting and simulation function.
    Inputs: 
        - m: Pyomo model with solutions to the parameter estimation problem
        - country_name: string containing the name of the country (refer the list obtained in data_read.py for the 
                        correct list of countries)  
        - t_horizon: time horizon for simulation of the dynamic SEAIRP model 
 '''
dae_model_parameter_estimation.plotting_fun(m, 'US', 90)

gc.collect()
