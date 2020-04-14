'''
Author: Fernando Lejarza (lejarza@utexas.edu)
Affilition: The University of Texas at Austin
Last modified: 04.13.2020
'''


import pandas as pd
import matplotlib.pyplot as plt

world_confirmed = pd.read_csv("data/time_series_covid19_confirmed_global.csv")
world_deaths = pd.read_csv("data/time_series_covid19_deaths_global.csv")
world_recovered = pd.read_csv("data/time_series_covid19_recovered_global.csv")


world_population = pd.read_csv("data/world_population.csv")

replace_dict = {'Bahamas, The': 'Bahamas', 'Brunei Darussalam': 'Brunei', 'Czech Republic': 'Czechia', 'Egypt, Arab Rep.': 'Egypt', 'Gambia, The': 'Gambia', 'Iran, Islamic Rep.': 'Iran', 'Korea, Rep.': 'Korea, South', 'Kyrgyz Republic': 'Kyrgyzstan', 'Russian Federation': 'Russia', 'St. Lucia': 'Saint Lucia', 'St. Vincent and the Grenadines': 'Saint Vincent and the Grenadines', 'Slovak Republic': 'Slovakia', 'United States': 'US', 'Venezuela, RB': 'Venezuela', 'Syrian Arab Republic': 'Syria'}

world_population.replace({'Country Name': replace_dict}, inplace = True)

list_of_countriesA = world_confirmed['Country/Region'].unique()
list_of_countriesB = world_population['Country Name'].unique()


def country_data(df, country_name):
    df = df.groupby(['Country/Region']).sum()
    # index = df['Country/Region'].isin(country_name)
    cols_all = df.columns
    cols_remove = ['Province/State', 'Lat', 'Long']
    cols_keep = [i for i in cols_all if i not in cols_remove]
    country_info = df.loc[country_name, cols_keep]
    # country_info.set_index('Country/Region', inplace=True)

    return country_info


def plot_country_data(df, country_names):
    country_info = country_data(df, country_names)
    country_info.plot()
    plt.draw()


def return_data_ready(data_type, countries):
    N = world_population[world_population['Country Name'] == countries].iloc[0]['2018']

    if data_type == 'I':
        country_info = country_data(world_confirmed, countries)
        country_info_list = country_info.tolist()
        country_info_dic = {}
        for i in range(0, len(country_info_list)):
            country_info_dic[i] = country_info_list[i]

        return country_info_dic, N

    if data_type == 'P':
        country_info = country_data(world_deaths, countries)
        country_info_list = country_info.tolist()
        country_info_dic = {}
        for i in range(0, len(country_info_list)):
            country_info_dic[i] = country_info_list[i]

    if data_type == 'R':
        country_info = country_data(world_recovered, countries)
        country_info_list = country_info.tolist()
        country_info_dic = {}
        for i in range(0, len(country_info_list)):
            country_info_dic[i] = country_info_list[i]

    return country_info_dic, N
