# -*- coding: utf-8 -*-
"""

Plot accumulated distributions of different methods

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator
timestr = time.strftime("%Y%m%d-%H%M%S")

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'font.size': 40})
plt.rc('xtick', labelsize=40) 
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm

# Initialize the variables for each method
VQE_StV_energies = None
VQE_QPU_energies = None
QA_energies = None
SA_energies = None
random_energies = None
VQE_StV_probs = None
VQE_QPU_probs = None
QA_probs = None
SA_probs = None
random_probs = None

# Define the method names
method = ['VQEcVARstV', 'VQEcVARibm_fez', 'SA', 'Random', 'QA']
method_name = ['VQE_StV', 'VQE_QPU', 'SA', 'random', 'QA']

# Preprocessing: Load the data for each method
for i, m in enumerate(method):
    # Load energy data and assign it to the corresponding variable
    # might need to trim file names to get working or use - *
    globals()[f'{method_name[i]}_energies'] = np.loadtxt(f'distr_data/Energy_data_prepro_{m}.txt')
    # Load probability data and assign it to the corresponding variable
    globals()[f'{method_name[i]}_probs'] = np.loadtxt(f'distr_data/Probs_data_prepro_{m}.txt')


n = 6
colour = cm.viridis(np.linspace(0, 1, n+1))

# Some probabilities might need renormalised
SA_probs = SA_probs*(1/np.sum(SA_probs))
VQE_StV_probs = VQE_StV_probs*(1/np.sum(VQE_StV_probs))
VQE_QPU_probs = VQE_QPU_probs*(1/np.sum(VQE_QPU_probs))
QA_probs = QA_probs*(1/np.sum(QA_probs))
# random_probs = random_probs*(1/np.sum(random_probs))

# x-axis range
lower_range = -22
upper_range = 100

# First plot 
plt.figure(figsize=(12, 6))
plt.bar(SA_energies, SA_probs, color=colour[0], label='Simulated Annealing')
plt.axvline(x=-20, color='r', linestyle='--')  # Thinner line
plt.xlim(-21, -17)
plt.ylim(0, 1)
plt.ylabel('Probability')
plt.xlabel('Energy (arbitrary units)')
plt.legend(loc="upper right")
plt.grid(True, which='both')
plt.tick_params(labelbottom=True)
# Get current Axes object
ax = plt.gca()
# Modify x-ticks
xticks = ax.get_xticks().tolist()
if -21 in xticks:
    xticks.remove(-21)
if -17 in xticks:
    xticks.remove(-17)
ax.set_xticks(xticks)
plt.savefig('final_plots/SA_distribution_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Second plot (VQE - State Vector)
plt.figure(figsize=(12, 6))
plt.bar(VQE_StV_energies, VQE_StV_probs, color=colour[1], label='VQE - State Vector')
plt.axvline(x=-20, color='r', linestyle='--', label="Min QUBO Energy")  # Thinner line
plt.xlim(-21, 105) #70 105
plt.ylim(0, 0.35)
plt.ylabel('Probability')
plt.xlabel('Energy (arbitrary units)')
plt.legend(loc="upper right")
plt.grid(True, which='both')
plt.tick_params(labelbottom=True)
plt.savefig('final_plots/VQE_StV_distribution_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Third plot (VQE - QPU)
VQE_QPU_probs = VQE_QPU_probs * (1 / VQE_QPU_probs.sum())
plt.figure(figsize=(12, 6))
plt.bar(VQE_QPU_energies, VQE_QPU_probs, color=colour[2], label='VQE - QPU')
plt.axvline(x=-20, color='r', linestyle='--')  # Thinner line
plt.xlim(lower_range, 105)
plt.ylim(0, 0.35)
plt.ylabel('Probability')
plt.xlabel('Energy (arbitrary units)')
plt.legend(loc="upper right")
plt.grid(True, which='both')
plt.tick_params(labelbottom=True)
plt.savefig('final_plots/VQE_QPU_distribution_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Fourth plot (Quantum Annealing)
plt.figure(figsize=(12, 6))
plt.bar(QA_energies, QA_probs, color=colour[3], label='Quantum Annealing')
plt.axvline(x=-20, color='r', linestyle='--')  # Thinner line
plt.xlim(-21, -9) # -12
plt.ylabel('Probability')
plt.xlabel('Energy (arbitrary units)')
plt.legend(loc="upper right")
plt.grid(True, which='both')
plt.tick_params(labelbottom=True)
# Get current Axes object
ax = plt.gca()
# Modify x-ticks
# xticks = ax.get_xticks().tolist()

# print(xticks)
# if -12.5 in xticks:
#     xticks.remove(-12.5)
    
# if -17.5 in xticks:
#     xticks.remove(-17.5)
# if -10 in xticks:
#     xticks.remove(-10)
# if -22.5 in xticks:
#     xticks.remove(-22.5)    
xticks = [-20,-18,-16,-14,-12,-10]

ax.set_xticks(xticks)
plt.savefig('final_plots/QA_distribution_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Fifth plot (Random Sampling)
plt.figure(figsize=(12, 6))
plt.bar(random_energies, random_probs, color=colour[4], label='Random Sampling')
plt.axvline(x=-20, color='r', linestyle='--')  # Thinner line
plt.xlim(lower_range, 0)
plt.ylabel('Probability')
plt.xlabel('Energy (arbitrary units)')
plt.legend(loc="upper right")
plt.grid(True, which='both')
plt.tick_params(labelbottom=True)
plt.savefig('final_plots/random_distribution_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()


#############################################################################

# Now post-selected distributions

## Pre processing
VQE_StV_energies = None
VQE_QPU_energies = None
QA_energies = None
SA_energies = None
random_energies = None

VQE_StV_probs = None
VQE_QPU_probs = None
QA_probs = None
SA_probs = None
random_probs = None

# Define the method names
method = ['VQEcVARstV', 'VQEcVARibm_fez', 'SA', 'Random', 'QA']
method_name = ['VQE_StV', 'VQE_QPU', 'SA', 'random', 'QA']

# Preprocessing: Load the data for each method
for i, m in enumerate(method):
    # Load energy data and assign it to the corresponding variable
    globals()[f'{method_name[i]}_energies'] = np.loadtxt(f'distr_data/Energy_data_postpro_{m}.txt')

    # Load probability data and assign it to the corresponding variable
    globals()[f'{method_name[i]}_probs'] = np.loadtxt(f'distr_data/Probs_data_postpro_{m}.txt')


# !!! VQE_StV_probs need renormalised
SA_probs = SA_probs*(1/np.sum(SA_probs))
VQE_StV_probs = VQE_StV_probs*(1/np.sum(VQE_StV_probs))
VQE_QPU_probs = VQE_QPU_probs*(1/np.sum(VQE_QPU_probs))
QA_probs = QA_probs*(1/np.sum(QA_probs))
# random_probs = random_probs*(1/np.sum(random_probs))

# Create a figure with 3x1 subplots
fig, axs = plt.subplots(5, 1, figsize=(8,20))

max_x = -17
min_x = -21

labelfontsize = 25

# First subplot (VQE State Vector)
axs[0].bar(SA_energies, SA_probs, color=colour[0], label='Simulated Annealing')
axs[0].axvline(x=-20, color='r', linestyle='--', label="Min QUBO Energy")
axs[0].set_xlim(min_x,max_x)
axs[0].set_ylim(0, 1)
# axs[0].set_ylabel('Probability')
# axs[0].set_title('18x18 QUBO, 3 vacancies, Post-selection')
axs[0].legend(loc="upper right",fontsize=labelfontsize)
axs[0].grid(True, which='both',axis='y')
# axs[0].minorticks_on()
# axs[0].grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
axs[0].tick_params(labelbottom=False)
yticks = axs[0].get_yticks().tolist()
if 1 in yticks:
    yticks.remove(1)
axs[0].set_yticks(yticks)


# Second subplot (QA)
axs[1].bar(VQE_StV_energies, VQE_StV_probs, color=colour[1], label='VQE - State Vector')
axs[1].axvline(x=-20, color='r', linestyle='--')
axs[1].set_xlim(min_x,max_x)
axs[1].set_ylim(0, 1)
# axs[1].set_ylabel('Probability', fontsize=30)
axs[1].legend(loc="upper right",fontsize=labelfontsize)
axs[1].grid(True, which='both',axis='y')
# axs[1].minorticks_on()
# axs[1].grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
axs[1].tick_params(labelbottom=False)
yticks = axs[2].get_yticks().tolist()
if 1 in yticks:
    yticks.remove(1)
axs[1].set_yticks(yticks)


# Second subplot (QA)
axs[2].bar(VQE_QPU_energies, VQE_QPU_probs, color=colour[2], label='VQE - QPU')
axs[2].axvline(x=-20, color='r', linestyle='--')
axs[2].set_xlim(min_x,max_x)
axs[2].set_ylim(0, 1)
axs[2].set_ylabel('Probability', fontsize=40)
axs[2].legend(loc="upper right",fontsize=labelfontsize)
axs[2].grid(True, which='both',axis='y')
# axs[1].minorticks_on()
# axs[1].grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
axs[2].tick_params(labelbottom=False)
yticks = axs[2].get_yticks().tolist()
if 1 in yticks:
    yticks.remove(1)
axs[2].set_yticks(yticks)


# Third subplot (SA)
axs[3].bar(QA_energies, QA_probs, color=colour[3], label='Quantum Annealing')
axs[3].axvline(x=-20, color='r', linestyle='--')
axs[3].set_xlim(min_x,max_x)
axs[3].set_ylim(0, 1)
# axs[2].set_xlabel('Energy', fontsize=30)
# axs[2].set_ylabel('Probability')
axs[3].legend(loc="upper right",fontsize=labelfontsize)
axs[3].grid(True, which='both',axis='y')
# axs[2].minorticks_on()
# axs[2].grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
axs[3].tick_params(labelbottom=False)
yticks = axs[3].get_yticks().tolist()
if 1 in yticks:
    yticks.remove(1)
axs[3].set_yticks(yticks)


# print('rand probs sum', random_probs.sum())
# Third subplot (SA)
axs[4].bar(random_energies, random_probs, color=colour[4], label='Random Sampling')
axs[4].axvline(x=-20, color='r', linestyle='--')
axs[4].set_xlim(min_x,max_x)
axs[4].set_ylim(0, 1)
axs[4].set_xlabel('Energy (arbitrary units)', fontsize=40)
# axs[2].set_ylabel('Probability')
axs[4].legend(loc="upper right",fontsize=labelfontsize)
axs[4].grid(True, which='both',axis='y')
# axs[2].minorticks_on()
# axs[2].grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
# Remove the '5' tick label from the y-axis
# yticks = axs[4].get_yticks().tolist()
if 1 in yticks:
    yticks.remove(1)
axs[4].set_yticks(yticks)


for ax in axs:
    ax.set_xticks([-20, -19, -18])

# Adjust layout to remove space between subplots
plt.subplots_adjust(hspace=0.05)
# Save the figure
plt.savefig('final_plots/final_distribution_plot_Post_selection_18x18_3v_vertical.pdf', format='pdf', bbox_inches='tight')
plt.show()