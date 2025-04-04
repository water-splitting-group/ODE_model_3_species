import numpy as np
import matplotlib.pyplot as plt
#import Liquid_Phase_O2_Analysis as lp
#from Reaction_ODE_Fitting import ODE_matrix_fit_func, reaction_string_to_matrix, reaction_string_to_numba_matrix
#from utility_functions import scientific_notation, plot_func
from scipy.integrate import odeint
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
from timeit import default_timer as timer
import pprint
from numba import njit


def ODE_system(y, t, p, cross_section, flux):
	'''ODE System assuming following reaction sequence:
	B -> A (life time p[0])
	A -- hv --> B (excitation probability p[1])
	B -- hv --> C (excitation probability p[2])
	C -> A (life time p[3])
	'''

	R1 = p[0] * y[1]
	R2 = p[1] * cross_section * flux * y[0]  
	R3 = p[2] * cross_section * flux * y[1]
	R6 = p[3] * y[2]

	ra = R1 - R2 + R6 
	rb = -R1 + R2 - R3
	rc = R3 - R6
	

	return [ra, rb, rc]

def ODE_explicit_rate_law(p, initial_state, t, flux, cross_section, ravel = False):

	sol = odeint(ODE_system, initial_state, t, args = (p, cross_section, flux))

	if ravel is True:
		sol = np.ravel(sol)
    
	return sol

def ODE_system_VF(y, t, p, cross_section, flux):
	'''
    Thedifference is in this case the flux driving the reaction
    from B--->C decreases directly as a result of the photons absorbed in A--->B
    The idea that I had was that each for each individual second that passes 1*quantum yield photons would be lost from the flux
	'''

	R1 = p[0] * y[1]
	R2 = p[1] * cross_section * flux * y[0]  
	R3 = p[2] * cross_section * flux *(1-p[1]*y[0]*t) * y[1]
	R6 = p[3] * y[2]
    
	ra = R1 - R2 + R6 
	rb = -R1 + R2 - R3
	rc = R3 - R6
	

	return [ra, rb, rc]

def ODE_explicit_rate_law_VF(p, initial_state, t, flux, cross_section, ravel = False):

	sol = odeint(ODE_system_VF, initial_state, t, args = (p, cross_section, flux))

	if ravel is True:
		sol = np.ravel(sol)
    
	return sol

initial=[10, 0, 0] 
'''Initial is a Vector of the initial amount of each species A, B, and C'''

t=np.linspace(0, 7, 200)

odds=[0.03, 0.34, 0.24, 0.1]
'''I was having a little bit of trouble knowing exactly where to find good theoretical probabilities,
 so I chose values that are somewhat arbitray. I would like to better know where I could get sensible values from'''
 
flux_phi=10
cross_section=0.2

result=ODE_explicit_rate_law(odds, initial, t, flux_phi, cross_section)

plt.figure(0)
plt.title("Concentration-Time graph of species A, B, and C")
plt.plot(t,result[:, 0], 'r--', label=r'[A] where $\frac{d[A]}{dt}=p_{BA}[B]-p_{AB}\sigma\phi[A]+p_{CA}[C]$')
plt.plot(t,result[:, 1], 'y--', label=r'[B] where $\frac{d[B]}{dt}=-p_{BA}[B]+p_{AB}\sigma\phi[A]-p_{BC}\sigma\phi[B]$')
plt.plot(t,result[:, 2], 'b--', label=r'[C] where $\frac{d[C]}{dt}=p_{BC}\sigma\phi[B]-p_{CA}[C]$')
plt.legend(loc='best')
plt.ylabel('Concentration') 
'''Concentration currently framed like percentage because I found that intuitive'''
plt.xlabel('Time')
'''Seconds? Hours? Fortnights?'''




result_VF=ODE_explicit_rate_law(odds, initial, t, flux_phi, cross_section)
plt.figure(1)
plt.title("Concentration-Time of Species C where the flux varies based on [A]")
plt.plot(t,result[:, 2], 'g--', label=r'[C] where $\frac{d[C]}{dt}=p_{BC}\sigma\phi(1-p_{AB}[A]*t)[B]-p_{CA}[C]$')
plt.legend(loc='best')
plt.ylabel('Concentration') 
plt.xlabel('Time')