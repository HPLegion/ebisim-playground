from ebisim.simulation._radial_dist import boltzmann_radial_potential_linear_density
import numpy as np
from ebisim.physconst import *
N = 50
E_KIN = 5000
V_E = np.sqrt(2 * E_KIN * Q_E/M_E)
R_E = 0.0001
R_D = 0.005
I = 1
RHO_0 = -I/(V_E * PI * R_E**2)
E0 = -I/(2*PI*EPS_0*V_E*R_E)
r = np.geomspace(R_E/100, R_D, N)
r[0] = 0
rho = np.zeros(N)
rho[r<R_E] = RHO_0
rho[-1] = 0
N0 = -RHO_0/5/Q_E
T0 = 60.0
phi_l, n, shape = boltzmann_radial_potential_linear_density(r, rho, [4.2095955e+09, 2.2095955e+09], [1*T0, 0.5*T0], [1, 2])
