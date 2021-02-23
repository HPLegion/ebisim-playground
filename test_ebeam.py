from ebisim.simulation import boltzmann_radial_potential_linear_density
from ebisim.simulation._radial_dist import boltzmann_radial_potential_linear_density_ebeam
from ebisim.physconst import Q_E, M_E, EPS_0, PI
import numpy as np
import matplotlib.pyplot as plt
from ebisim.beams import ElectronBeam

e = ElectronBeam(1, 2, 0.005, 0, 0.006, 1300)
E_KIN = 8000 + e.space_charge_correction(8000)
R_E = e.herrmann_radius(8000)
N = 500
V_E = np.sqrt(2 * E_KIN * Q_E/M_E)
R_D = 0.005
I = 1.
RHO_0 = -I/(V_E * PI * R_E**2)
def analytical_solution(r):
    f = r<R_E
    nf = np.logical_not(f)
    phi = np.zeros_like(r)
    phi0 = I/(4*PI*EPS_0*V_E)
    phi[f] = phi0 * ((r[f]/R_E)**2 + 2*np.log(R_E/R_D)-1)
    phi[nf] = phi0 * 2*np.log(r[nf]/R_D)
    return phi

NLI = np.array([4.2095955e+09, 2.2095955e+09])
KT = np.array([150, 50])
Q = np.array((5, 3))

r = np.geomspace(R_E/100, R_D, N)
r[0] = 0
rho = np.zeros(N)
rho[r < R_E] = RHO_0

phi_b, _, __ = boltzmann_radial_potential_linear_density_ebeam(r, I, R_E, 8000, 0, 1, 1)
phi_c, _, __ = boltzmann_radial_potential_linear_density(r, rho,NLI[:, np.newaxis], KT[:, np.newaxis], Q[:, np.newaxis])


phi, n, shape = boltzmann_radial_potential_linear_density_ebeam(
    r, I, R_E, 8000, NLI[:, np.newaxis], KT[:, np.newaxis], Q[:, np.newaxis]
)

plt.figure()
plt.plot(r, phi)
plt.plot(r, phi_b)
plt.plot(r, phi_c)

plt.show()