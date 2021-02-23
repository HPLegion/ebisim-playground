import os
import numpy as np
from numba import njit


EPS_0 = 8.854187817620389e-12
PI = 3.141592653589793
M_E = 9.1093837015e-31
Q_E = 1.602176634e-19

@njit(cache=True)
def tridiagonal_matrix_algorithm(l, d, u, b):
    """
    Tridiagonal Matrix Algorithm [TDMA]_.
    Solves a system of equations M x = b for x, where M is a tridiagonal matrix.

    M = np.diag(d) + np.diag(u[:-1], 1) + np.diag(l[1:], -1)

    Parameters
    ----------
    l : np.ndarray
        Lower diagonal vector l[i] = M[i, i-1].
    d : np.ndarray
        Diagonal vector d[i] = M[i, i].
    u : np.ndarray
        Upper diagonal vector u[i] = M[i, i+1].
    b : np.ndarray
        Inhomogenety term.

    Returns
    -------
    x : np.ndarray
        Solution of the linear system.

    References
    ----------
    .. [TDMA] "Tridiagonal matrix algorithm"
           https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    """
    n = l.size
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)
    cp[0] = u[0]/d[0]
    dp[0] = b[0]/d[0]
    for k in range(1, n):
        cp[k] = u[k]               /(d[k]-l[k]*cp[k-1])
        dp[k] = (b[k]-l[k]*dp[k-1])/(d[k]-l[k]*cp[k-1])
    x[-1] = dp[-1]
    for k in range(n-2, -1, -1):
        x[k] = dp[k] - cp[k]*x[k+1]
    return x



@njit(cache=True)
def fd_system_nonuniform_grid(r):
    """
    Sets up the three diagonal vectors for a finite Poisson problem with radial symmetry on a
    nonuniform grid.
    d phi/dr = 0 at r = 0, and phi = phi0 at r = (n-1) * dr = r_max
    The finite differences are developed according to [Sundqvist1970]_.

    Parameters
    ----------
    r : np.ndarray
        <m>
        Radial grid points, with r[0] = 0, r[-1] = r_max.

    Returns
    -------
    l : np.ndarray
        Lower diagonal vector.
    d : np.ndarray
        Diagonal vector.
    u : np.ndarray
        Upper diagonal vector.

    References
    ----------
    .. [Sundqvist1970] "A simple finite-difference grid with non-constant intervals",
            Sundqvist, H., & Veronis, G.,
            Tellus, 22(1), 26–31 (1970),
            https://doi.org/10.3402/tellusa.v22i1.10155

    See Also
    --------
    ebisim.simulation._radial_dist.fd_system_uniform_grid
    """
    dr = r[1:] - r[:-1]
    n = r.size

    weight1 = 2/(dr[1:] * dr[:-1] * (dr[1:] + dr[:-1]))
    weight2 = 1/(r[1:-1]*(dr[1:]**2 * dr[:-1] + dr[1:] * dr[:-1]**2))

    d = np.zeros(n)
    d[0] = -2/dr[0]**2
    d[1:-1] = -(dr[:-1] + dr[1:]) * weight1 + (dr[1:]**2 - dr[:-1]**2) * weight2
    d[-1] = 1

    l = np.zeros(n)
    l[1:-1] = dr[1:] * weight1 - dr[1:]**2 * weight2

    u = np.zeros(n)
    u[0] = 2/dr[0]**2
    u[1:-1] = dr[:-1] * weight1 + dr[:-1]**2 * weight2

    return l, d, u

@njit(cache=True)
def radial_potential_nonuniform_grid(r, rho):
    """
    Solves the radial Poisson equation on a nonuniform grid.
    Boundary conditions are d phi/dr = 0 at r = 0 and phi(rmax) = 0

    Parameters
    ----------
    r : np.ndarray
        <m>
        Radial grid points, with r[0] = 0, r[-1] = r_max.
    rho : np.ndarray
        <C/m^3>
        Charge density at r.

    Returns
    -------
    np.ndarray
        Potential at r.
    """
    l, d, u = fd_system_nonuniform_grid(r)
    rho_ = rho.copy()
    rho_[-1] = 0 #Boundary condition
    phi = tridiagonal_matrix_algorithm(l, d, u, -rho/EPS_0)
    return phi

@njit(cache=True)
def boltzmann_radial_potential_linear_density_ebeam(
        r, current, r_e, e_kin, nl, kT, q, first_guess=None, ldu=None
    ):
    """
    Solves the Boltzmann Poisson equation for a static background charge density rho_0 and particles
    with line number density n, Temperature kT and charge state q.
    The electron beam charge density is computed from a uniform current density and
    the iteratively corrected velocity profile of the electron beam.

    Below, nRS and nCS are the number of radial sampling points and charge states.

    Solution is found through Newton iterations, cf. [PICNPS]_.

    Parameters
    ----------
    r : np.ndarray
        <m>
        Radial grid points, with r[0] = 0, r[-1] = r_max.
    current : float
        <A>
        Electron beam current (positive sign).
    r_e : float
        <m>
        Electron beam radius.
    e_kin : float
        <eV>
        Uncorrected electron beam energy.
    nl : np.ndarray (1, nCS)
        <1/m>
        Line number density of Boltzmann distributed particles.
    kT : np.ndarray (1, nCS)
        <eV>
        Temperature of Boltzmann distributed particles.
    q : np.ndarray (1, nCS)
        Charge state of Boltzmann distributed particles.
    ldu : (np.ndarray, np.ndarray, np.ndarray)
        The lower diagonal, diagonal, and upper diagonal vector describing the finite difference
        scheme. Can be provided if they have been pre-computed.

    Returns
    -------
    phi : np.ndarray (nRS, )
        <V>
        Potential at r.
    nax : np.ndarray (1, nCS)
        <1/m^3>
        On axis number densities.
    shape : np.ndarray (nRS, nCS)
        Radial shape factor of the particle distributions.

    References
    ----------
    .. [PICNPS] "Nonlinear Poisson Solver"
           https://www.particleincell.com/2012/nonlinear-poisson-solver/
    """
    # Solves the nonlinear radial poisson equation for a dynamic charge distribution following
    # the Boltzmann law
    # A * phi = b0 + bx (where b0 and bx are the static and dynamic terms)
    # Define cost function f = A * phi - b0 - bx
    # Compute jacobian J = A - diag(d bx_i / d phi_i)
    # Solve J y = f
    # Next guess: phi = phi - y
    # Iterate until adjustment is small
    cden = np.zeros(r.size)
    cden[r < r_e] = -current/PI/r_e**2


    if ldu is not None:
        l, d, u = ldu
    else:
        l, d, u = fd_system_nonuniform_grid(r) # Set up tridiagonal system
    # A = np.diag(d) + np.diag(u[:-1], 1) + np.diag(l[1:], -1)

    if first_guess is None:
        phi = radial_potential_nonuniform_grid(r, cden/np.sqrt(2 * Q_E * e_kin/M_E))
    else:
        phi = first_guess

    nl = np.atleast_2d(np.asarray(nl))
    kT = np.atleast_2d(np.asarray(kT))
    q = np.atleast_2d(np.asarray(q))

    for _ in range(500):

        shape = np.exp(-q * (phi - phi[0])/kT)
        i_sr = np.atleast_2d(np.trapz(r*shape, r)).T
        nax = nl / 2 / PI / i_sr

        # dynamic rhs term
        _bx_a = - nax * q * shape * Q_E / EPS_0 # dynamic rhs term
        _bx_b = - cden/np.sqrt(2 * Q_E * (e_kin+phi)/M_E) / EPS_0
        _bx_a[:, -1] = 0  # boundary condition
        bx = np.sum(_bx_a, axis=0) + _bx_b

        # F = A.dot(phi) - (b0 + bx)
        f = d * phi - bx # Target function
        f[:-1] += u[:-1] * phi[1:]
        f[1:] += l[1:] * phi[:-1]

        _c = np.zeros_like(shape)
        _c[:, :-1] = r[:-1] * (r[1:]-r[:-1]) * shape[:, :-1]
        #Diagonal of the Jacobian df/dphi_i
        j_d = -(np.sum(_bx_a * q/kT *(i_sr-_c)/i_sr, axis=0)
                + Q_E/M_E*_bx_b/(2 * Q_E * (e_kin+phi)/M_E))#Diagonal of the Jacobian df/dphi_i

        y = tridiagonal_matrix_algorithm(l, d - j_d, u, f)
        res = np.linalg.norm(y)/phi.size
        phi = phi - y
        if res < 1e-3:
            break
    return phi, nax, shape


if __name__ == "__main__":
    rgrid = np.geomspace(1e-6, 0.005, 500)
    rgrid[0] = 0
    current = 1
    r_e = 1.e-4
    e_kin = 8000
    nl = np.atleast_2d(np.full(60, 1.e7)).T
    q = np.atleast_2d(np.arange(60)).T
    kT = 80*(q+1)

    ldu = fd_system_nonuniform_grid(rgrid)

    phi, nax, shape = boltzmann_radial_potential_linear_density_ebeam(
        rgrid, current, r_e, e_kin, nl, kT, q, first_guess=None, ldu=ldu
    )
    print("Still alive")
