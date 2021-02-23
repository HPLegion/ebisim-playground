import ebisim
from string import Template
import math
import numba
import numpy as np
import timeit
import textwrap
from numba.types import(
    int64 as nint64,
    float64 as nfloat64
)

M_E_EV = ebisim.M_E_EV

def eixs_code(element):
    """
    Electron ionisation cross section according to a simplified version of the models given in [1]_.

    Parameters
    ----------
    element : ebisim.Element
        An ebisim.Element object that holds the required physical information for cross section
        calculations.
    e_kin : float
        <eV>
        Kinetic energy of the impacting electron.

    Returns
    -------
    out : numpy.ndarray
        <m^2>
        The cross sections for each individual charge state, where the array-index corresponds
        to the charge state, i.e. out[q] ~ cross section of q+ ion.

    References
    ----------
    .. [1] "An empirical formula for the electron-impact ionization cross-section",
           W. Lotz,
           Zeitschrift Für Physik, 206(2), 205–211 (1967),
           https://doi.org/10.1007/BF01325928

    See Also
    --------
    ebisim.xs.eixs_mat : Similar method with different output format.

    """
    GRYS_TEMP = Template("grys_fact = (2+$i)/(2+t) * ((1+t) / (1+$i))**2 * ((($i+t) * (2+t) * (1+$i)**2) / (t * (2+t) * (1+$i)**2 + $i * (2+$i)))**1.5")
    XS_ABC_TEMP = Template("xs[$cs] += grys_fact(t, i) * $a * $n * math.log(e_kin/$e) / (e_kin * $e) * (1 - $b*math.exp(-$c * (e_kin/e -1)))")
    XS_A_TEMP = Template("xs[$cs] += grys_fact(t, i) * $a * $n * math.log(e_kin/$e) / (e_kin * $e)")

    shells = element.e_bind.shape[1]
    avail_factors = element.ei_lotz_a.shape[0]
    locs = []
    locs.append(f"xs = np.zeros({element.z + 1})")
    locs.append("t = e_kin / M_E_EV")
    for cs in range(element.z):
        # locs.append("xs = 0")
        locs.append("")
        for shell in range(shells):
            e = element.e_bind[cs, shell]
            n = element.e_cfg[cs, shell]
            if n > 0:
                i = e / ebisim.M_E_EV
                locs.append(f"e = {e:.8e}")
                locs.append(f"i = {i:.8e}")
                locs.append(f"if e_kin > e:")
                # locs.append("    " + GRYS_TEMP.substitute(i="i"))
                if cs < avail_factors:
                    a = element.ei_lotz_a[cs, shell]
                    b = element.ei_lotz_b[cs, shell]
                    c = element.ei_lotz_c[cs, shell]
                    if b == 0.0:
                        locs.append("    " + XS_A_TEMP.substitute(a=f"{a:.2e}", e="e", n=n, cs=cs))
                    else:
                        locs.append("    " + XS_ABC_TEMP.substitute(a=f"{a:.2e}", b=b, c=c, e="e", n=n, cs=cs))
                else:
                    locs.append("    " + XS_A_TEMP.substitute(a="4.5e-18", e="e", n=n, cs=cs))
        # locs.append(f"xs_vec[{cs}] = xs\n")
    locs.append("return xs")
    locs = ["    " + l for l in locs]
    code = "def eixs_unrolled(e_kin):\n" + "\n".join(locs)
    return code

def eixs_code2(element):
    """
    Electron ionisation cross section according to a simplified version of the models given in [1]_.

    Parameters
    ----------
    element : ebisim.Element
        An ebisim.Element object that holds the required physical information for cross section
        calculations.
    e_kin : float
        <eV>
        Kinetic energy of the impacting electron.

    Returns
    -------
    out : numpy.ndarray
        <m^2>
        The cross sections for each individual charge state, where the array-index corresponds
        to the charge state, i.e. out[q] ~ cross section of q+ ion.

    References
    ----------
    .. [1] "An empirical formula for the electron-impact ionization cross-section",
           W. Lotz,
           Zeitschrift Für Physik, 206(2), 205–211 (1967),
           https://doi.org/10.1007/BF01325928

    See Also
    --------
    ebisim.xs.eixs_mat : Similar method with different output format.

    """
    XS_A_TEMP =   Template("xs[$cs] += max(0, grys_fact(t, $ik) * $scale * (log_e_kin - $log_ek) / e_kin)")
    XS_BC_TEMP = Template(" * (1 - $bk * math.exp(-$ck * (e_kin / $ek - 1)))")
    def subshell_term(cs, ak, bk, ck, nk, ek, ik):
        scale =  ak * nk / ek
        log_ek = math.log(ek)
        t = XS_A_TEMP.substitute(
            cs=cs,
            scale=f"{scale:.16e}",
            log_ek=f"{log_ek:.16e}",
            ik=f"{ik:.16e}"
        )
        if bk > 0:
            t = t[:-1] + XS_BC_TEMP.substitute(
                bk=f"{bk:.2f}",
                ck=f"{ck:.2f}",
                ek=f"{ek:.16e}"
            ) + ")"
        return t

    shells = element.e_bind.shape[1]
    avail_factors = element.ei_lotz_a.shape[0]
    locs = []
    locs.append("if xs is None:")
    locs.append(f"    xs = np.zeros({element.z + 1})")
    locs.append("t = e_kin / M_E_EV")
    locs.append("log_e_kin = math.log(e_kin)")
    for cs in range(element.z):
        # locs.append("xs = 0")
        locs.append("")
        for shell in range(shells):
            e = element.e_bind[cs, shell]
            n = element.e_cfg[cs, shell]
            if n > 0:
                i = e / ebisim.M_E_EV
                # locs.append(f"e = {e:.8e}")
                # locs.append(f"i = {i:.8e}")
                # locs.append(f"if t > i:")
                if cs < avail_factors:
                    a = element.ei_lotz_a[cs, shell]
                    b = element.ei_lotz_b[cs, shell]
                    c = element.ei_lotz_c[cs, shell]
                else:
                    a = 4.5e-18
                    b, c = 0, 0
                locs.append(textwrap.indent(subshell_term(cs, a, b, c, n, e, i), ""))
    locs.append("return xs")
    locs = textwrap.indent("\n".join(locs), "    ")
    code = "def eixs_unrolled2(e_kin, xs=None):\n" + locs
    return code

def eixs_numpy(element, e_kin):
    g = grys_fact(e_kin/M_E_EV, element.e_bind/M_E_EV)
    temp = g * element.ei_lotz_a * element.e_cfg \
           * np.log(e_kin/element.e_bind)/(e_kin * element.e_bind) \
           * (1 - element.ei_lotz_b * np.exp(-element.ei_lotz_c * (e_kin/element.e_bind-1)))
    temp = np.maximum(temp,0)
    xs = np.zeros(element.z + 1)
    xs[:-1] =  np.nansum(temp, axis=1)
    return xs

@numba.vectorize([nfloat64(nfloat64, nfloat64)], nopython=True)
def grys_fact(t, i):
    return (2+i)/(2+t) * ((1+t) / (1+i))**2 * (((i+t) * (2+t) * (1+i)**2) / (t * (2+t) * (1+i)**2 + i * (2+i)))**1.5

@numba.vectorize(
    [nfloat64(nfloat64, nfloat64, nint64, nfloat64, nfloat64, nfloat64)],
    # target="parallel",
    nopython=True
)
def eixs_low(e_kin, e_bind, e_cfg, ei_lotz_a, ei_lotz_b, ei_lotz_c):
    if (e_kin > e_bind) & (e_cfg > 0):
        g = grys_fact(e_kin/M_E_EV, e_bind/M_E_EV)
        t = g * ei_lotz_a * e_cfg * np.log(e_kin/e_bind)/(e_kin * e_bind)
        if ei_lotz_b > 0:
            t *= (1 - ei_lotz_b * np.exp(-ei_lotz_c * (e_kin/e_bind-1)))
    else:
        t = 0
    return t

@numba.njit()
def eixs_low_call(element, e_kin):
    temp = eixs_low(
        e_kin,
        element.e_bind,
        element.e_cfg,
        element.ei_lotz_a,
        element.ei_lotz_b,
        element.ei_lotz_c
    )
    xs = np.zeros(element.z + 1)
    xs[:-1] = np.sum(temp, axis=1)
    return xs


def benchmark(z, e=300, n=10000):
    global eixs_unrolled
    element = ebisim.get_element(z)
    print(element)

    code = eixs_code2(element)
    # print(code)
    exec(code, globals())


    eixs_unrolled = numba.njit()(eixs_unrolled2)

    @numba.njit()
    def unroll_multicall(e_kin, n=1):
        for _ in range(n):
            xs = eixs_unrolled(e_kin)
        return xs

    @numba.njit()
    def loop_multicall(element, e_kin, n=1):
        for _ in range(n):
            xs = ebisim.eixs_vec(element, e_kin)
        return xs

    @numba.njit()
    def loop_multicall_prealloc(element, e_kin, n=1):
        xs = np.zeros(element.z + 1)
        for _ in range(n):
            ebisim.eixs_vec(element, e_kin, xs)
        return xs

    @numba.njit()
    def low_multicall(element, e_kin, n=1):
        for _ in range(n):
            xs = eixs_low_call(element, e_kin)
        return xs

    passed = np.allclose(unroll_multicall(e), loop_multicall(element, e), atol=0, rtol=1e-4)
    print(f"Test_unroll: {passed}")

    # passed = np.allclose(eixs_numpy(element, e), loop_multicall(element, e), atol=0)
    # print(f"Test_numpy: {passed}")

    # passed = np.allclose(low_multicall(element, e), loop_multicall(element, e), atol=0)
    # print(f"Test_low_vec: {passed}")

    t_loop = timeit.timeit(lambda: loop_multicall(element, e, n), number=1)
    # t_loop_pre = timeit.timeit(lambda: loop_multicall_prealloc(element, e, n), number=1)
    t_unrolled = timeit.timeit(lambda: unroll_multicall(e, n), number=1)
    # t_numpy = timeit.timeit(lambda: eixs_numpy(element, e), number=n)
    # t_low = timeit.timeit(lambda: low_multicall(element, e, n), number=1)
    print(f"T_loop: {t_loop}")
    # print(f"T_loop_pre: {t_loop_pre}")
    print(f"T_unrolled: {t_unrolled}")
    # print(f"T_numpy: {t_numpy}")
    # print(f"T_low: {t_low}")

# benchmark(4)
benchmark(10)
benchmark(30)
benchmark(60)
# benchmark(100)

# print(eixs_code_2(ebisim.get_element(4)))
