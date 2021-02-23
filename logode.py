import numpy as np
from scipy.integrate import solve_ivp

def f(logt, logy):
    y = np.exp(logy)
    t = np.exp(logt)
    return 3

sol = solve_ivp(f, np.log((1e-3, 5)), np.atleast_1d(np.log(1e-9)))

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.plot(np.exp(sol.t), np.exp(sol.y[0]),"x")
xs = np.logspace(-3, 1.5,1000)
ax.loglog(xs, xs**3)
ax.set_xlim()
ax.set_ylim()
plt.show()