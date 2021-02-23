import ebisim as eb
import numpy as np
import matplotlib.pyplot as plt

target = eb.simulation.get_gas_target("Ar", 1e-10/(0.025*eb.Q_E))
device = eb.simulation.Device(3000, 8000, 1.2, 1e-4, 500, 1500, 2, 5e-3)

sol = eb.advanced_simulation(device, target, 1, solver_kwargs=dict(
    method="Radau",
    atol=1e-6
))
print(sol.res.message)
sol.plot_charge_states(xlim=(None,None), ylim=(None, None), yscale="log")
sol.plot_rate("R_cx")
plt.show()
