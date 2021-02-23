import ebisim as eb
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
# os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
plt.rcParams['figure.figsize'] = (8,6)
plt.rcParams['figure.dpi'] = 100
# eb.simulation._advanced.compile_adv_model()

print(os.environ.get("NUMBA_DISABLE_INTEL_SVML", "Not set"))


k = eb.Target.get_ions("K", 1e-8 * 1.5e15, kT_per_q=80)
ne = eb.Target.get_gas("Ne", .1e-11, 5e-3)

# k = eb.Target.get_ions("K", .1*1e-8 * 1.5e15, kT_per_q=80)
# ne = eb.Target.get_gas("Ne", 100*.1e-11, 1e-4)


# k = eb.Target.get_gas("Ar", .5e-10, 1e-4)
dev = eb.Device.get(1, 8000, 1e-4, 0.8, 1300, 2, 0.005, n_grid=500,)
print(ne.n)
print(k.n)
# res = eb.advanced_simulation(dev, k,60, rates=True)

TMAX = 1
# T0 = 0.0001
kr, nr = eb.advanced_simulation(
    dev, [k, ne],TMAX,
    options=eb.ModelOptions(RADIAL_DYNAMICS=True),
    solver_kwargs={"dense_output":True, "max_step":.001},
    rates=True
)
# nr= eb.advanced_simulation(
#     dev, [ne, ],TMAX,
#     options=eb.ModelOptions(RADIAL_DYNAMICS=True),
#     solver_kwargs={"method":"Radau","dense_output":True, "max_step":.001, "min_step":1e-8},
#     rates=True,
# )
