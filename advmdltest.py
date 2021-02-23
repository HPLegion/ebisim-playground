import ebisim.simulation._modelbuilder as mb
import numba
import numpy as np
dev = mb.Device(
    100., 5000., 1., 1e-4, 12., 1500., 2000., 2., 5e03
)
he = mb.get_ion_target(2, 1e10)
li = mb.get_ion_target(3, 1e10)
targets = numba.typed.List((he,li))
bg_gas = numba.typed.List((mb.BgGas(1.0, 1e-10), mb.BgGas(3.0, 3e-10)))
am = mb.AdvancedModel(dev, targets, bg_gas)
am.rhs(0.0, np.arange(1,15, dtype=np.float64))
print(Hallo)