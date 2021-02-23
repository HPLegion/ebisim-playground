import ebisim as eb
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"

k = eb.Target.get_ions("K", 1e-8 * 1.5e15, kT_per_q=80)
ne = eb.Target.get_gas("Ne", .1e-11, 1e-4)
# k = eb.Target.get_gas("Ar", .5e-10, 1e-4)
dev = eb.Device.get(1, 8000, 1e-4, 0.8, 1300, 2, 0.005, n_grid=500,)
print(ne.n)
print(k.n)
# res = eb.advanced_simulation(dev, k,60, rates=True)

# print("done")
TMAX = 10
T0 = 0.0001
res = eb.advanced_simulation(dev, k, TMAX, options=eb.ModelOptions(RADIAL_DYNAMICS=True), solver_kwargs={"dense_output":True, "max_step":1})
kr, nr = eb.advanced_simulation(dev, [k, ne],TMAX, options=eb.ModelOptions(RADIAL_DYNAMICS=True), solver_kwargs={"dense_output":True, "max_step":1}, rates=False)

# # res.plot_radial_distribution_at_time(.0005)
# print()
# print(np.amax(np.diff(res.t)))
# # res.plot_radial_distribution_at_time(.05)
# res.plot_radial_distribution_at_time(.13)
# # plt.plot(dev.rad_grid, dev.rad_phi_ax_barr)

# res.plot_radial_distribution_at_time(.5)
# res.plot_temperature()
# res.plot(relative=False)
# res.plot_energy_density()
# plt.show()

# plt.plot(dev.rad_grid, dev.rad_phi_ax_barr)
# # res.plot_radial_distribution_at_time(5)

fig, axs = plt.subplots(2, 3, figsize=(18, 12))

res.plot_charge_states(ax=axs[0, 0], title=None)
kr.plot_charge_states(ax=axs[1, 0], title=None)
# axs[0, 0].set_title(res._param_title("Linear density"))

_xlim = (0, 0.0003)
_ylimphi = (-2000,200)
_ylim = (0, 8e15)
_kwargs = dict(title=None, ylimphi=_ylimphi, ylim=_ylim, xlim=_xlim, yscale="linear", xscale="linear")

rax, rax2 = res.plot_radial_distribution_at_time(T0, ax=axs[0, 1], **_kwargs)
rax3, rax4 = kr.plot_radial_distribution_at_time(T0, ax=axs[1, 1], **_kwargs)
# axs[0, 1].set_title(res._param_title("Radial distribution"))
axs[0, 1].set_title(res._param_title(""))

res.plot_temperature(ax=axs[0, 2], title=None, ylim=(10,10000))
kr.plot_temperature(ax=axs[1, 2], title=None, ylim=(10,10000))
# axs[0, 2].set_title(res._param_title("Temperature"))

l = axs[0, 0].axvline(T0)
l2 = axs[0, 2].axvline(T0)
l3 = axs[1, 0].axvline(T0)
l4 = axs[1, 2].axvline(T0)

for i, t in enumerate(np.geomspace(T0, TMAX, 600)):
    print(i)
    l.remove()
    l2.remove()
    l3.remove()
    l4.remove()
    l = axs[0, 0].axvline(t)
    l2 = axs[0, 2].axvline(t)
    l3 = axs[1, 0].axvline(t)
    l4 = axs[1, 2].axvline(t)

    rax.clear()
    rax2.clear()
    res.plot_radial_distribution_at_time(t, ax=rax, ax2=rax2, **_kwargs)

    rax3.clear()
    rax4.clear()
    kr.plot_radial_distribution_at_time(t, ax=rax3, ax2=rax4, **_kwargs)

    # axs[0, 1].set_title(res._param_title("Radial distribution"))
    axs[0, 1].set_title(res._param_title(""))
    plt.savefig(f"./img/K3_{i:04}.png")
    # plt.close()
    # plt.show()


# for i, t in enumerate(np.geomspace(0.0001,TMAX, 100)):
#     print(i)
#     res.plot_radial_distribution_at_time(t, ylim=(1.e8, 1.e18), ylimphi=(-2000, 0))
#     plt.savefig(f"./img/E_{i:04}.png")
#     plt.close()

# res.plot_charge_states()
# res.plot_rate("R_ax")
# res.plot_rate("V_it")

# res.plot_temperature()
kr, nr = eb.advanced_simulation(dev, [k, ne],TMAX, options=eb.ModelOptions(RADIAL_DYNAMICS=True), solver_kwargs={"dense_output":True, "max_step":1}, rates=True)
kr.plot()
kr.plot_temperature()
nr.plot()
nr.plot_temperature()
plt.show()
