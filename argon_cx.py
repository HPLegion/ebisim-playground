import ebisim as eb
import matplotlib.pyplot as plt
import numpy as np

ar = eb.get_element(18)
print(ar)

j = 100.
e_kin = 12000.
t_max = 10
adv_param = dict(
    Vtrap_ax = 1000,
    Vtrap_ra = 1000,
    bg_N0 = 1e-7 / (0.025 * eb.Q_E) * 1000,#1e-9mbar *x
    bg_IP = ar.ip
)

res = eb.advanced_simulation(ar, j, e_kin, t_max, adv_param=adv_param)

res.plot_charge_states(relative=False)
res.plot_temperature()
res.plot_rate("R_ei")
res.plot_rate("R_cx")
# res.plot_rate("R_ei")
plt.show()