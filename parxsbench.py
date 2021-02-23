import numba
import ebisim as es
import timeit
import matplotlib.pyplot as plt
import tqdm


@numba.njit()
def bc(element, n):
    for _ in range(n):
        for e in [1.e1, 1.e2, 1.e3, 1.e4, 1.e5]:
            es.eixs_vec(element, e)

Z = []
T = []


bc(es.get_element(1), 10)
bc(es.get_element(19), 10)

for z in tqdm.tqdm(range(1,100)):
    elem = es.get_element(z)
    # bc(elem, 1)
    t = timeit.timeit(lambda: bc(elem, 1000), number=1)
    Z.append(z)
    T.append(t)

# print(bc.inspect_types(pretty=True))
plt.subplots()
plt.plot(Z, T)
plt.show()
