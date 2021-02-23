import numba
from numba.core.types import literal

t1 = literal("on")
t2 = literal("off")

@numba.generated_jit()
def sim(x, switch):
    if switch.key == t1.key:
        def _sim(x, switch):
            return x
    elif switch.key == t2.key:
        def _sim(x, switch):
            return 2*x
    return _sim


@numba.generated_jit()
def sim2(x, switch):
    def _sim(x, switch):
        if switch.key == t1.key:
            return x
        elif switch.key == t2.key:
            return 2*x
    return _sim


print(sim(1, t1))
print(sim2(1, t1))
