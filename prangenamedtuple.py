from numba import njit, prange
from collections import namedtuple

mytype = namedtuple("mytype", ("a", "b"))

@njit(parallel=True, nogil=True)
def outer(mydata):
    for k in prange(3):
        inner(k, mydata)

@njit(nogil=True)
def inner(k, mydata):
    print(k, mydata.a)
    print(k, mydata.b)

mydata = mytype(a="a", b="b")

# inner(100, mydata) ## -> Works as expected

outer(mydata) ## -> does not work