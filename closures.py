#  NUMBA_DEBUG_PRINT_AFTER='nopython_type_inference' python <whatever you called the file>
import numba

def sim_generator(effect1_switch=True, effect2_model="simple"):

    def sim(*args):
        # Some default stuff
        TOKEN1 = 10
        if effect1_switch:
            TOKEN2 = 20
        if effect2_model == "simple":
            TOKEN3 = 30
        else:
            TOKEN4 = 40
        return 50

    sim = numba.njit(sim)
    return sim

sim_generator(effect1_switch=False, effect2_model="simple")(1)

# def sim_generator(effect1_switch=True, effect2_model="simple")

#     def sim(*args):
#         # Some default stuff
#         if effect1_switch:
#             # Include effect1 code here
#         if effect2_model == "simple":
#             # simple model
#         else:
#             # advanced model
#         return sim_results

#     sim = numba.njit(sim)
#     return sim


# @generated_jit():
# def sim(*args, effect1_switch="on", effect2_model="simple"):
#     def _sim(*args)

# ```
# -----------------------nopython: nopython_type_inference------------------------
# label 0:
#     args = arg(0, name=args)                 ['args']
#     del args                                 []
#     $const0.1 = const(int, 10)               ['$const0.1']
#     TOKEN1 = $const0.1                       ['$const0.1', 'TOKEN1']
#     del TOKEN1                               []
#     del $const0.1                            []
#     $0.2 = freevar(effect1_switch: False)    ['$0.2']
#     branch $0.2, 8, 12                       ['$0.2']
# label 8:
#     del $0.2                                 []
#     $const8.1 = const(int, 20)               ['$const8.1']
#     TOKEN2 = $const8.1                       ['$const8.1', 'TOKEN2']
#     del TOKEN2                               []
#     del $const8.1                            []
#     jump 12                                  []
# label 12:
#     del $0.2                                 []
#     $12.1 = freevar(effect2_model: simple)   ['$12.1']
#     del $12.1                                []
#     $const12.2 = const(str, simple)          ['$const12.2']
#     del $const12.2                           []
#     $12.3 = const(int, 1)                    ['$12.3']
#     del $12.3                                []
#     jump 20                                  []
# label 20:
#     $const20.1 = const(int, 30)              ['$const20.1']
#     TOKEN3 = $const20.1                      ['$const20.1', 'TOKEN3']
#     del TOKEN3                               []
#     del $const20.1                           []
#     jump 30                                  []
# label 30:
#     $const30.1 = const(int, 50)              ['$const30.1']
#     $30.2 = cast(value=$const30.1)           ['$30.2', '$const30.1']
#     del $const30.1                           []
#     return $30.2                             ['$30.2']
# ```