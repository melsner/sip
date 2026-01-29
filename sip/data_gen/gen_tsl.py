import pynini, pywrapfst

from sip.data_gen.gen_isl import *

def make_2tsl_transducer(factors, tier:str="", alphabet:str="", minimize:bool=True):
    projected_machine = make_2isl_transducer(factors, alphabet=tier, minimize=False)
    # print("original machine")
    # print_fst(projected_machine)

    projected_machine = replace_star_transitions(projected_machine)
    projected_machine = replace_star_state(projected_machine)
    # print("after mins")
    # print_fst(projected_machine)
    
    rstate = {}
    for kk, vv in projected_machine.state_names.items():
        rstate[vv] = kk
    projected_machine.state_names = rstate
    # print("new names:")
    # print(projected_machine.state_names)

    # print("before adding anything:")
    # print_fst(projected_machine)
    # print("-------")
    
    non_projected_chars = set(alphabet).difference(tier)
    # print("not projected:", non_projected_chars)
    isyms = pynini.SymbolTable()
    old_isyms = projected_machine.fst.input_symbols()
    for ii in range(old_isyms.num_symbols()):
        key = old_isyms.get_nth_key(ii)
        val = old_isyms.find(key)
        isyms.add_symbol(val)
    for ch in non_projected_chars:
        isyms.add_symbol(ch)

    osyms = pynini.SymbolTable()
    old_osyms = projected_machine.fst.output_symbols()
    for ii in range(old_osyms.num_symbols()):
        key = old_osyms.get_nth_key(ii)
        val = old_osyms.find(key)
        osyms.add_symbol(val)
    for ch in non_projected_chars:
        osyms.add_symbol(ch)

    for ch in non_projected_chars:
        for state in projected_machine.fst.states():
            if len(list(projected_machine.fst.arcs(state))) > 0:
                projected_machine.fst.add_arc(state,
                                              pynini.Arc(isyms.find(ch),
                                                         osyms.find(ch),
                                                         0,
                                                         state))
    projected_machine.fst.set_input_symbols(isyms)
    projected_machine.fst.set_output_symbols(osyms)

    # print("after adding stuff")
    # print_fst(projected_machine)
    # print("-------")

    if minimize:
        copied_fst = projected_machine.copy()
        projected_machine.fst = projected_machine.fst.minimize(allow_nondet=False)
        # Check that minimization didn't introduce any transitions that have epsilon on the input tape
        minimized_ok = True
        for state in projected_machine.fst.states():
            for arc in projected_machine.fst.arcs(state):
                if isyms.find(arc.ilabel) == "<e>":
                    minimized_ok = False
                    break
        if not minimized_ok:
            # print("Failed to minimize")
            projected_machine.fst = copied_fst
        else:
            projected_machine.state_names = None

    projected_machine.tier = tier
    return projected_machine

if __name__ == "__main__":
    print("Simple progressive pattern:")
    print()
    fst = make_2isl_transducer(
        [
            (("a", "b"), ("a", "c")),
        ],
        "abc",
        minimize=False)
    print_fst(fst)
    print("State names:", fst.state_names)
    test_fst(fst, "ababc", "acacc")

    print("Simple progressive harmony:")
    print()
    fst = make_2tsl_transducer(
        [
            (("a", "b"), ("a", "c")),
        ],
        tier="abc",
        alphabet="abcde",
        minimize=False)
    print_fst(fst)
    print("State names:", fst.state_names)
    test_fst(fst, "ababc", "acacc")
    test_fst(fst, "aeeebdadbc", "aeeecdadcc")
    print()

    print("What if d blocks harmony:")
    print()
    fst = make_2tsl_transducer(
        [
            (("a", "b"), ("a", "c")),
        ],
        tier="abcd",
        alphabet="abcde",
        minimize=False)
    print_fst(fst)
    print("State names:", fst.state_names)
    test_fst(fst, "ababc", "acacc")
    test_fst(fst, "aeeebdadbc", "aeeecdadbc")
    print()

    print("Simple progressive harmony:")
    print()
    fst = make_2tsl_transducer(
        [
            (("a", "b"), ("a", "c")),
        ],
        tier="abc",
        alphabet="abcde",
        minimize=True)
    print_fst(fst)
    print("State names:", fst.state_names)
    test_fst(fst, "ababc", "acacc")
    test_fst(fst, "aeeebdadbc", "aeeecdadcc")
    print()

    print("What if d blocks harmony:")
    print()
    fst = make_2tsl_transducer(
        [
            (("a", "b"), ("a", "c")),
        ],
        tier="abcd",
        alphabet="abcde",
        minimize=True)
    print_fst(fst)
    print("State names:", fst.state_names)
    test_fst(fst, "ababc", "acacc")
    test_fst(fst, "aeeebdadbc", "aeeecdadbc")
    print()
