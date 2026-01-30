import sys
import os
import glob
import re
import pickle
from collections import *
import pandas as pd

def gatherErrors(datadir):
    errors = defaultdict(lambda: defaultdict(list))
    with open(f"{datadir}/data.pkl", "rb") as ifh:
        while True:
            try:
                obj = pickle.load(ifh)
                for (inp, outp, pred) in listPredictions(obj):
                    inp = inp.rstrip(u"\x13")
                    if pred != outp:
                        if inp == outp:
                            eType = "copy"
                        else:
                            eType = "alter"
                        dSize = len(obj["train"])
                        errors[dSize][eType].append((inp, outp, pred))

            except EOFError:
                break

    return errors

def listPredictions(obj):
    test = obj["test"]
    pred = obj["predict"]
    assert(pred[0] == "input\toutput\n")
    pred = pred[1:]
    pred = [xx.strip().split("\t") for xx in pred]
    #print(len(test), len(pred))
    assert(len(test) == len(pred))

    pred = dict(pred)

    # for ti in test:
    #     print(ti)

    # for kk, vv in pred.items():
    #     print(kk, "---", vv)
    
    joined = [(ti.rstrip(u"\x13"), vi, pred.get(vi, "ERROR")) for ti, vi in test]
    
    # for row in joined:
    #     print(row)
    # assert(0)
    
    return joined
    
if __name__ == "__main__":
    datadir = sys.argv[1]

    errors = gatherErrors(datadir)

    for size in sorted(errors.keys()):
        for eType in errors[size]:
            es = errors[size][eType]
            nUnchanged = len([(inp, outp, pred) for (inp, outp, pred) in es if pred == inp])
            print(f"errors of {eType} at {size} ({len(es)} of which {nUnchanged} don't change)")
            for ei in es[:20]:
                print("\t", ei)
            print()
