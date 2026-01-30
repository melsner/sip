import sys
import os
import glob
import re
from collections import *
import pandas as pd

def gatherErrors(datadir, resultsdir):
    errors = defaultdict(lambda: defaultdict(list))
    for fi in glob.glob(f"{datadir}/*.test"):
        for (inp, outp, pred) in listPredictions(datadir, resultsdir, fi):
            inp = inp.rstrip(u"\x13")
            if pred != outp:
                if inp == outp:
                    eType = "copy"
                else:
                    eType = "alter"
                dSize = re.match(r"(\d)-(\d+).test", os.path.basename(fi)).group(2)
                errors[dSize][eType].append((inp, outp, pred))

    return errors

def listPredictions(datadir, resultsdir, fi):
    name = os.path.basename(fi).replace(".test", ".train")
    rFile = f"{resultsdir}/{name}.pred"
    actual = pd.read_csv(fi, sep="\t")
    print("results file at", rFile)
    if not os.path.exists(rFile):
        print("---NOT FOUND")
        return []
    pred = pd.read_csv(rFile, sep="\t")
    # print(actual.head())
    # print()
    # print(pred.head())
    pred = pred.rename(columns={"output" : "prediction"})
    joint = actual.set_index("output").join(pred.set_index("input"))
    #print(joint.head())
    joint = joint.reset_index()
    #print(joint.head())
    joint = joint.loc[:, ["input", "output", "prediction"]]
    #print(joint.head())
    return joint.to_numpy()
    
if __name__ == "__main__":
    datadir = sys.argv[1]
    resultsdir = sys.argv[2]

    errors = gatherErrors(datadir, resultsdir)

    for size in sorted(errors.keys()):
        for eType in errors[size]:
            es = errors[size][eType]
            nUnchanged = len([(inp, outp, pred) for (inp, outp, pred) in es if pred == inp])
            print(f"errors of {eType} at {size} ({len(es)} of which {nUnchanged} don't change)")
            for ei in es[:20]:
                print("\t", ei)
            print()
