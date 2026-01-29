import os
import numpy as np
import argparse

from sip.data_gen.gen_isl import SYMBOL_RDELIM

def parse_experiment_arguments(name):
    parser = argparse.ArgumentParser(prog=name)
    parser.add_argument("--language", choices=["english", "german", "polish", "finnish", "turkish-childes", "turkish-morpho", "latin"])
    parser.add_argument("--samples", default=10, type=int)
    parser.add_argument("--sizes", default="100,200,300,400")
    parser.add_argument("--output", default="data/eval/belth")
    parser.add_argument("--cipher", action="store_true")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_experiment_arguments("create_belth_local_dsets")

    langFiles = {
        "german" : "https://raw.githubusercontent.com/cbelth/PLP/refs/heads/main/data/german/ger.txt",
        "english" : "https://raw.githubusercontent.com/cbelth/PLP/refs/heads/main/data/english/eng.txt",
        "polish" : "https://raw.githubusercontent.com/cbelth/PLP/refs/heads/main/data/polish/pol.txt",
        "finnish" : "https://raw.githubusercontent.com/cbelth/Learning-Based-Tiers/refs/heads/main/data/finnish/finnish.txt",
        "latin" : "https://raw.githubusercontent.com/cbelth/Learning-Based-Tiers/refs/heads/main/data/latin/latin.txt",
        "turkish-childes" : "https://raw.githubusercontent.com/cbelth/Learning-Based-Tiers/refs/heads/main/data/turkish/childes.txt",
        "turkish-morpho" : "https://raw.githubusercontent.com/cbelth/Learning-Based-Tiers/refs/heads/main/data/turkish/morpho.txt",
    }
    dataset = langFiles[args.language]
    data = np.loadtxt(dataset, dtype=str, delimiter="\t", skiprows=1)

    if args.language == "finnish":
        #drop extra orthographic "word" col
        data = data[:, 1:]
        assert(data.shape[1] == 3)

    #Sept 30--- try without stripping syll bdry.
    #strip syllable boundary, since there is no way to exclude it from the charset, and add delim
    # data[:, 0] = [xx.replace(".", "") + chr(SYMBOL_RDELIM) for xx in data[:, 0].tolist()]
    # data[:, 1] = [xx.replace(".", "") for xx in data[:, 1].tolist()]
    data[:, 0] = [xx + chr(SYMBOL_RDELIM) for xx in data[:, 0].tolist()]
    data[:, 1] = [xx for xx in data[:, 1].tolist()]
    print(data[:5])
    freqs = data[:, -1].astype(float)
    probs = freqs / freqs.sum()
    idxs = np.arange(data.shape[0])

    sizes = sorted([int(xx) for xx in args.sizes.split(",")])

    for sample in range(args.samples):
        sizeCt = 0

        charset = set()
        for xx in data[:, 0]:
            charset.update(xx)
        for xx in data[:, 1]:
            charset.update(xx)

        if args.cipher:
            minC = np.min([ord(ci) for ci in charset])
            maxC = np.max([ord(ci) for ci in charset])
            ciph = np.random.choice(maxC - minC)
            def enc(string):
                delim = False
                if string.endswith(chr(SYMBOL_RDELIM)):
                    string = string[:-1]
                    delim = True
                ciphered = [
                    chr(
                        ((ord(ci) + ciph) % (maxC - minC)) + minC)
                    for ci in string]
                if delim:
                    ciphered.append(chr(SYMBOL_RDELIM))
                return "".join(ciphered)

            cData = np.copy(data)
            cData[:, 0] = [enc(xx) for xx in data[:, 0].tolist()]
            cData[:, 1] = [enc(xx) for xx in data[:, 1].tolist()]
            print("enciphered", data[:5, 0], "=>")
            print(cData[:5, 0])            
        else:
            cData = data        
        
        dset = set()

        while sizeCt < len(sizes):
            nextSize = sizes[sizeCt]

            ind = np.random.choice(idxs, p=probs, size=1)[0]
            chosen = cData[ind]
            #print(ind, chosen)
            dset.add(tuple(chosen[:2]))
            if len(dset) == nextSize:

                if args.cipher:
                    outDir = f"{args.output}/{args.language}-ciph/"
                else:
                    outDir = f"{args.output}/{args.language}/"

                os.makedirs(outDir, exist_ok=True)

                with open(f"{outDir}/{sample}-{nextSize}.train", "w") as tfh:
                    tfh.write("input\toutput\n")
                    varRate = 0
                    for pair in dset:
                        tfh.write("\t".join(pair) + "\n")
                        if pair[0].strip(chr(SYMBOL_RDELIM)) != pair[1]:
                            varRate += 1
                    print(f"{varRate} pairs out of {nextSize} altered: {varRate/nextSize}")

                with open(f"{outDir}/{sample}-{nextSize}.test", "w") as tfh:
                    tfh.write("input\toutput\n")
                    for pair in cData[:, :2]:
                        if tuple(pair) not in dset:
                            tfh.write("\t".join(pair) + "\n")
                        
                sizeCt += 1
