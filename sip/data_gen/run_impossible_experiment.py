import os
import csv
import re
import itertools
import numpy as np

from sip.data_gen.gen_isl import *
from sip.data_gen.isl_sampling_utilities import *
from sip.data_gen.utils import write_tsv

import transformers
from config_evaluator import Lazy
from sip.data_loading import *
from sip.embed_finetune import *
from sip.task_finetune import *

from sip.data_gen.run_simple_experiment import *

if __name__ == "__main__":
    args = parse_experiment_arguments("impossible")

    spanish = np.loadtxt("https://raw.githubusercontent.com/unimorph/spa/refs/heads/master/spa",
                        dtype=str, delimiter="\t")
    spanish_words = list([xx for xx in spanish[:, 1] if xx.isalpha()])
    spanish_chars = set()
    for vi in spanish_words:
        spanish_chars.update(vi)
    spanish_alphabet = "".join(sorted(list(spanish_chars)))
    print(spanish_alphabet)

    #ABCDEFGHIJKLMNOPRSTUVYZabcdefghijklmnopqrstuvwxyzÉáâçéíîñóöúü
    vowels = "aeiouáéíóú"
    
    #per email:
    #aCo
    #a..[V-a]..o
    #^a...o$
    #a...o..
    #[-a]*o
    
    def categorize(word, char, trigger):
        patterns = [
            f"{trigger}..?{char}",
            f"^{trigger}.*{char}$",
            f"{trigger}[^{trigger}][^{trigger}][^{trigger}]+{char}",
            f"{char}.*{trigger}",
            f"{char}"
        ]
        for pattern in patterns:
            if re.search(re.compile(pattern), word):
                return pattern
        return None

    catWords = defaultdict(list)
    for word in spanish_words:
        catWords[categorize(word, "o", "a")].append(word)
        #catWords[categorize(word, "e", "a")].append(word)
        #catWords[categorize(word, "i", "a")].append(word)
        #catWords[categorize(word, "e", "i")].append(word)

    for cat, words in catWords.items():
        print(cat, ":", words[:10])
        print()

    def run_fn_s1(ss): #first-last
        return re.sub("^a(.*)o$", "a\\1O", ss)

    def run_fn_s2(ss): #strictly non-local
        sOld = ss
        sNew = re.sub(f"a([^a][^a][^a]+)o", "a\\1O", ss)
        while sNew != sOld:
            sOld = sNew
            sNew = re.sub(f"a([^a][^a][^a]+)o", "a\\1O", ss)            
        return sNew

    def run_fn_s3(ss): #progressive harmony
        sOld = ss
        sNew = re.sub("a(.*)o", "a\\1O", ss)
        while sNew != sOld:
            sOld = sNew
            sNew = re.sub("a(.*)o", "a\\1O", sOld)
        return sNew

    def run_fn_s4(ss): #strictly 2-local
        return re.sub(f"a(..?)o", "a\\1O", ss)

    def run_fn_s5(ss): #(one-sided) majority
        na = ss.count("a")
        no = ss.count("o")
        if na >= no:
            return ss.replace("o", "O")
        else:
            return ss
    
    process = args.process
    assert(0 < process <= 5)
    fns = [run_fn_s1, run_fn_s2, run_fn_s3, run_fn_s4, run_fn_s5]
    fn = fns[process - 1]
    print("Running for process", process - 1)

    # train, test = gen_balanced_problem(catWords, fn, 4, 2)
    # for pair in train:
    #     print(pair)
    # print()
    # for pair in test:
    #     print(pair)
    # print("-----")

    # assert(0)
    
    model = args.model
    fst_format = args.fst_format

    run_name = f"impossible_{process}_{model}_{fst_format}"
    ciph = None
    if args.cipher_type is not None:
        run_name += f"_ciph_{args.cipher_type}_{args.cipher_key}"
        ciph = Cipher(args.cipher_key, args.cipher_type, spanish_alphabet)

    os.makedirs(f"data/eval/{run_name}", exist_ok=True)
    with open(f"data/eval/{run_name}/scores.tsv", "w") as sfh:
        fields = ["num_train", "sample", "step", "acc", "edit_dist", "per",
                  "acc_avg_1", "edit_dist_avg_1", "per_avg_1", "tpr", "tnr", "fpr", "inform"]
        scoreWriter = csv.DictWriter(sfh, fieldnames=fields, dialect="excel-tab")
        scoreWriter.writeheader()

    load_model = get_model_loader(model, fst_format=fst_format)

    for size in range(args.train_min, args.train_max, args.train_incr):
        run_experiment(catWords, fn, run_name, size, args.n_test, n_trials=args.n_samples,
                       load_model_function=load_model, cipher=ciph)
