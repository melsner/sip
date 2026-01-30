import os
import csv
import itertools
import numpy as np
import argparse
import pickle

from sip.data_gen.gen_isl import *
from sip.data_gen.isl_sampling_utilities import *
from sip.data_gen.utils import write_tsv

import transformers
from config_evaluator import Lazy
from sip.data_loading import *
from sip.embed_finetune import *
from sip.task_finetune import *

class NotEnoughExamplesError(Exception):
    def __init__(self, nn, balance):
        self.nn = nn
        self.balance = balance

    def __str__(self):
        if self.balance:
            return f"Cannot find {nn//2} examples of each class."
        else:
            return f"Cannot find {nn} examples."

def parse_experiment_arguments(name):
    parser = argparse.ArgumentParser(prog=name)
    parser.add_argument("--model", choices=["local_isl", "local_tsl", "t5", "SIP"])
    parser.add_argument("--fst_format", default=None, choices=["isl_canon", "isl_markov", "tsl_canon", "tsl_markov"])
    parser.add_argument("--process", type=int)
    parser.add_argument("--train_min", default=2, type=int)
    parser.add_argument("--train_max", default=40, type=int)
    parser.add_argument("--train_incr", default=2, type=int)
    parser.add_argument("--n_test", default=8, type=int)
    parser.add_argument("--n_samples", default=16, type=int)
    parser.add_argument("--cipher_key", default=0, type=int)
    parser.add_argument("--cipher_type", default=None, choices=["monoalphabetic", "shift"])

    return parser.parse_args()

def gen_phonology_problem(vocabulary, replace_fn, balance=True, n_exes=20):
    v_order = vocabulary[:]
    np.random.shuffle(v_order)

    e_changed = []
    e_unchanged = []
    e_all = []
    for vi in v_order:
        ri = replace_fn(vi)
        e_all.append((vi, ri))
        if ri != vi:
            e_changed.append((vi, ri))
        else:
            e_unchanged.append((vi, ri))

        if balance:
            if len(e_changed) >= n_exes // 2 and len(e_unchanged) >= n_exes // 2:
                break
        else:
            if len(e_all) >= n_exes:
                break

    if balance:
        if len(e_changed) >= n_exes // 2 and len(e_unchanged) >= n_exes // 2:
            return e_changed[:n_exes // 2], e_unchanged[:n_exes // 2]
    else:
        if len(e_all) >= n_exes:
            return e_all[:n_exes]

    raise NotEnoughExamplesError(n_exes, balance)

def train_test_split(changed, unchanged, n_test):
    test = changed[:n_test // 2] + unchanged[:n_test // 2]
    train = changed[n_test//2:] + unchanged[n_test//2:]
    np.random.shuffle(train)
    np.random.shuffle(test)
    return train, test

class ExperimentLogger:
    def __init__(self, writer, num_train, sample, fh=None):
        self.writer = writer
        self.step = itertools.count()
        self.num_train = num_train
        self.sample = sample
        self.fh = fh

    def progress_bar(self, generator):
        for item in generator:
            yield item

    def log_metrics(self, name, fields):
        if "loss" in fields:
            return #don't need this
        fields["sample"] = self.sample
        fields["step"] = next(self.step)
        fields["num_train"] = self.num_train
        self.writer.writerow(fields)
        if self.fh:
            self.fh.flush()

def gen_balanced_problem(cat_words, process, num_train, num_test):
    train = []
    test = []
    for cat, words in cat_words.items():
        selected = np.random.choice(words, size=num_train)
        train += list(zip(selected, [process(xx) for xx in selected]))
        selected = np.random.choice(words, size=num_test)
        test += list(zip(selected, [process(xx) for xx in selected]))

    np.random.shuffle(train)
    np.random.shuffle(test)

    return train, test

def get_model_loader(mode, fst_format=None):
    if mode in ["local_isl", "local_tsl"]:
        assert(fst_format in ["isl_canon", "isl_markov", "tsl_canon", "tsl_markov"])
        if mode == "local_tsl":
            if fst_format == "tsl_markov":
                model = "models/w_fsts_pretrain_s4_32_tsl_moretsl_markov"
            else:
                model = "models/w_fsts_pretrain_s4_32_tsl_moretsl_canon"
        else:
            if fst_format == "isl_canon":
                model = "models/w_fsts_pretrain_s4_32_isl_canon"
            else:
                model = "models/w_fsts_pretrain_s4_32_isl_markov"

        tokenizer = transformers.AutoTokenizer.from_pretrained("google/byt5-small")

        def load_model():
            return load_struct_prefix_with_init(
                prefix_length=50,
                num_examples=32,
                random_selection=True,
                fst_tokenizer_path="unicode_char_tokenizer_ipa.json",
                tokenizer=tokenizer,
                model_str=model,
                fst_format=fst_format,
                map_location="cpu")
    elif mode == "t5":
        def load_model():
            load_model = "google/byt5-small"
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(load_model)
            return model
    elif mode == "SIP":
        def load_model():
            load_model = "namednil/sip-d4"
            model = SIPFinetuningModel.from_pretrained(load_model)
            return model
    else:
        assert(0), f"Bad model loader {mode}"

    return load_model

class Cipher:
    def __init__(self, key, mode, alphabet):
        self.key = key
        self.mode = mode
        self.alphabet = alphabet

        self.minC = np.min([ord(ci) for ci in self.alphabet])
        self.maxC = np.max([ord(ci) for ci in self.alphabet])

        self.mapping = {}
        if mode == "shift":
            for pi in self.alphabet:
                self.mapping[pi] = chr(
                    ((ord(pi) + self.key - minC) % (maxC - minC)) + minC)
        elif mode == "monoalphabetic":
            rgen = np.random.default_rng(seed=self.key)
            perm = list(self.alphabet)
            rgen.shuffle(perm)
            for pi, ci in zip(self.alphabet, perm):
                self.mapping[pi] = ci
        else:
            assert(0), "unknown cipher mode"

    def encipher(self, data):
        def enc(string):
            delim = False
            if string.endswith(chr(SYMBOL_RDELIM)):
                string = string[:-1]
                delim = True

            ciphered = [self.mapping[ci] for ci in string]

            if delim:
                ciphered.append(chr(SYMBOL_RDELIM))

            return "".join(ciphered)

        d0 = [enc(xx[0]) for xx in data]
        d1 = [enc(xx[1]) for xx in data]
        print("enciphered", data[:5], "=>")
        print(list(zip(d0, d1))[:5])
        assert(len(d0) == len(d1))
        return list(zip(d0, d1))

def run_experiment(words, process, run, num_train, num_test, n_trials, load_model_function, cipher=None):
    os.makedirs(f"data/eval/{run}", exist_ok=True)
    with open(f"data/eval/{run}/scores.tsv", "a") as sfh:
        fields = ["num_train", "sample", "step", "acc", "edit_dist", "per",
                  "acc_avg_1", "edit_dist_avg_1", "per_avg_1", "tpr", "tnr", "fpr", "inform"]
        scoreWriter = csv.DictWriter(sfh, fieldnames=fields, dialect="excel-tab")

        with open(f"data/eval/{run}/data.pkl", "ab") as dfh:
        
            for ii in range(n_trials):
                #changed, unchanged = gen_phonology_problem(words, process, balance=True, n_exes=num_train + num_test)
                #train, test = train_test_split(changed, unchanged, num_test)
                train, test = gen_balanced_problem(words, process, num_train, num_test)
                train = [(inp + chr(SYMBOL_RDELIM), outp) for (inp, outp) in train]
                test = [(inp + chr(SYMBOL_RDELIM), outp) for (inp, outp) in test]

                if cipher != None:
                    train = cipher.encipher(train)
                    test = cipher.encipher(test)
                
                trainFile = f"data/eval/{run}/i_{ii}_t_{num_train}_v_{num_test}_train.tsv"
                testFile = f"data/eval/{run}/i_{ii}_t_{num_train}_v_{num_test}_val.tsv"
                predFile =  f"data/eval/{run}/i_{ii}_t_{num_train}_v_{num_test}_pred.tsv"
                write_tsv(trainFile, train)
                write_tsv(testFile, test)

                logger = ExperimentLogger(scoreWriter, num_train=num_train, sample=ii)
            
                #copied from sip_isl_canon.jsonnet
                tokenizer = transformers.AutoTokenizer.from_pretrained("google/byt5-small")
                train_loader = prepare_task_dataset(batch_size=32,
                                                    path=f"data/eval/{run}/i_{ii}_t_{num_train}_v_{num_test}_train.tsv",
                                                    tokenizer=tokenizer)
                val_loader = prepare_task_dataset(batch_size=32,
                                                  path=f"data/eval/{run}/i_{ii}_t_{num_train}_v_{num_test}_val.tsv",
                                                  tokenizer=tokenizer)
                model = load_model_function()
                finetune_model(model=model,
                               tokenizer=tokenizer,
                               train_data_loader=train_loader,
                               validation_data_loader=val_loader,
                               optimizer=Lazy({"lr":5e-4}, torch.optim.Adam),
                               num_epochs=50,
                               optimizer_groups=[
                                   [".*prefix_embedding.*", {"lr": 1.0}],
                                   [".*", {"lr": 3e-4}],
                               ],
                               grad_scale=1.0,
                               logger=logger,
                               eval_predictions_file=predFile,
                               eval_only_last_epochs=True,
                               moving_avg_steps=1,
                               )

                #copy the dataset to an archive
                with open(predFile) as ipf:
                    predict = np.array(ipf.readlines())
                    
                data = {
                    "train" : train,
                    "test" : test,
                    "predict" : predict
                    }
                pickle.dump(data, dfh)

                #clean up the working files
                for fp in [trainFile, testFile, predFile]:
                    os.unlink(fp)
            
if __name__ == "__main__":
    #implement a simple assimilation pattern
    toy_vocab = ["inner", "inmer", "inter"]
    toy_alphabet = set()
    for vi in toy_vocab:
        toy_alphabet.update(vi)
    toy_alphabet = "".join(toy_alphabet)
    print("Alphabet:", toy_alphabet)

    fst = make_2isl_transducer(
        [
            (("n", "m"), ("m", "m")),
        ],
        toy_alphabet)

    print_fst(fst)
    test_fst(fst, "inmi", "immi")

    fst = postprocess_for_sampling(fst)
    def run_fst(ss):
        return apply_fst(fst, ss)

    for ss in toy_vocab:
        print(ss, run_fst(ss))

    german = np.loadtxt("https://raw.githubusercontent.com/unimorph/deu/refs/heads/master/deu",
                        dtype=str, delimiter="\t")
    german_words = list([xx for xx in german[:, 1] if xx.isalpha()])
    german_chars = set()
    for vi in german_words:
        german_chars.update(vi)
    german_alphabet = "".join(sorted(list(german_chars)))
    print(german_alphabet)

    devoice_d = make_2isl_transducer(
        [
            (("d", "</s>"), ("t",)),
        ],
        german_alphabet)

    devoice_d = postprocess_for_sampling(devoice_d)
    def run_devoice(ss):
        return apply_fst(devoice_d, ss)

    change, unchange = gen_phonology_problem(german_words, run_devoice)
    for pair in change:
        print(pair)
    print()
    for pair in unchange:
        print(pair)
    print("-----")

    train, test = train_test_split(change, unchange, 8)
    for pair in train:
        print(pair)
    print()
    for pair in test:
        print(pair)

    run_name = "devoice_d"
    with open(f"data/eval/{run_name}/scores.tsv", "w") as sfh:
        fields = ["num_train", "sample", "step", "acc", "edit_dist", "per",
                  "acc_avg_1", "edit_dist_avg_1", "per_avg_1"]
        scoreWriter = csv.DictWriter(sfh, fieldnames=fields, dialect="excel-tab")
        scoreWriter.writeheader()
        
    for size in range(10, 60, 10):
        run_experiment(german_words, run_devoice, run_name, size, 16)
