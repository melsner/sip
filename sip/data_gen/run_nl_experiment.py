import os
import csv
import re
import itertools
import numpy as np
import argparse
import glob

from sip.data_gen.gen_isl import *
from sip.data_gen.isl_sampling_utilities import *
from sip.data_gen.utils import write_tsv

import transformers
from config_evaluator import Lazy
from sip.data_loading import *
from sip.embed_finetune import *
from sip.task_finetune import *

from sip.data_gen.run_simple_experiment import get_model_loader, ExperimentLogger

def parse_experiment_arguments(name):
    parser = argparse.ArgumentParser(prog=name)
    parser.add_argument("--model", choices=["local_isl", "local_tsl", "t5", "SIP"])
    parser.add_argument("--fst_format", default=None, choices=["isl_canon", "isl_markov", "tsl_canon", "tsl_markov"])
    parser.add_argument("--language", type=str)
    return parser.parse_args()

def run_experiment(scoreWriter, scoreFH, trainFile, testFile, predFile, load_model_function):        
    sample, num_train = re.search("(\d+)-(\d+).train", trainFile).groups()
    num_train = int(num_train)
    sample = int(sample)

    logger = ExperimentLogger(scoreWriter, num_train=num_train, sample=sample, fh=scoreFH)

    #copied from sip_isl_canon.jsonnet
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/byt5-small")
    train_loader = prepare_task_dataset(batch_size=32,
                                        path=trainFile,
                                        tokenizer=tokenizer)
    val_loader = prepare_task_dataset(batch_size=32,
                                      path=testFile,
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

if __name__ == "__main__":
    args = parse_experiment_arguments("nl")
    language = args.language

    data = f"data/eval/belth/{language}"
    
    model = args.model
    fst_format = args.fst_format

    run_name = f"nl_{language}_{model}_{fst_format}"
    os.makedirs(f"{data}/{run_name}", exist_ok=True)
    with open(f"{data}/{run_name}/scores.tsv", "w") as sfh:
        fields = ["num_train", "sample", "step", "acc", "edit_dist", "per",
                  "acc_avg_1", "edit_dist_avg_1", "per_avg_1", "tpr", "tnr", "fpr", "inform"]
        scoreWriter = csv.DictWriter(sfh, fieldnames=fields, dialect="excel-tab")
        scoreWriter.writeheader()

        load_model = get_model_loader(model, fst_format=fst_format)

        for fi in glob.glob(data + "/*.train"):
            print("Running on", fi)
            trainBase = os.path.basename(fi)
            predF = f"{data}/{run_name}/{trainBase}.pred"
            run_experiment(scoreWriter, sfh, fi, fi.replace(".train", ".test"), predF,
                           load_model_function=load_model)
