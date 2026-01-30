import sys
import os
import glob
import re
from collections import *
import pandas as pd
import scipy.stats
import numpy as np

from plot_amp import *

def bootstrapped_med_conf(stts, conf):
    stts = np.array(stts)[None, :]
    boot = scipy.stats.bootstrap(stts, statistic=np.median, confidence_level=conf, method="percentile")
    return boot
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="plot_amp")
    parser.add_argument("data_root", help="Location of log file subdirs")
    parser.add_argument(
        "--metric",
        choices=["acc", "edit_dist", "inform"],
        default="inform",
        help="Metric to plot",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        choices=EXPERIMENTS,
        required=True,
        help="Experiment to select tasks from",
    )
    args = parser.parse_args()
    
    resultsdir = args.data_root
    metric = args.metric
    taskid = ""
    group = None
    #sizes = None
    sizes = [2, 10, 20, 30]

    set_experiment_paths(args.experiment)
    df = loadScores(resultsdir)
    #df = loadScoresSynth(resultsdir, experiment_group=group)
    currTask = None
    currModel = None

    task = "task"
    if args.experiment in NL_NAMES:
        sizes = None #sample sizes from data for nl

    if sizes is None:
        sizes = df["num_train"].unique()
        print("sizes", sizes)

    for (task, model, size), block in df.groupby(by=[task, "model", "num_train"]):
        if size not in sizes:
            continue

        if currTask != task:
            print("\\\\")
            currTask = task
            print(f" {taskid}{currTask} & ", " & ".join(str(xx) for xx in sizes), "\\\\ \n \\hline \n")
        if currModel != model:
            print("\\\\")
            currModel = model
            print(currModel, end="")
            
        median = block[[metric]].median().tolist()[0]
        bscf = bootstrapped_med_conf(block[metric].tolist(), .95)
        bscf = bscf.confidence_interval
        mean = block[[metric]].mean().tolist()[0]
        sem = block[[metric]].sem().tolist()[0]
        #print(f" & {mean:.03f} $\\pm$ {sem:.03f}", end="")
        print(f" & {median:.02f} ({bscf[0]:.02f} $-$ {bscf[1]:.02f})", end="")
    print()
