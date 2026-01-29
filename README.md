# SIP

This is the code for the AMP 2025 paper "Sub-regular inductive biases in a phonological transformer", based on the code for the ACL 2024 paper [SIP: Injecting a Structural Inductive Bias into a Seq2Seq Model by Simulation](https://arxiv.org/abs/2310.00796) by Lindemann et al. Our code extends theirs and is a fork of their github repository [namednil/sip](https://github.com/namednil/sip). More documentation is available there. We are very grateful to Matthias Lindemann and his coauthors for distributing their software and providing us with the technical assistance we needed to use it.

# Installation

Taken from the original SIP repository:

```
conda create -n sip python=3.10
conda activate sip
conda install -c conda-forge pynini # FST library
# install pytorch (we used v 2.2.1 but newer versions such as 2.6.0 should work fine as well)
#e.g. via
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Clone git repository
git clone https://github.com/namednil/sip

# Install remaining pip requirements
# (potentially uncomment neptune.ai dependency if desired)
cd sip
pip install -r requirements.txt
```

## Creating training data

To create 2ISL training instances: `python -u -m sip.data_gen.gen_isl_pretrain`
To create 2TSL training instances: `python -u -m sip.data_gen.gen_tsl_pretrain`

Those files can be edited to alter the number of items to generate and
the representation (markov or canonical). The paper uses the
"canonical" setting, in which the transducers are minimized and TSL
machines are encoded using self-loops for non-projected
characters. The "markov" encodings use non-minimal machines in which
each states is labeled with the previously-seen character.

To create natural language data: `python -u -m sip.data_gen.create_belth_local_dsets --language [german] --samples [10] --sizes [100,200,300,400]`

The datasets will be sampled automatically from the vocabulary files in [github.com/cbelth/PLP](github.com/cbelth/PLP) and [github.com/cbelth/Learning-Based-Tiers](github.com/cbelth/Learning-Based-Tiers).

## Pretraining SIP models

All pretraining is handled via jsonnet scripts in the `pretrain_non_hub` directory, e.g. `python -u config_evaluator.py configs/pretrain_non_hub/pretrain_SIP_isl_markov.jsonnet`

## Running experiments

The experiments in section A can be run using: `python -u -m sip.data_gen.run_harmony_experiment --process [1] --model [local_isl] --fst_format [isl_canon] --cipher_type [monoalphabetic] --cipher_key [2]`

* Process: 1 through 5. See code; the table in the paper is reordered from the numbers in the actual code file.
* Model and format: defined in `data_gen/run_simple_experiment.py`; one of "local_isl", "local_tsl", "t5", "SIP" and one of "isl_canon", "isl_markov", "tsl_canon", "tsl_markov".
* Cipher type: "monoalphabetic" or leave blank for no cipher. (There's also a "shift" mode we don't use in the paper.) Cipher key: arbitrary integer used to seed the RNG.
* You can also use `train_min`, `train_max`, `n_test`, `n_samples` to control how many trials are run and what set sizes are used. See `run_simple_experiment.py` for details.

The experiment will produce a tsv of results and a `.pkl` file which contains all the input and output instances for each run (one pickle per run).

## Plots and tables

Our plotting tool is `plot_amp.py`, which can be run as follows:

```
python plot_amp.py [data/eval] -e [condition] -m SIP-TSL -m SIP-ISL -m SIP-FST -m t5 --metric inform --multi [overlay] -t all -c remove -o [filename].svg
```

* *data/eval* is the directory in which experiment logs are located.
* *condition* is the experiment code (e.g. *turkish*, *harmony*, etc); see the help message for a full list.
* The *--multi* flag controls how information is divided into plots: either grouped by task, grouped by model, or fully faceted by task/model.
* The *-c* flag controls how ciphered runs are displayed: removed from the plot, aggregated with non-cipher runs and plotted as a single series, or faceted.
* The model, task and metric flags can be used to select different data series to show.

The `tabular_output.py` tool makes Latex tables showing the median and 95% confidence of the scores. It takes the same flags as the plotter.

You can print errors with `nl_errors.py [directory with the data] [model run directory]` and `synth_errors.py [model run directory]`. 

An example of how to read the output: The first block shows errors from the 100-item training set size for which the gold input and output differ ("alter"). There are 5026 errors of this type across all runs, and in 4723 of these, the proposed output is identical to the input ("don't change"), that is, the change is underapplied. The tuples shown are (input, gold output, prediction). For instance, the first tuple shows the model failing to devoice the final z of "gantz".

The next block shows errors where the gold input and output are identical ("copy"). There are 2072 of these and in each case, the proposed output is not identical to the input (since otherwise it would not be an error!). These show various kinds of more or less unprincipled alterations, eg. Schreck to Schre[g].

```
python nl_errors.py data/eval/belth/german-syll/ data/eval/belth/german-syll/nl_german-syll_local_tsl_tsl_canon/
errors of alter at 100 (5026 of which 4723 don't change)
	 ('gantz.', 'gants.', 'gantz.')
	 ('ɔb.', 'ɔp.', 'ɔb.')
	 ('tsug.', 'tsuk.', 'tsug.')
	 ('kʊrtz.', 'kʊrts.', 'kʊrtz.')
	 ('vɛg.', 'vɛk.', 'vɛg.')
	 ('ab.', 'ap.', 'ab.')
	 ('flug.tsɔyg.', 'fluk.tsɔyk.', 'flug.tsɔyg.')
	 ('gɛlb.', 'gɛlp.', 'gɛlb.')
	 ('gɛlb.', 'gɛlp.', 'gɛlb.')
	 ('gɘ.nug.', 'gɘ.nuk.', 'gɘ.nug.')
	 ('gib.', 'gip.', 'gib.')
	 ('mag.', 'mak.', 'mag.')
	 ('platz.', 'plats.', 'platz.')
	 ('hʊb.ʃrau.bɘr.', 'hʊp.ʃrau.bɘr.', 'hʊb.ʃrau.bɘr.')
	 ('dɛs.halb.', 'dɛs.halp.', 'dɛs.halb.')
	 ('flug.ha.fɘn.', 'fluk.ha.fɘn.', 'flug.ha.fɘn.')
	 ('hab.', 'hap.', 'hab.')
	 ('tag.', 'tak.', 'tag.')
	 ('ʃvartz.', 'ʃvarts.', 'ʃvartz.')
	 ('ʃvartz.', 'ʃvarts.', 'ʃvartz.')

errors of copy at 100 (2072 of which 0 don't change)
	 ('mø.rɘ.', 'mø.rɘ.', 'mŸ.rɘ.')
	 ('ʃrɛk.', 'ʃrɛk.', 'ʃrɛg.')
	 ('tɛ.di.bɛ.rɘn.', 'tɛ.di.bɛ.rɘn.', 'tɛ.ti.bɛ.rɘn.')
	 ('gɘ.løst.', 'gɘ.løst.', 'gɘ.list.')
	 ('mø.vɘ.', 'mø.vɘ.', 'mŸ.vɘ.')
	 ('auf.gɘ.løst.', 'auf.gɘ.løst.', 'auf.gɘ.list.')
	 ('mys.te.ri.øs.', 'mys.te.ri.øs.', 'mys.te.ri.is.')
	 ('aus.gɘ.løst.', 'aus.gɘ.løst.', 'aus.gɘ.list.')
	 ('ʃtek.', 'ʃtek.', 'ʃteg.')
	 ('ʃøn.', 'ʃøn.', 'ʃon.')
	 ('gɘ.hørt.', 'gɘ.hørt.', 'gɘ.hirt.')
	 ('ʃø.nɘ.', 'ʃø.nɘ.', 'ʃo.nɘ.')
	 ('ʃø.nɘ.', 'ʃø.nɘ.', 'ʃo.nɘ.')
	 ('hørt.', 'hørt.', 'hirt.')
	 ('ʃø.nɘs.', 'ʃø.nɘs.', 'ʃo.nɘs.')
	 ('ʃø.nɘs.', 'ʃø.nɘs.', 'ʃo.nɘs.')
	 ('lø.vɘ.', 'lø.vɘ.', 'li.vɘ.')
	 ('hø.rɘn.', 'hø.rɘn.', 'hi.rɘn.')
	 ('ʃø.nɘn.', 'ʃø.nɘn.', 'ʃo.nɘn.')
	 ('ʃø.nɘr.', 'ʃø.nɘr.', 'ʃo.nɘr.')
```