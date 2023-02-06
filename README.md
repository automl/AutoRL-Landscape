# AutoRL-Landscape

## Installation

```bash
git clone https://github.com/automl-private/AutoRL-Landscape.git
cd AutoRL-Landscape

conda env create -f ma-kwie.yaml
conda activate ma-kwie
pip install -e . # install the executable of this repository into the newly made environment
```

### Additional dependency for `phases ana modalities ...`

This test uses
[python3-libfolding](https://github.com/asiffer/python3-libfolding), which is
dependent on the `C++` libraries
[libfolding](https://asiffer.github.io/libfolding/cpp) and
[armadillo](https://gitlab.com/conradsnicta/armadillo-code). The former can be
installed through `pip`, while the latter can be installed following the
upstream instructions.

## Usage

Code can be run through the `phases` command line interface. Before use, `wandb` may need to be set
up to allow for data to be uploaded to the user's account.

When running (`phases run`) experiments, run data is uploaded to `wandb` as configured by the
command arguments. The `wandb` project should be unique per invocation of `phases run` for easy
access of the datasets through `phases dl`.

Usage examples:

```bash
# Reproduce the experiment from the thesis on the local machine:
phases run combo=dqn_cartpole ls=sobol_2 slurm=local phases=100k num_confs=256 num_seeds=5 wandb.entity=entity_name wandb.project=project_name

# Download the data of an experiment to a single file into the data/entity_name directory:
phases dl entity_name project_name

# Create visualizations, specifying to either construct an ILM ("rbf") or an IGPR ("triple-gp") model if a landscape model is required:
phases ana maps --data=path/to/data.csv --model={rbf,triple-gp} [--grid-length=grid_length]
phases ana graphs --data=path/to/data.csv --model={rbf,triple-gp}
phases ana modalities --data=path/to/data.csv [--grid-length=grid_length]
phases ana concavity --data=path/to/data.csv --model={rbf,triple-gp} [--grid-length=grid_length]
```

## Experiment Configuration

We use `hydra` to configure the data collection process. Used values can be found in the `conf/`
directory. `conf/config.yaml` is the root of the configuration tree. The different subfolders hold
exchangeable configurations for RL contexts (`conf/combo/`), the search space (`conf/ls/`), the
phase configuration for the data collection (`conf/phases/`), and the SLURM cluster configuration
(`conf/slurm/`).
