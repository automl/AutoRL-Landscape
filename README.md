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

## Dataset Description

We provide the data collected as part of the experimental part of the thesis in
`data/kwie/dqn_cartpole_3_phases.csv`. The dataset comprises the following columns:

| Column Name   | Description    |
|--------------- | --------------- |
| ID (empty)                                      | unique `wandb` run id |
| name                                            | unique `wandb` human readable run name |
| ls.gamma                                        | discount factor |
| ls.learning_rate                                | learning rate |
| final_eval_i/ep_length_hist                     | eval episode length numpy histogram for ith final stage |
| final_eval_i/ep_lengths                         | eval episode lengths for ith final stage |
| final_eval_i/mean_ep_length                     | eval mean episode length for ith final stage |
| final_eval_i/mean_return                        | eval mean return for ith final stage |
| final_eval_i/return_hist                        | eval return numpy histogram for ith final stage |
| final_eval_i/returns                            | eval returns for ith final stage |
| ls_eval/*                                       | same information, but for the landscape stage |
| meta.ancestor                                   | path to folder where snapshot of last phase's best policy is saved |
| meta.conf_index                                 | index for this configuration (unique per phase) |
| meta.phase                                      | phase of this run (starting with 0!) |
| meta.seed                                       | RL algorithm seed of this run |
| meta.timestamp                                  | timestamp of experiment start |

These further columns hold information about the experiment (all rows hold the same value):

| Column Name   | Description    |
|--------------- | --------------- |
| conf.agent.hps.*                                | other optimized static hyperparameters for the RL algorithm (from sb3 zoo)|
| conf.agent.name                                 | RL algorithm name |
| conf.agent.zoo_optimal_ls.*                     | optimized hyperparameter values for landscape hyperparameters (if ls dimension is specified as `Constant`) |
| conf.env.name                                   | gym environment name |
| conf.eval.final_eval_episodes                   | number of landscape evaluation episodes |
| conf.eval.final_eval_start                      | first final evaluation (multiplier for total time) |
| conf.eval.final_eval_times                      | number of final evaluation stages |
| conf.eval.freq_eval_episodes                    | number of learning curve evaluation episodes |
| conf.eval.freq_eval_interval                    | interval in between learning curve evaluations |
| conf.eval.ls_eval_episodes                      | number of landscape evaluation episodes |
| conf.ls.dims                                    | search space dimension info (parsed in `autorl_landscape/util/download.py`) |
| conf.ls.type                                    | sampling strategy |
| conf.num_confs                                  | number of distinct hyperparameter configurations |
| conf.num_seeds                                  | number of runs per configuration |
| conf.phases                                     | time steps for landscape stages |
| conf.seeds.*                                    | random seeds (conf.seeds.agent is first agent seed, further seeds are simply increments of this number)|
| conf.slurm.*                                    | SLURM cluster configuration |
| conf.total_timesteps                            | time step for final stage |
| conf.wandb.entity                               | `wandb` entity name |
| conf.wandb.project                              | `wandb` project name|
| ls.exploration_final_eps                        | final exploration rate |
| ls.nn_length                                    | neural net length |
| ls.nn_width                                     | neural net width |
