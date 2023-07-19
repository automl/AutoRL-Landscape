# AutoRL-Landscape

## Installation

```bash
git clone https://github.com/automl-private/AutoRL-Landscape.git
cd AutoRL-Landscape

conda env create -f env-setup.yaml
conda activate autorl-landscape
pip install -e . # install the executable of this repository into the newly made environment
```

### Additional dependency for Mujoco
The environments require Mujoco which needs to be installed in addition to the python bindings.
You can find installation info [here](https://github.com/openai/mujoco-py#install-mujoco).

### Additional dependency for `phases ana modalities ...`

The modality analysis uses `python3-libfolding`, which depends on the `C++` library
[libfolding](https://asiffer.github.io/libfolding/cpp). This extra dependency
may be installed from source.

### Additional OpenGL dependency

The environment does not provide an OpenGL header file. A workaround is to
softlink the system installation as described in `env-setup.yaml`.

## Usage

Code can be run through the `phases` command line interface. Before use, `wandb` may need to be set
up to allow for data to be uploaded to the user's account.

When running (`phases run`) experiments, run data is uploaded to `wandb` as configured by the
command arguments. The `wandb` project should be unique per invocation of `phases run` for easy
access of the datasets through `phases dl`.

Usage examples:

```bash
# Commands used to create the final datasets:
phases run combo=dqn_cartpole ls=dqn slurm=local phases=150k    num_confs=128 num_seeds=5 wandb.entity=entity_name wandb.project=project_name wandb.experiment_tag=experiment_tag
phases run combo=sac_hopper   ls=sac slurm=local phases=500k_4  num_confs=128 num_seeds=5 wandb.entity=entity_name wandb.project=project_name wandb.experiment_tag=experiment_tag
phases run combo=ppo_walker   ls=ppo slurm=local phases=150k    num_confs=128 num_seeds=5 wandb.entity=entity_name wandb.project=project_name wandb.experiment_tag=experiment_tag

# Download the data of an experiment to a single file into the data/ directory:
phases dl entity_name project_name experiment_tag

# Create visualizations, specifying to either construct an ILM or an IGPR model if a landscape model is required:
# You can also run `bash make_plots.sh ALGORITHM` with ALGORITHM either sac, dqn or ppo
# --savefig saves the figures to disk to `figures/ALGORITHM/...
phases ana maps data/ALGORITHM.csv {ilm,igpr}  --savefig
phases ana graphs data/ALGORITHM.csv {ilm,igpr} --savefig
phases ana modalities data/ALGORITHM.csv --savefig
phases ana concavity data/ALGORITHM.csv {ilm,igpr}  --savefig
```

## Experiment Configuration

We use `hydra` to configure the data collection process. Used values can be found in the `conf/`
directory. `conf/config.yaml` is the root of the configuration tree. The different subfolders hold
exchangeable configurations for RL contexts (`conf/combo/`), the search space (`conf/ls/`), the
phase configuration for the data collection (`conf/phases/`), and the SLURM cluster configuration
(`conf/slurm/`).
You can have a look at the slurm config files to configure your cluster and save it to `confs/slurm/yourconfig.yaml`. You then need to add  `slurm=yourconfig` to the commandline, as seen in the runcommands above. The slurm interface is managed by hydra, see [here](https://hydra.cc/docs/plugins/submitit_launcher/) for more information.

## Dataset Description

We provide the data collected as part of the experimental part of the thesis in
`data/kwie/dqn_cartpole_3_phases.csv`. The dataset comprises the following columns:

| Column Name   | Description    |
|--------------- | --------------- |
| ID (empty)                                      | unique `wandb` run id |
| name                                            | unique `wandb` human readable run name |
| ls.*                                            | hyperparameters varied as part of the landscape analyis |
| final_eval_i/ep_length_hist                     | eval episode length numpy histogram for ith final stage |
| final_eval_i/ep_lengths                         | eval episode lengths for ith final stage |
| final_eval_i/mean_ep_length                     | eval mean episode length for ith final stage |
| final_eval_i/mean_return                        | eval mean return for ith final stage |
| final_eval_i/return_hist                        | eval return numpy histogram for ith final stage |
| final_eval_i/returns                            | eval returns for ith final stage |
| ls_eval/*                                       | same information, but for the landscape stage |
| meta.ancestor                                   | path to folder where snapshot of last phase's best policy is saved |
| meta.conf_index                                 | index for this configuration (unique per phase) |
| meta.phase                                      | phase of this run (starting with 1!) |
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
