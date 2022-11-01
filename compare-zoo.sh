#!/bin/bash

for i in {1..5}
do
    # python train.py --algo dqn --env CartPole-v1 --eval-freq 1000 --eval-episodes 10 --n-eval-envs 1 --tensorboard-log tensorboard -n 100000 --track --wandb-project-name zoo_dqn --seed $i &
    python main.py seeds=[$i] &
done


