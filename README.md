# MCTSOC: Monte Carlo Tree Search in Optimal Control

This repository contains a series of experiments involving Monte Carlo Tree Search (MCTS) algorithms in the 
context of Optimal Control tasks. They are part of my Bachelor Thesis:

#### [An Evaluation of MCTS Methods for Continuous Control Tasks](https://www.dropbox.com/s/9acbtihfmagn7el/Bachelor_Thesis___An_Evaluation_of_MCTS_Methods_for_Continuous_Control_Tasks_FINAL.pdf?dl=0) 

In the first part of the experiments we look at pure mcts which uses 
[this implementation](https://github.com/PatrickKorus/mcts-general).
The second part uses several variations of [muzero-general](https://github.com/werner-duvaud/muzero-general) which
can be found in the branches of [my fork](https://github.com/PatrickKorus/muzero-general). 

This repository is primarily for the reproducability of the experiments. For running your own experiments we recommend
looking at the above mentioned repositories. 

## Dependencies

All dependencies including the variation of [muzero](https://github.com/PatrickKorus/muzero-general) can be installed
by running

```shell script
pip install -r requirements.txt
```

For the `pgf` plot back end to work you need a valid lualatex installation. On a unix machine please run
```shell script
sudo apt install texlive-luatex
```

I recommend using a clean virtual environment, see conda.

## How to run

Run
```shell script
python3 mcts_general_main.py
```
for the general MCTS experiments,

```shell script
python3 muzero_general_main.py
```
for the MuZero experiments,

```shell script
python3 dqn_baseline_main.py
```
for the DQN experiments.

The code that created the plots in the thesis can be found in each `visualization.py` file.

