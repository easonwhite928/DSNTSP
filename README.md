# Dual Sequential Network for Temporal Sets Prediction

## Introduction

DSNTSP (Dual Sequential Network for Temporal Sets Prediction) is a novel model used for temporal sets prediction prediction problem.

Please refer to our SIGIR 2020 paper ["Dual Sequential Network for Temporal Sets Prediction"](https://dl.acm.org/doi/abs/10.1145/3397271.3401124) for more details.

## Project Architecture

The descriptions of principal files in this project are explained as follows:

- `config/`: containing JSON format configuration files.
- `data/`: containing JSON format datasets and `DataLoader` implementations for our models.
- `learner/`: codes for the learner and metric definitions for our models.
- `model/`: codes for our temporal sets prediction models.
- `paper/`: containing our published paper.
- `registry/`: codes for registering models, only the registered model classes can be recognized by the argument parser.
- `run.py`: script used for training and testing models.

## How to use:

Train the model:  
```
python run.py --mode=train --config=./config/taobao_buy/dsntsp.json --cuda=0
```

Test the model:
```
python run.py --mode=test --config=./config/taobao_buy/dsntsp.json --cuda=0
```

You can modify the value of the attribute `best_epoch` in the JSON format configuration file in the `config/` to choose which trained model to test.

