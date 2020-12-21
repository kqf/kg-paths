# KG paths [![Build Status](https://travis-ci.com/kqf/kg-paths.svg?branch=master)](https://travis-ci.com/kqf/kg-paths)

Predict a smooth path between the given two nodes in a graph.


## Dataset

The dataset is `yoochoose` (RecSys15 challenge). It contains the history sessions (clicks) on the e-commerce platform. It is available on kaggle.
In order to download one needs to accept the rules on kaggle and place kaggle API token file in your home directory (as mentioned in their docs `~/.kaggle/kaggle.json`).

Use this command to download the data (or do it manually, no token required)

```bash
make data/
```


## Installation
To install the packages do:

```bash
pip instal -r requirements.txt
pip install -e .
```

## Run 
It should not take long to run anything (except for preprocessing) as everything works on samples
```bash

# Runs the baseline approach
make baseline

# Runs the search with heuristics
make
```
