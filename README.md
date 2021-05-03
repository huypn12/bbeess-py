# bbeess-py
Python implementation of Bayesian parameter synthesis frameworks RF-SMC and SMC-ABC-SMC.

## Requirements
Storm and PRISM must be installed before. Storm installation must provide headers and shared libraries available on system path. For example
```
$ storm
Storm 1.6.4 (dev)

$ prism
PRISM
=====

Version: 4.6
Date: Mon May 03 12:04:54 CEST 2021
Memory limits: cudd=1g, java(heap)=1g
Command line: prism
Usage: prism [options] <model-file> [<properties-file>] [more-options]

For more information, type: prism -help
```

## Installation
```
$ pipenv shell
$ pipenv install --dev
```

## Running experiments
```
$ source env.sh
$ python experiments/[experiment_name]
```