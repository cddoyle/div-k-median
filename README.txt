## Overview
This software repository contains an experimental software implementation of
algorithms accompanying the paper "Clustering with fair center representation:
parameterized approximation algorithms and heuristics". The software is
written in the Python programming language.

See LICENSE.txt for copyright details.


## Requirements
See `requirements.txt` for python package requirements.


## General comments
The source code is written in a python programming language. The files are names
to ensure the corresponding functionality of the implementation, for instance
file `coresets.py` contains implementation for generating coresets.
Additionally, each file contains test stubs to demonstrate the usage of APIs.

The current version of the source code is the experimental version. A more polished version of the source code along with a user guide will be made available in the final release.


## Scalability experiments
Usage: 
```bash
./run-scalability.sh
```

## Input

Dataset file format: The file contains `N` data points with `D` dimensions for each
point. Each line in the input file is a datapoint with `D` entries
separated by a comma as a delimiter. See `data.txt` for an example.

Color matrix format: The file contains `t` lines where each line represents a
group. Each line has `N` entries separated by a comma as a delimiter. An entry 1 in line `j` at position `i` imply that datapoint `i` belongs to group `j`, and an
entry `0` implies that the data point does not belong to a group. See `groups.txt` for example.
