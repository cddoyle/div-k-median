#!/bin/sh
nohup python3 -u scalability.py -batch feasibility -results scaling-feasibility &
nohup python3 -u scalability.py -batch bicriteria -obj kmedian -results scaling-bicriteria &
