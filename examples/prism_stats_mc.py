sample_stats_mc_output = """
PRISM
=====

Version: 4.6
Date: Tue Jan 19 21:44:20 CET 2021
Memory limits: cudd=1g, java(heap)=1g
Command line: prism ../../../examples/data/die.pm ../../../examples/data/die.pctl -prop 1 -sim

Parsing model file "../../../examples/data/die.pm"...

Type:        DTMC
Modules:     die 
Variables:   s d 

Parsing properties file "../../../examples/data/die.pctl"...

1 property:
(1) P=? [ F "one" ]

---------------------------------------------------------------------

Simulating: P=? [ F "one" ]

Simulation method: CI (Confidence Interval)
Simulation method parameters: width=unknown, confidence=0.01, number of samples=1000
Simulation parameters: max path length=10000

Sampling progress: [ 10% 20% 30% 40% 50% 60% 70% 80% 90% 100% ]

Sampling complete: 1000 iterations in 0.072 seconds (average 7.2e-05)
Path length statistics: average 4.4, min 3, max 12

Simulation method parameters: width=0.02949365114353477, confidence=0.01, number of samples=1000

Simulation result details: confidence interval is 0.155 +/- 0.02949365114353477, based on 99.0% confidence level

Result: 0.155
"""
