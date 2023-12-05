# Example of simple Rejection Sampling

import numpy as np

target_prob = 0.04
samples = []
allowed = 0

for i in range(10_000):
    x = 0
    allow = False
    if np.random.uniform(0, 1) < 0.5:
        allow = True
        allowed += 1

    if len(samples) > 0:
        g = allowed / len(samples)
    else:
        g = 0.5
    
    acceptance_prob = target_prob / (g + 1e-10)
    if allow and np.random.uniform(0, 1) < acceptance_prob:
        x = 1
    
    samples.append(x)

print(sum(samples) / len(samples))