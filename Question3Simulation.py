import random
import math
import numpy as np
import matplotlib.pyplot as plt

def run_trial():
    """
    Determines the probability of 
    A, B, C, D given the bayesian network
    """
    
    # probability of A = 0
    A = False

    # probability of B is 0.90
    B = np.random.choice([True,False], p = [0.90,0.10])

    # what is C going to be based off A and B?
    if A == False and B == True:
        C = np.random.choice([True,False], p = [0.50,0.50])
    else:
        C = False

    # deciding the value for D
    if B == True and C == True:
        D = np.random.choice([True,False], p = [0.75,0.25])
    elif B == True and C == False:
        D = np.random.choice([True,False], p = [0.10,0.90])
    elif B == False and C == True:
        D = np.random.choice([True,False], p = [0.50,0.50])
    elif B == False and C == False:
        D = np.random.choice([True,False], p = [0.20,0.80])
            
    # returning al the values
    return (A, B, C, D)

probs = {"A":[],"B":[],"C":[],"D":[]}
for _ in range(0,1_000):
    A, B, C, D = run_trial()
    probs["A"].append(A)
    probs["B"].append(B)    
    probs["C"].append(C)    
    probs["D"].append(D)  
    
# Finding P(B|C)
numerator = 0
denominator = 0
for b,c in zip(probs["B"],probs["C"]):
    
    if b == c and c == True:
        numerator += 1
    if c == True:
        denominator += 1
print("P(B|C) = " + str(numerator/denominator))

# Finding P(D|C)
numerator = 0
denominator = 0
to_graph = []
for d,c in zip(probs["D"],probs["C"]):
    
    if d == c and c == True:
        numerator += 1
    if c == True:
        denominator += 1

    if denominator > 0:
        to_graph.append(numerator/denominator)
    
print("P(D|C) = " + str(numerator/denominator))

# Finding P(D|¬A,B)
numerator = 0
denominator = 0
for d,a,b in zip(probs["D"],probs["A"],probs["B"]):
    
    if d == True and a == False and b == True:
        numerator += 1
    if a == False and b == True:
        denominator += 1
print("P(D|¬A,B) = " + str(numerator/denominator))

plt.plot(to_graph)
plt.title("Approximated Value for P(D|C)")
plt.xlabel("Trials")
plt.ylabel("Probability")
plt.show()