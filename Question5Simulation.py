import numpy as np # vectorization and arrays
import pandas as pd # visualization
import random # for randomization
import matplotlib.pyplot as plt # for graphing
import copy # to mutate lists that we want to mutate
import os # to make folder and save files

# given actions and sensor readings
actions = ["Right", "Right", "Down", "Down"]
sensor_readings = ["N","N","H","H"]
allowable_start_positions = [(0,0),(0,1),(0,2),
                             (1,0),(1,1),(1,2),
                             (2,0),(2,2)]
our_world = [["H", "H", "T"],
             ["N", "N", "N"],
             ["N", "B", "H"]]

# will hold our probabilities for certain positions (4 graphs)
probabilities = {"Graph 1": {(0,0): 0, (0,1): 0,(0,2): 0,
                             (1,0): 0,(1,1): 0,(1,2): 0,
                             (2,0): 0,(2,2): 0},
                 "Graph 2": {(0,0): 0, (0,1): 0,(0,2): 0,
                             (1,0): 0,(1,1): 0,(1,2): 0,
                             (2,0): 0,(2,2): 0},
                 "Graph 3": {(0,0): 0, (0,1): 0,(0,2): 0,
                             (1,0): 0,(1,1): 0,(1,2): 0,
                             (2,0): 0,(2,2): 0},
                 "Graph 4": {(0,0): 0, (0,1): 0,(0,2): 0,
                             (1,0): 0,(1,1): 0,(1,2): 0,
                             (2,0): 0,(2,2): 0}}

trials = {"Graph 1": 0,"Graph 2": 0,"Graph 3": 0,"Graph 4": 0,}

# how many times are we running the above actions?
for _ in range(0,10_000_000):

    # picking a random point that is not a blocked cell
    pos_y, pos_x  = random.choice(allowable_start_positions)
    sensor_readings = []

    graph_number = 0
    # going through actions
    for action in actions:

        # for our 4 different graphs
        graph_number += 1

        # we make the correct action 90% of the time
        if np.random.choice([True,False], p = [0.90,0.10]):

            if action == "Right":

                if pos_x < 2 and our_world[pos_y][pos_x + 1] != "B":
                    pos_x += 1

            elif action == "Down":

                if pos_y < 2 and our_world[pos_y + 1][pos_x] != "B":
                    pos_y += 1

        # getting sensor readings
        current_terrain = our_world[pos_y][pos_x]
        all_terrain_types = ["N","T","H"]
        del all_terrain_types[all_terrain_types.index(current_terrain)]

        # now let's use the broken sensor...
        sensor_reading = np.random.choice([current_terrain,all_terrain_types[0],all_terrain_types[1]], p = [0.90,0.05,0.05])
        sensor_readings.append(sensor_reading)

        # we have to keep track of the probabilites
        if graph_number == 1 and sensor_readings == ["N"]:

            # incrememnting trials and position place
            trials["Graph 1"] += 1
            probabilities["Graph 1"][(pos_y,pos_x)] += 1

        elif graph_number == 2 and sensor_readings == ["N","N"]:

            # incrememnting trials and position place
            trials["Graph 2"] += 1
            probabilities["Graph 2"][(pos_y,pos_x)] += 1

        elif graph_number == 3 and sensor_readings == ["N","N","H"]:

            # incrememnting trials and position place
            trials["Graph 3"] += 1
            probabilities["Graph 3"][(pos_y,pos_x)] += 1

        elif graph_number == 4 and sensor_readings == ["N","N","H","H"]:

            # incrememnting trials and position place
            trials["Graph 4"] += 1
            probabilities["Graph 4"][(pos_y,pos_x)] += 1   

# calculating probabilites 
for key in probabilities:
    for coords in probabilities[key]:
        # dividing outcomes and trials to get a probability
        if trials[key] != 0:
            probabilities[key][coords] = probabilities[key][coords] / trials[key]
        else:
             probabilities[key][coords] = 0

# confirming that all probabilities sum to 1 in each graph
for g in ["Graph 1","Graph 2","Graph 3","Graph 4"]:
    ii = 0
    for i in probabilities[g]:
        ii += probabilities[g][i]
    print(ii, g)

print(probabilities["Graph 1"])
print(probabilities["Graph 2"])
print(probabilities["Graph 3"])
print(probabilities["Graph 4"])
