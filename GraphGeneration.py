def generate_grid(size = 2):
    """
    Parameters:
    size - an integer to resize the graph if the user
    has a different size screen
    
    Returns:
    The map and the random starting points
    """

    # user can change this value based on the computer screen size (default = 2)
    if int(size) < 0:
        size = 2

    graph_size_x = 100 # pixel size x is double pixel size y
    graph_size_y = 50 # pixel size y is double pixel size x
        
    # size of the figure on the screen
    fig_x = size * (max([graph_size_x,graph_size_y])/min([graph_size_x,graph_size_y])) * 5 # fig x is double fig y
    fig_y = int(fig_x / 2) # fig y is half of fig x
    
    global all_square_coordinates
    # let's see if the all_square_coordinates variable exists
    # if it does we do not intialize it again
    # if it does NOT exist we initialize the variable once to be used throughout
    # every single random graph generation
    try:
        len(all_square_coordinates) # if we can't get the length it doesnt exist
    except: # so let's initialize it
        # generating random x and y coordinates
        x_range = np.arange(0, graph_size_x + 1)
        y_range = np.arange(0, graph_size_y + 1)

        # this double for loop is not O(n^2) it is O(m x n), so it's good - not exponential
        all_square_coordinates = []
        for x_coord in x_range:
            for coordinate_pair in [(x_coord,y) for y in y_range]:
                all_square_coordinates.append(coordinate_pair)
        
    # intializing the fig size (screen size)
    fig_size = (fig_x,fig_y)
    fig = plt.figure()
    fig.set_size_inches(fig_x, fig_y)

    # adding one subplot of our graph (we can add more later)
    ax = fig.add_subplot(1, 1, 1, autoscale_on = False)
    # make the graph 100 x 50 pixels
    ax.scatter([0,100 + 1],[0,50 + 1], color = 'white')
    
    # creating grid lines for x and y
    ax.set_xticks(np.arange(0,graph_size_x + 1))
    ax.set_yticks(np.arange(0,graph_size_y + 1))

    # getting rid of tick labels with an empty list
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # enabling the grid
    ax.grid()

    # calculating size of square to fit the graph
    square_shade_size = fig_x * fig_y / (graph_size_x / graph_size_y)

    # to ensure that we don't randomly pick a cell we have already chosen
    available_coords = np.array(copy.deepcopy(all_square_coordinates))
    
    #############################################################################
    # NORMAL CELLS
    normal_cells_index = np.random.choice(len(available_coords), size = 2500, replace = False)
    normal_cells = available_coords[normal_cells_index]
    
    # deleting already used cells
    available_coords = np.delete(available_coords, np.where(available_coords == normal_cells))
    print(len(available_coords))
    # getting x and y pairs
    normal_cells_x = np.array([coord[0] for coord in normal_cells])
    normal_cells_y = np.array([coord[1] for coord in normal_cells])
    
    # shade in cells and color points
    # using broadcasting to apply a -0.5 on all x  y  to center our shaded square on the graph
    normal_cells_ax = ax.scatter(normal_cells_x+.5, normal_cells_y+.5, color = 'orange', s = square_shade_size, marker = ",",
                             label = "Normal Cells")
    #############################################################################
    # HIGHWAY CELLS
    highway_cells_index = np.random.choice(len(available_coords), size = 1000, replace = False)
    highway_cells = available_coords[highway_cells_index]

    # deleting already used cells
    for cell in highway_cells:
        available_coords = np.delete(available_coords, np.array((cell[0],cell[1])), None)
    
    # getting x and y pairs
    highway_cells_x = np.array([coord[0] for coord in highway_cells])
    highway_cells_y = np.array([coord[1] for coord in highway_cells])
    
    # using broadcasting to apply a -0.5 on all x and y coordinates to center our shaded square on the graph
    highway_cells_ax = ax.scatter(highway_cells_x+.5, highway_cells_y+.5, color = 'red', s = square_shade_size, marker = ",",
                             label = "Highway Cells")
    ##############################################################################
    # HARD TO TRAVERSE CELLS
    hard_cells_index = np.random.choice(len(available_coords), size = 1000, replace = False)
    hard_cells = available_coords[hard_cells_index]
        
    # deleting already used cells
    for cell in hard_cells:
        available_coords = np.delete(available_coords, np.array((cell[0],cell[1])), None)
    
    # getting x and y pairs
    hard_cells_x = np.array([coord[0] for coord in hard_cells])
    hard_cells_y = np.array([coord[1] for coord in hard_cells])
    
    # shade in cells and color points
    # using broadcasting to apply a -0.5 on all x and y coordinates to center our shaded square on the graph
    hard_cells_ax = ax.scatter(hard_cells_x+.5, hard_cells_y+.5, color = 'pink', s = square_shade_size, marker = ",",
                             label = "Hard To Traverse Cells")
    ##############################################################################
    # CHOOSING A RANDOM START CELL
    # plotting the random starting point
    random_pair = random.choice([normal_cells] + [highway_cells] + [hard_cells])
    random_x_start = random_pair[0]
    random_y_start = random_pair[1]
    ##############################################################################
    
    # BLOCKED CELLS, rest of the available coords are now blocked
    blocked_cells = available_coords

    # getting x and y pairs
    blocked_cells_x = np.array([coord[0] for coord in blocked_cells])
    blocked_cells_y = np.array([coord[1] for coord in blocked_cells])
    
    # shade in cells and color points
    # using broadcasting to apply a -0.5 on all x and y coordinates to center our shaded square on the graph
    blocked_cells_ax = ax.scatter(blocked_cells_x+.5, blocked_cells_y+.5, color = 'black', s = square_shade_size, marker = ",",
                             label = "Blocked Cells")
    ##############################################################################

    starting_point = ax.scatter(random_x_start+.5,random_y_start+.5, color = 'blue', s = square_shade_size * 4.5, marker = '*',
                                   label = "Start")
    
    # remove the exta ticks (making our graph look pretty)
    plt.tick_params(bottom = False, left = False)

    # darken the grid lines
    x_gridlines = ax.get_xgridlines()
    y_gridlines = ax.get_ygridlines()
    for x in x_gridlines:
        x.set_linewidth(size)
    for y in y_gridlines:
        y.set_linewidth(size)
    
    # creating handles
    handles = [blocked_cells_ax,normal_cells_ax,highway_cells_ax,hard_cells_ax,starting_point]

    # legend for labels
    ax.legend(handles = handles)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5,prop={'size': size * 10})
    
    plt.show()
    
    
    # creating a map with the cells and their corresponding types
    unknown_map = dict()

    counter = 0
    all_cells = [(normal_cells_x,normal_cells_y), (highway_cells_x,highway_cells_y), 
                 (hard_cells_x,hard_cells_y), (blocked_cells_x,blocked_cells_y)]
    # iterating through each cell type and creating a map to traverse for testing
    for cell_type in all_cells:
        counter += 1
        if counter == 1:
            cell_string = 'normal'
        elif counter == 2:
            cell_string = 'highway'
        elif counter == 3:
            cell_string = 'hard'
        elif counter == 4:
            cell_string = 'blocked'
        for x, y in zip(cell_type[0],cell_type[1]):
            unknown_map[(x,y)] = cell_string 
    
    return (random_x_start,random_y_start), unknown_map
