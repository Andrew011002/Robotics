import numpy as np
import matplotlib.pyplot as plt

def map_objects(dist, pos=(0, 0), width=100, height=100):
    # set up grid and angle
    grid = np.zeros((width, height, 3), dtype=np.uint8) + 211
    offset = ((width - 1) // 2, (height - 1) // 2)
    grid[pos[1] + offset[1], pos[0] + offset[0], 1:] = 0
    theta = 360 / len(dist) 
    place_point(grid, pos, (-1, 2), offset)

    for i, v in enumerate(dist):
        # get x and y components of distances
        rad = (theta * i * np.pi / 180)
        
        
    return grid

def place_point(grid, origin, point, offset):
    i, j = origin[1] - point[1], origin[0] + point[0]
    grid[i + offset[0], j + offset[1], :] = 0

if __name__ == "__main__":
    low, high = 5, 10
    beams = 10
    dist = np.random.randint(low, high + 1, (beams, ))
    grid = map_objects(dist, pos=(0, 0), width=20, height=20)
    plt.imshow(grid, cmap="rainbow")
    plt.show()

