import numpy as np
import matplotlib.pyplot as plt





class LocalMap:
    
    def __init__(self, dim, fov):
        self.grid = np.zeros((dim, dim, 3), dtype=int) + 211
        self.center = (dim // 2, dim // 2)
        self.grid[self.center[1], self.center[0]] = (0, 255, 0) 
        self.dim = dim
        self.beam = dim // 2
        self.fov = fov
        self.size = (dim, dim)

    # given some pos grid will update w/ objs
    def update(self, objs):
        for pos in objs:
            self.place(pos)
            
    def raytrace(self):
        pass

    # places object at given r,c indices & color
    def place(self):
        pass

    def free(self):
        pass

    def cells(self):
        return self.grid

    def view(self, cmap=None):
        if cmap is None:
            cmap = "gray"
        plt.imshow(self.grid, cmap=cmap)
        plt.show()

class GlobalMap:
    pass

class OccupancyMap:
    pass

def calc_theta(pos, deg=True):
    x, y = pos
    theta =  np.arctan2(y, x)
    if deg:
        return theta * 180 / np.pi
    return theta

def calc_point(dist, theta):
    rad = np.pi / 180 * theta
    return np.sin(rad) * dist, np.cos(rad) * dist

def calc_length(pos):
    x, y = pos
    return np.sqrt(x**2 + y**2)

def rotate_pos(pos, theta):
    # get vector length & theta
    dist = calc_length(pos)
    otheta = calc_theta(pos)
    # find new pos
    theta = (360 + otheta + theta) % 360 * np.pi / 180
    x, y = dist * np.cos(theta), dist * np.sin(theta)
    return int(np.rint(x)), int(np.rint(y))


if __name__ == "__main__":
    local_map = LocalMap(11, 3)
    
    



    
