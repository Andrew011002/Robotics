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
        for theta in np.arange(self.fov + 0.1, step=0.1):
            pos = calc_point(self.beam, theta)
            m = calc_slope((0, 0), pos)
            self.trace(m, endpoint=pos[0])
    
    def trace(self, m, endpoint):
        if m == np.inf:
            y = 1
            while self.free((0, y)) and y < self.beam + 1:
                self.place((0, y), 255)
                y += 1
        elif m == -np.inf:
            y = -1
            while self.free((0, y)) < -self.beam - 1:
                self.place((0, y), 255)
                y -= 1
        else:
            return None


    # places object at given r,c indices & color
    def place(self, pos, rgb=0):
        gpos = convert(self.center, pos)
        self.grid[gpos[0], gpos[1]] = rgb

    def free(self, pos):
        gpos = convert(self.center, pos)
        if not np.array_equal(self.grid[gpos[0], gpos[1]], (0, 0, 0)):
            return True
        return False

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

def convert(origin, pos):
    # convert from (x, y) coordincates to grid indices (r, c)
    return origin[0] - pos[1], origin[1] + pos[0]

def revert(origin, indices):
    # revert from grid indices (r, c) to (x, y) coordinates
    return indices[1] - origin[1], origin[0] - indices[0]

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

def inbounds(gpos, size):
    # out of bounds for row
    if gpos[0] < 0 or gpos[0] > size[1] - 1:
        return False
    # out of bounds for col
    if gpos[1] < 0 or gpos[1] > size[0] - 1:
        return False
    return True

def calc_slope(p1, p2):
    return round((p2[1] - p1[1]) / (p2[0] - p1[0]), 3)


if __name__ == "__main__":
    local_map = LocalMap(100, 360)
    local_map.update([(3, 1)])
    local_map.raytrace()
    local_map.view()
    



    
