import numpy as np
import matplotlib.pyplot as plt

class LocalMap:

    def __init__(self, width, height, pos, rgb=(255, 0, 0)):
        self.grid = np.zeros((width, height, 3), dtype=np.uint8) + 211
        self.origin = (width - 1) // 2, (height - 1) // 2
        self.size = (width, height)
        self.pos = pos
        self.rgb = rgb
        self.objects = []
        self.place_obj(pos, rgb, object=False)

    def place_obj(self, pos, rgb=(0, 0, 0), object=True):
        # convert (x, y) coordinates to grid indices
        gpos = convert(self.origin, pos)
        self.grid[gpos[0], gpos[1], :] = rgb
        if object:
            self.objects.append((pos, rgb))

    def get_objs(self):
        return self.objects
    
    def get_bot(self):
        # get (x, y) coordinates and color
        return self.pos, self.rgb

    def show(self, cmap="gray"):
        plt.imshow(self.grid, cmap=cmap)
        plt.show()

    def __str__(self):
        return str(self.grid.T)

class GlobalMap:

    def __init__(self, width, height, origin):
        self.grid = np.zeros((width, height, 3), dtype=np.uint8) + 211
        self.origin = origin
        self.size = (width, height)
        self.objects = []
        self.bot = None

    def place_local_map(self, local_map, rotation):
        # place robot
        pos, rgb = local_map.get_bot()
        pos = rotate_pos(pos, rotation)
        gpos = convert(self.origin, pos)
        self.place_obj(gpos, rgb, object=False)
        # place objects
        objs = local_map.get_objs()
        for pos, rgb in objs:
            pos = rotate_pos(pos, rotation)
            gpos = convert(self.origin, pos)
            self.place_obj(gpos, rgb)

    def place_obj(self, gpos, rgb=(0, 0, 0), object=True):
        # place obj based on grid indices
        self.grid[gpos[0], gpos[1], :] = rgb
        if object:
            self.objects.append((gpos, rgb))
        else:
            self.bot = gpos, rgb

    def get_objs(self):
        return self.objects

    def get_bot(self):
        return self.bot

    def show(self, cmap="gray"):
        plt.imshow(self.grid, cmap=cmap)
        plt.show()

    def __str__(self):
        return str(self.grid.T)

def convert(origin, pos):
    # convert from (x, y) coordincates to grid indices
    return origin[0] - pos[1], origin[1] + pos[0]

def revert(origin, indices):
    # revert from grid indices to (x, y) coordinates
    return indices[1] - origin[1], origin[0] - indices[0]

def calc_theta(pos, deg=True):
    x, y = pos
    theta =  np.arctan2(y, x)
    if deg:
        return theta * 180 / np.pi
    return theta

def calc_length(pos):
    x, y = pos
    return np.sqrt(x**2 + y**2)

def rotate_pos(pos, theta):
    # get vector length and theta
    dist = calc_length(pos)
    otheta = calc_theta(pos)
    # find new pos
    theta = (360 + otheta + theta) % 360 * np.pi / 180
    x, y = dist * np.cos(theta), dist * np.sin(theta)
    return int(np.rint(x)), int(np.rint(y))


if __name__ == "__main__":
    local_map = LocalMap(26, 26, (0, 0))
    local_map.place_obj((-10, 10))
    global_map = GlobalMap(101, 101, (50, 50))
    global_map.place_local_map(local_map, 170)
    local_map.show()
    global_map.show()


    
