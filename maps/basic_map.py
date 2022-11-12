import numpy as np
import matplotlib.pyplot as plt




class LocalMap:
    
    def __init__(self, dim, nearest=False):
        if (dim % 2 == 0):
            raise ValueError("dim must be odd")
        self.dim = dim
        self.beam = dim // 2
        self.size = (dim, dim)
        self.center = (dim // 2, dim // 2)
        self.nearest = nearest
        self.grid = np.zeros((dim, dim, 3), dtype=int) + 211
        self.grid[self.center[1], self.center[0]] = (0, 255, 0) 
            
    def place_objs(self, objs):
        # (dist, angle)
        for dist, theta in objs:
            pos = calc_point(dist, theta)
            if not self.inbounds(pos):
                continue
            self.place(pos)

    # finds ray given a position
    def raytrace(self, pos, delta=0.25):
        grad = calc_grad((0, 0), pos)
        if isinstance(grad, tuple):
            for y in np.arange(0, self.beam + 1):
                pos = (0, y)
                if self.free(pos):
                    self.place(pos, (255, 255, 255))
                else:
                    break
        else:
            pass

    # place pixel given (x, y) localization 
    def place(self, pos, rgb=(0, 0, 0)):
        if not self.inbounds(pos):
            raise ValueError(f"can't place pixel at {pos}")
        if self.free(pos, rgb=(0, 255, 0)):
            gpos = convert(self.center, pos, self.nearest)
            self.grid[gpos[0], gpos[1]] = rgb

    # determines if pixel at (x, y) location is not an rgb color
    def free(self, pos, rgb=(0, 0, 0)):
        if not self.inbounds(pos):
            raise ValueError(f"can't place pixel at {pos}")
        gpos = convert(self.center, pos, self.nearest)
        pixel = self.grid[gpos[0], gpos[1]].squeeze()
        if tuple(pixel) == rgb:
            return False
        return True

    def cells(self):
        return self.grid

    # shows current state of map
    def view(self, cmap=None):
        if cmap is None:
            cmap = "gray"
        plt.imshow(self.grid, cmap=cmap)
        plt.show()

    # indicates if a given (x, y) location is inbounds
    def inbounds(self, pos):
        gpos = convert(self.center, pos, self.nearest)
        if gpos[0] < 0 or gpos[0] > self.dim - 1:
            return False
        if gpos[1] < 0 or gpos[1] > self.dim - 1:
            return False
        return True

# converts (x, y) location to row col location
def convert(center, pos, nearest):
    row, col = center[1] - pos[1], center[0] + pos[0]
    if nearest:
        row, col = np.rint(row), np.rint(col)
    return int(row), int(col)

# finds the tangent angle given two points
def calc_theta(pos, deg=True):
    x, y = pos
    theta =  np.arctan2(y, x)
    if deg:
        return theta * 180 / np.pi
    return theta

# finds (x, y) location given a distance and angle
def calc_point(dist, theta):
    rad = np.pi / 180 * theta
    return np.sin(rad) * dist, np.cos(rad) * dist

# finds the gradient between two points
def calc_grad(start, end):
    if int(start[0]) == int(end[0]) == 0:
        return (end[1],)
    return (end[1] - start[1]) / (end[0] - start[0])

# finds tangent line of point (assumes origin = (0, 0))
def calc_length(pos):
    x, y = pos
    return np.sqrt(x**2 + y**2)

# rotates a point by an angle (assumes origin = (0, 0))
def rotate_pos(pos, theta):
    # get vector length & theta
    dist = calc_length(pos)
    otheta = calc_theta(pos)
    # find new pos
    theta = (360 + otheta + theta) % 360 * np.pi / 180
    x, y = dist * np.cos(theta), dist * np.sin(theta)
    return int(np.rint(x)), int(np.rint(y))


if __name__ == "__main__":
    local_map = LocalMap(11, nearest=False)
    local_map.place((0.5, 0))
    local_map.place((0, 5))
    local_map.raytrace((0, 5))
    local_map.view()
    
    
    



    
