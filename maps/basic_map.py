import numpy as np
import matplotlib.pyplot as plt

class GlobalMap:

    def __init__(self, dim):
        if (dim % 2 == 0):
            raise ValueError("dim must be odd")
        self.grid = np.zeros((dim, dim, 3), dtype=int) + 211
        self.dim = dim
        self.size = (dim, dim)
        self.center = (dim // 2, dim // 2)

    def place_map(self, pixels, pos, agnle):
        pass

    def view(self, cmap="gray"):
        plt.imshow(self.grid, cmap=cmap)
        plt.show()


class LocalMap:
    
    def __init__(self, dim, fov, resolution=100, nearest=False):
        if (dim % 2 == 0):
            raise ValueError("dim must be odd")
        self.dim = dim
        self.beam = dim // 2
        self.size = (dim, dim)
        self.center = (dim // 2, dim // 2)
        self.fov = fov
        self.resolution = resolution
        self.nearest = nearest
        self.grid = np.zeros((dim, dim, 3), dtype=int) + 211
        self.grid[self.center[1], self.center[0]] = (0, 255, 0) 
        self.objects = set()
            
    def place_objs(self, objs):
        # (dist, angle)
        for dist, theta in objs:
            pos = calc_point(dist, theta)
            if not self.free(pos):
                continue
            self.place(pos, obj=True)

    # finds ray given a position
    def raytrace(self, dtheta=0.1):
        origin = (0, 0) # origin of ray casting
        
        # locate objs in fov
        for theta in np.arange(-self.fov // 2, self.fov // 2 + 1e-9, step=dtheta):
            pos = calc_point(self.beam, theta)
            grad = calc_grad(origin, pos)
            self.trace(pos, grad)

    def trace(self, pos, grad):
        
        # no gradient
        if grad is None:
            for y in self.beam_range(pos[1]):
                fpos = (0, y)
                if not self.free(fpos):
                    return None
                self.place(fpos, rgb=(255, 255, 255))
        # has gradient
        else:
            for x in self.beam_range(pos[0]):
                fpos = (x, x * grad)
                if not self.free(fpos):
                    return None
                self.place(fpos, rgb=(255, 255, 255))
                
    # finds the increments of x or y values for ray tracing
    def beam_range(self, endpoint):
        delta = endpoint / self.resolution
        return np.arange(0, endpoint, delta)

    # place pixel given (x, y) localization 
    def place(self, pos, rgb=(0, 0, 0), obj=False):
        if not self.inbounds(pos):
            raise ValueError(f"can't place pixel at {pos}")
        # place the pixel/object
        gpos = convert(self.center, pos, self.nearest)
        # don't place on robot
        if gpos == self.center:
            return None
        if gpos not in self.objects:
            self.grid[gpos[0], gpos[1]] = rgb
        if obj:
            self.objects.add(gpos)

    # determines if pixel at (x, y) location is not an rgb color
    def free(self, pos):
        if not self.inbounds(pos):
            return False
        # see if on robot space or in occupied cell
        gpos = convert(self.center, pos, self.nearest)
        ngpos, sgpos, egpos, wgpos, = (gpos[0] - 1, gpos[1]), (gpos[0] + 1, gpos[1]),\
            (gpos[0], gpos[1] + 1), (gpos[0], gpos[1] - 1)
        # occupied (with uncertainties)
        if gpos in self.objects or ngpos in self.objects or sgpos in self.objects\
            or egpos in self.objects or wgpos in self.objects:
            return False
        return True

    def pixels(self):
        pass

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

# reverts row col location to (x, y) location
def revert(center, gpos, nearest):
    pass

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
    if start[0] == end[0] == 0:
        return None
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
    local_map = LocalMap(25, fov=360, resolution=100, nearest=False)
    objects = [(np.random.randint(5, 13), np.random.randint(0, 361)) for _ in range(10)]
    local_map.place_objs(objects)
    # local_map.raytrace()
    # local_map.view()
    global_map = GlobalMap(5)
    global_map.view()
        
        
    
    
    



    
