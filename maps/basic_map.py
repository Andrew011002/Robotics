import numpy as np
import matplotlib.pyplot as plt




class LocalMap:
    
    def __init__(self, dim, fov, nearest=False):
        if (dim % 2 == 0):
            raise ValueError("dim must be odd")
        self.dim = dim
        self.beam = dim // 2
        self.size = (dim, dim)
        self.center = (dim // 2, dim // 2)
        self.fov = fov
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
    def raytrace(self, dtheta=0.25):
        
        # locate objs in fov
        for theta in np.arange(-self.fov // 2, self.fov // 2 + 1e-9, step=dtheta):
            pos = calc_point(self.beam, theta)
            grad = calc_grad((0, 0), pos)
            self.trace(pos, grad)

    def trace(self, pos, grad, delta=0.25):

        beam = self.beam
    
        # vertical line
        if isinstance(grad, tuple):
            ystart = 1
            if grad[0] < 0:
                ystart *= -1
                beam *= -1
                delta *= -1
            
            for y in np.arange(ystart, beam, delta):
                fpos = (0, y)
                if self.free(fpos):
                    self.place(fpos, rgb=(255, 255, 255))
                else:
                    break

        else:
            xstart = 1
            if pos[0] < 0:
                xstart *= -1
                beam *= -1
                delta *= -1

            for x in np.arange(xstart, beam, delta):
                fpos = (x, grad * x)
                if self.free(fpos):
                    self.place(fpos, rgb=(255, 255, 255))
                else:
                    break

    # place pixel given (x, y) localization 
    def place(self, pos, rgb=(0, 0, 0), obj=False):
        if not self.inbounds(pos):
            raise ValueError(f"can't place pixel at {pos}")
        # place the pixel/object
        gpos = convert(self.center, pos, self.nearest)
        if gpos not in self.objects:
            self.grid[gpos[0], gpos[1]] = rgb
        if obj:
            self.objects.add(gpos)

    # determines if pixel at (x, y) location is not an rgb color
    def free(self, pos, rgb=(0, 0, 0)):
        if not self.inbounds(pos):
            return False
        # see if on robot space or in occupied cell
        gpos = convert(self.center, pos, self.nearest)
        pixel = self.grid[gpos[0], gpos[1]].squeeze()
        if gpos == self.center or gpos in self.objects or tuple(pixel) == rgb:
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
    if start[0] == end[0] == 0:
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
    local_map = LocalMap(101, 180, nearest=True)
    objects = [(5, 0), (15, 21), (21, -90)]
    local_map.place_objs(objects)
    local_map.raytrace()
    local_map.view()
        
    
    
    



    
