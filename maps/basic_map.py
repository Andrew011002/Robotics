import numpy as np
import matplotlib.pyplot as plt

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
            # ignore this object
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
                    break
                self.place(fpos, rgb=(255, 255, 255))
        # has gradient
        else:
            for x in self.beam_range(pos[0]):
                fpos = (x, x * grad)
                if not self.free(fpos):
                    break
                self.place(fpos, rgb=(255, 255, 255))
                
    # finds the increments of x or y values for ray tracing
    def beam_range(self, endpoint):
        delta = endpoint / self.resolution
        return np.arange(0, endpoint + 1e-9, delta)

    # place pixel given (x, y) localization 
    def place(self, pos, rgb=(0, 0, 0), obj=False):
        if not self.inbounds(pos):
            raise ValueError(f"can't place pixel at {pos}")
        # place the pixel/object
        gpos = convert(self.center, pos, self.nearest)
        # don't place on robot space
        if gpos == self.center:
            return None
        # don't place where object is located
        if gpos not in self.objects:
            self.grid[gpos[0], gpos[1]] = rgb
        # store object
        if obj:
            self.objects.add(gpos)

    # determines if pixel at (x, y) location is not an rgb color
    def free(self, pos):
        if not self.inbounds(pos):
            return False
        # make sure not occupied in radius
        gpos = convert(self.center, pos, self.nearest)
        ngpos, sgpos, egpos, wgpos, = (gpos[0] - 1, gpos[1]), (gpos[0] + 1, gpos[1]),\
            (gpos[0], gpos[1] + 1), (gpos[0], gpos[1] - 1)
        # occupied (with uncertainties)
        if gpos in self.objects or ngpos in self.objects or sgpos in self.objects\
            or egpos in self.objects or wgpos in self.objects:
            return False
        return True

    # gets both free and occupied pixels
    def pixels(self):
        obj_pixels, free_pixels = [], []
        for row in range(self.dim):
            for col in range(self.dim):
                gpos = row, col
                rgb = tuple(self.grid[gpos[0], gpos[1]].squeeze()) 
                # occupied space
                if rgb == (0, 0, 0):
                    pos = revert(self.center, gpos, self.nearest)
                    obj_pixels.append((pos, rgb))
                # free space
                elif rgb == (255, 255, 255):
                    pos = revert(self.center, gpos, self.nearest)
                    free_pixels.append((pos, rgb))

        # make sure bot is placed last
        bgpos = self.center
        bpos, rgb = (0, 0), tuple(self.grid[bgpos[0], bgpos[1]].squeeze())
        obj_pixels.append((bpos, rgb))
        return obj_pixels, free_pixels


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


class GlobalMap:

    def __init__(self, dim, nearest=False):
        if (dim % 2 == 0):
            raise ValueError("dim must be odd")
        self.grid = np.zeros((dim, dim, 3), dtype=int) + 211
        self.dim = dim
        self.size = (dim, dim)
        self.center = (dim // 2, dim // 2)
        self.nearest = nearest
        self.local_map = None

    # place pixel given (x, y) localization 
    def place(self, pos, rgb=(0, 0, 0)):
        if not self.inbounds(pos):
            raise ValueError(f"can't place pixel at {pos}")
        # place the pixel/object
        gpos = convert(self.center, pos, self.nearest)
        self.grid[gpos[0], gpos[1]] = rgb

    # will place a map with specified rotation if it fits
    def place_map(self, ppos, local_map, rotation, precision=2):
        if not self.map_inbounds(ppos, local_map):
            raise ValueError(f"can't place map at {ppos}")
        # get pixels, rotate them, then place them
        obj_pixels, free_pixels = local_map.pixels()
        # place free spaces first
        for pos, rgb in free_pixels:
            rpos = rotate_pos(pos, rotation + 1, self.nearest, precision)
            tpos = self.translate(ppos, rpos)
            self.place(tpos, rgb)
        # fix holes created
        self.refill(ppos, local_map)
        # place objects next
        for pos, rgb in obj_pixels:
            rpos = rotate_pos(pos, rotation + 1, self.nearest, precision)
            tpos = self.translate(ppos, rpos)
            self.place(tpos, rgb)
        # set local map
        self.local_map = local_map

    # translate local maps (x, y) location to global maps (x, y) location
    def translate(self, ppos, pos):
        tpos = ppos[0] + pos[0], ppos[1] + pos[1]
        return tpos

    # during rotation fixes some issues with pixel holes
    def refill(self, ppos, local_map, min_count=4):
        beam = local_map.beam 
        ggpos = convert(self.center, ppos, self.nearest)
        # gets pixels within border of local_map
        for row in range(ggpos[0] - beam + 1, ggpos[0] + beam):
            for col in range(ggpos[1] - beam + 1, ggpos[1] + beam):
                gpos = (row, col)
                # find bordering pixels
                radi = self.get_radi(gpos)
                # 3 or more are white, so this pixel should be white
                if np.count_nonzero(radi == 255) / 3 >= min_count:
                    self.grid[row, col] = (255, 255, 255)

    # gets radius of a pixel
    def get_radi(self, gpos):
        r, c = gpos
        # indices
        rows, cols = np.array([r - 1, r + 1, r, r]), \
            np.array([c, c, c + 1, c - 1])
        # rgb values of radi
        return self.grid[rows, cols]  

    # shows the global map or global and local map
    def view(self, map=2, cmap="gray"):
        # show noth maps if possible
        if map > 0 and self.local_map is None:
            raise ValueError("no LocalMap for this instance of GlobalMap")
        # show both
        if map == 2:
            plt.subplot(1, 2, 1)
            plt.imshow(self.local_map.grid, cmap=cmap)
            plt.subplot(1, 2, 2)
            plt.imshow(self.grid, cmap=cmap)
        # show local map
        elif map == 1:
            local_map.view(cmap=cmap)
        # show global map
        elif map == 0:
            plt.imshow(self.grid, cmap=cmap)
        else:
            raise ValueError(f"invalid map id ({map}) 0: GlobalMap 1: LocalMap 2: Both")
        plt.show()

        

    # indicates if a given (x, y) location is inbounds
    def inbounds(self, pos):
        gpos = convert(self.center, pos, self.nearest)
        if gpos[0] < 0 or gpos[0] > self.dim - 1:
            return False
        if gpos[1] < 0 or gpos[1] > self.dim - 1:
            return False
        return True

    # determines if a map fits at a placement pos
    def map_inbounds(self, ppos, local_map):
        beam = local_map.beam
        # north south east and west edges
        npos, spos, epos, wpos = (ppos[0], ppos[1] + beam), (ppos[0], ppos[1] - beam),\
            (ppos[0] + beam, ppos[1]), (ppos[0] - beam, ppos[1])
        return self.inbounds(npos) and self.inbounds(spos) and self.inbounds(epos) and self.inbounds(wpos)

# converts (x, y) location to row col location
def convert(center, pos, nearest):
    row, col = center[1] - pos[1], center[0] + pos[0]
    if nearest:
        row, col = np.rint(row), np.rint(col)
    return int(row), int(col)

# reverts row col location to (x, y) location
def revert(center, gpos, nearest):
    x, y = gpos[1] - center[1], center[0] - gpos[0]
    if nearest:
        x, y = np.rint(x), np.rint(y)
    return int(x), int(y)

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
def rotate_pos(pos, theta, nearest=False, percision=2):    
    # get vector length & theta
    dist = calc_length(pos)
    otheta = calc_theta(pos)
    # find new pos
    theta = (360 + otheta + theta) % 360 * np.pi / 180
    # did not change rotation
    if round(theta * 180 / np.pi, percision) == round(otheta, 2):
        return pos
    # changed rotation, find new pos
    x, y = dist * np.cos(theta), dist * np.sin(theta)
    if nearest:
        x, y = np.rint(x), np.rint(y)
    return int(x), int(y)

if __name__ == "__main__":
    min_dist = 4
    max_dist = 12
    count = 5
    local_map = LocalMap(25, 360)
    objects = [(np.random.randint(min_dist, max_dist + 1), np.random.randint(0, 361))\
        for _ in range(count)]

    local_map.place_objs(objects)
    local_map.raytrace()
    global_map = GlobalMap(101)
    ppos = (0, 0)
    # global_map.place_map(ppos, local_map, 90)
    global_map.view(map=3)
    
        
        
    
    
    



    
