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
        pos_ = convert(self.origin, pos)
        self.grid[pos_[0], pos_[1], :] = rgb
        if object:
            self.objects.append((pos, rgb))

    def get_objs(self):
        return self.objects

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

    def place_local_map(self, local_map, angle):
        objs = local_map.get_objs()
        for pos, rgb in objs:
            pos_ = convert(self.origin, pos)
            self.place_obj(pos_, rgb)

    def place_obj(self, pos_, rgb=(0, 0, 0)):
        self.grid[pos_[0], pos_[1], :] = rgb
        self.objects.append((pos_, rgb))

    def get_objs(self):
        return self.objects

    def show(self, cmap="gray"):
        plt.imshow(self.grid, cmap=cmap)
        plt.show()

    def __str__(self):
        return str(self.grid)


def convert(origin, pos):
    return origin[0] - pos[1], origin[1] + pos[0]

def revert(origin, indices):
    return indices[1] - origin[1], origin[0] - indices[0]

if __name__ == "__main__":
    local_map = LocalMap(25, 25, (0, 0))
    local_map.place_obj((-3, 3))
    gloabl_map = GlobalMap(51, 51, (18, 18))
    gloabl_map.place_local_map(local_map, None)
    gloabl_map.show()
    
