import numpy as np
import matplotlib.pyplot as plt

grid = np.zeros((11, 11, 3), dtype=int) + 211

def convert(center, pos, nearest=False):
    row, col = center[1] - pos[1], center[0] + pos[0]
    if nearest:
        row, col = np.rint(row), np.rint(col)
    return int(row), int(col)

def place(grid, gpos, rgb=(255, 255, 255)):
    grid[gpos[0], gpos[1], :] = rgb

center = (5, 5)
free_spaces = [(0, 1), (0, -1), (1, 0), (-1, 0)]
rows = []
cols = []
for pos in free_spaces:
    gpos = convert(center, pos)
    place(grid, gpos)
    rows.append(gpos[0])
    cols.append(gpos[1])
rows.append(5)
cols.append(5)
rows = np.array(rows, int)
cols = np.array(cols, int)

total = 0

for r in range(1, 11 - 1):
    for c in range(1, 11 - 1):
        if tuple(grid[r, c]) == (211, 211, 211):
            if tuple(grid[r - 1, c]) == (255, 255, 255):
                total += 1
            if tuple(grid[r + 1, c]) == (255, 255, 255):
                total += 1
            if tuple(grid[r, c + 1]) == (255, 255, 255):
                total += 1
            if tuple(grid[r, c - 1]) == (255, 255, 255):
                total += 1
        if total >= 3:
            grid[r, c] = (255, 255, 255)
        total = 0


plt.imshow(grid, cmap="gray")
plt.show()
