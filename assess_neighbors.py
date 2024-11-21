import numpy as np


def average_hv_difference(grid_data, rows, cols, row, col):

    # Ensure the indices are within bounds
    if not (0 <= row < rows) or not (0 <= col < cols):
        raise ValueError("Row or column index is out of bounds.")

    target = grid_data[row, col]

    neighbors = []
    if row > 0:
        neighbors.append(grid_data[row - 1, col])
    if row < rows - 1:
        neighbors.append(grid_data[row + 1, col])
    if col > 0:
        neighbors.append(grid_data[row, col - 1])
    if col < cols - 1:
        neighbors.append(grid_data[row, col + 1])

    # Calculate average difference
    differences = [np.linalg.norm(target - neighbor) for neighbor in neighbors]
    average_difference = np.mean(differences) if differences else 0.0

    return average_difference


def average_diag_difference(grid_data, ROWS, COLS, row, col):

    if not (0 <= row < ROWS) or not (0 <= col < COLS):
        raise ValueError("Row or column index is out of bounds.")

    target = grid_data[row, col]

    neighbors = []
    if row > 0 and col > 0:
        neighbors.append(grid_data[row - 1, col - 1])
    if row > 0 and col < COLS - 1:
        neighbors.append(grid_data[row - 1, col + 1])
    if row < ROWS - 1 and col > 0:
        neighbors.append(grid_data[row + 1, col - 1])
    if row < ROWS - 1 and col < COLS - 1:
        neighbors.append(grid_data[row + 1, col + 1])

    differences = [np.linalg.norm(target - neighbor) for neighbor in neighbors]
    average_difference = np.mean(differences) if differences else 0.0

    return average_difference