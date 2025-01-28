import matplotlib.pyplot as plt
import numpy as np
import os

# Check if the file exists
file_path = 'cavity02.mtx'
if not os.path.exists(file_path):
    print(f"File {file_path} not found.")
    exit(1)

try:
    # Read the matrix manually
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip comments and read the header
    lines = [line for line in lines if not line.startswith('%')]
    header = lines[0].strip().split()
    num_rows, num_cols, num_entries = map(int, header)

    # Initialize the matrix
    matrix = np.zeros((num_rows, num_cols))

    # Read the entries
    for line in lines[1:]:
        row, col, value = map(float, line.strip().split())
        matrix[int(row)-1, int(col)-1] = value

    # Print matrix details for debugging
    print(f"Matrix type: {type(matrix)}")
    print(f"Matrix shape: {matrix.shape}")

    # Plot the matrix
    plt.spy(matrix, markersize=1)
    plt.title('Visualization of ' + file_path)
    plt.show()
except Exception as e:
    print(f"An error occurred: {e}")