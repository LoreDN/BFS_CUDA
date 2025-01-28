# PROJECT GOAL
This Project Goal is to implement and optimise in CUDA programming language, for GPU devices, the Breadth-First Search (BFS) Algorithm, starting from a sequential BFS script wrote in C++ programming language.



# IMPLEMENTATION

### READING THE MATRIX
First of all, in order to read and visualize the matrix 'cavity02.mtx' file, I wrote a phyton script which, given as imput an '.mtx' file, returns a '.png' file representing the matrix.


## BFS ALGORITHM

### FIRST LAYER OF PARALLELISM (BY BLOCK)
The BFS Algortihm analyzes a tree structure, in this case represented by the 'cavity02.mtx' file, seeking one layer at time.
In order to parallelyze this analysis, each vertex of the current layer is assigned to a block of the GPU device.

### SECOND LAYER OF PARALLELISM (BY THREAD)
Once a block is assigned to a vertex, it has to analyze each outgoing edge.
The most efficient way to do so, is by dividing the analyses by thread, in this way each thread analyzes its edge in parallel.
