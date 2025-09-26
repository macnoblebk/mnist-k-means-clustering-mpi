# Color K-Means Clustering
A C++ implementation of k-means clustering algorithm with both sequential and MPI parallel versions, specifically 
designed for clustering RGB color data. In this project, I extend the provided sequential k-means implementation with 
MPI parallelization.

# Features 
- Sequential K-Means: Standard k-means clustering implementation
- MPI Parallel K-Means: Distributed k-means using Message Passing Interface
- Color-Specific Operations: RGB color handling with Euclidean distance in color space
- X11 Color Dataset: Built-in dataset of 140+ named colors
- HTML Visualization: Generate visual output for web browser display
- Template Design: Generic implementation supporting different cluster counts and dimensionalities

# Project Structure
```
├── Color.h                 # Color class declaration (provided)
├── Color.cpp               # Color class implementation (provided)
├── KMeans.h                # Sequential k-means template class (provided)
├── KMeansMPI.h             # MPI parallel k-means template class (my implementation)
├── ColorKMeans.h           # Color-specific sequential k-means (provided)
├── ColorKMeansMPI.h        # Color-specific MPI k-means (provided)
├── kmean_color_test.cpp    # Sequential version test program (provided)
└── hw5.cpp                 # MPI version test program (provided)

```

# Compilation
## Using Makefile (Recommended)
```bash
make                    # Build both sequential and MPI versions
make kmean_color_test   # Build sequential version only
make hw5                # Build MPI version only
make clean              # Clean build artifacts
```

# Run Programs
```bash
make run_sequential     # Run sequential version
make run_hw5            # Run MPI version with 2 processes
make bigger_test        # Run MPI version with 10 processes
make valgrind           # Run MPI version with valgrind memory checking

```

# Usage
## Sequential K-Means
```bash
./kmeans_sequential
```

## MPI K-Means
```bash
    mpirun -np # ./kmeans_mpi
```
where ```#``` is the number of parallel processes to execute (e.g. ```mpirun -np 32 ./kmeans_mpi```)

## Program Output
![Screenshot 2025-09-25 at 17.06.42.png](Screenshot%202025-09-25%20at%2017.06.42.png)
![Screenshot 2025-09-25 at 17.05.51.png](Screenshot%202025-09-25%20at%2017.05.51.png)
![Screenshot 2025-09-25 at 17.05.20.png](Screenshot%202025-09-25%20at%2017.05.20.png)
![Screenshot 2025-09-25 at 17.04.39.png](Screenshot%202025-09-25%20at%2017.04.39.png)
![Screenshot 2025-09-25 at 15-11-18 .png](Screenshot%202025-09-25%20at%2015-11-18%20.png)