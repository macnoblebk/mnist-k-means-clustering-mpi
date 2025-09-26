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
```
make                    # Build all targets
make sequential         # Build sequential version only
make mpi               # Build MPI version only
make clean             # Clean build artifacts
```

# Usage
## Sequential K-Means
```bash
    ./kmeans_sequential
```

## MPI K-Means
```bash
    mpirun -np 4 ./kmeans_mpi
```

## Program Output
![Screenshot 2025-09-25 at 17.06.42.png](Screenshot%202025-09-25%20at%2017.06.42.png)
![Screenshot 2025-09-25 at 17.05.51.png](Screenshot%202025-09-25%20at%2017.05.51.png)
![Screenshot 2025-09-25 at 17.05.20.png](Screenshot%202025-09-25%20at%2017.05.20.png)
![Screenshot 2025-09-25 at 17.04.39.png](Screenshot%202025-09-25%20at%2017.04.39.png)
![Screenshot 2025-09-25 at 15-11-18 .png](Screenshot%202025-09-25%20at%2015-11-18%20.png)