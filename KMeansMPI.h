/**
 * @file KMeansMPI.h
 * @author Mac-Noble Brako-Kusi
 * @date 03-07-2025
 * @see "Seattle University, CPSC5600, Winter 2025"
 */


template <int k, int d>
class KMeansMPI {
public:
    // Define common types and constants
    using Element = std::array<u_char, d>;
    using DistanceMatrix = std::vector<std::array<double, k>>;

    static const int MAX_FIT_STEPS = 300;  // Maximum number of iterations before giving up
    static const int ROOT_PROCESS = 0;     // Rank of the root/coordinator process

    /**
     * @brief Equality operator for clusters
     *
     * Compares only centroids, regardless of the elements in the cluster.
     *
     * @param left Left operand
     * @param right Right operand
     * @return true if centroids are identical, false otherwise
     */
    struct Cluster {
        Element centroid;                // Center (mean) of the elements in the cluster
        std::vector<int> elements;       // Indices of elements belonging to this cluster

        friend bool operator==(const Cluster& left, const Cluster& right) {
            return left.centroid == right.centroid;
        }
    };

    using Clusters = std::array<Cluster, k>;

  /**
   * @brief Constructor
   */
    KMeansMPI() : isVerboseMode(false) {}

/**
 * @brief Enable or disable verbose debug output
 * @param verbose True to enable verbose output, false to disable
 */
    void setVerboseMode(bool verbose) {
        isVerboseMode = verbose;
    }

  /**
   * @brief Expose the clusters to the client readonly
   *
   * Provides access to the final clustering results after fit() has been called.
   *
   * @return Reference to the array of clusters from the latest call to fit()
   */
    virtual const Clusters& getClusters() {
        return clusters;
    }

    /**
     * @brief Main k-means clustering algorithm entry point
     *
     * This method is called by the ROOT process only. It initializes the data
     * and calls fitWork() to perform the clustering in parallel.
     *
     * @param data Pointer to the array of data elements for k-means
     * @param nData The number of data elements in the array
     */
    virtual void fit(const Element* data, int nData) {
        elements = data;
        totalElements = nData;
        fitWork(ROOT_PROCESS);
    }

    /**
     * @brief Per-process work for fitting the k-means model
     *
     * This is the main worker method for all processes.
     *
     * @param rank Process rank within MPI_COMM_WORLD
     */
    virtual void fitWork(int rank) {
        initializeClusteringProcess(rank);
        runClusteringIterations(rank);
        finalizeClusteringProcess(rank);
    }

protected:
    // Data members
    const Element* elements = nullptr;     // Set of elements to classify
    Element* partition = nullptr;          // Partition of elements for the current process
    int* elementIds = nullptr;             // Locally track original indices in elements
    int totalElements = 0;                 // Total number of elements in the dataset
    int localElements = 0;                 // Number of elements in this process's partition
    int numProcesses = 0;                  // Number of processes in MPI_COMM_WORLD
    Clusters clusters;                     // k clusters resulting from latest call to fit()
    DistanceMatrix dist;                   // Distance matrix for local elements to centroids
    bool isVerboseMode;                    // Flag for verbose debug output

    template<typename... Args>
    void logDebug(Args&&... args) {
        if (isVerboseMode) {
            using namespace std;
            (cout << ... << args);
        }
    }

    void initializeClusteringProcess(int rank) {
        broadcastDatasetSize();
        partitionDatasetElements(rank);

        if (rank == ROOT_PROCESS) {
            initializeClusterCentroids();
        }

        broadcastClusterCentroids(rank);
    }

    void runClusteringIterations(int rank) {
        Clusters previousClusters = clusters;
        // Make initial state different to ensure at least one iteration
        previousClusters[0].centroid[0]++;

        for (int iteration = 0; iteration < MAX_FIT_STEPS; iteration++) {
            // Check for convergence
            if (previousClusters == clusters && iteration > 0) {
                logDebug(rank, " converged at iteration ", iteration, "\n");
                break;
            }

            logDebug(rank, " working on iteration ", iteration, "\n");

            // Perform one iteration of the clustering algorithm
            calculateElementDistances();
            previousClusters = clusters;
            updateLocalClusters();
            mergeClusterResults(rank);
            broadcastClusterCentroids(rank);
        }
    }

    void finalizeClusteringProcess(int rank) {
        gatherClusterAssignments(rank);

        // Clean up local data
        delete[] partition;
        delete[] elementIds;
        partition = nullptr;
        elementIds = nullptr;
    }

    void broadcastDatasetSize() {
        MPI_Bcast(&totalElements, 1, MPI_INT, ROOT_PROCESS, MPI_COMM_WORLD);
    }


    /**
     * @brief Distribute elements among all processes
     * @param rank The ID of the current process
     */
    void partitionDatasetElements(int rank) {
        MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
        u_char* sendbuf = nullptr, *recvbuf = nullptr;
        int* sendcounts = nullptr, *displs = nullptr;
        int elementsPerProcess = totalElements / numProcesses;

        // Calculate local partition size
        localElements = elementsPerProcess;
        if (rank == numProcesses - 1) {
            localElements = totalElements - (elementsPerProcess * (numProcesses - 1));
        }

        // Prepare distance matrix for local elements
        dist.resize(localElements);

        // Marshall data on root process
        if (rank == ROOT_PROCESS) {
            marshallElementData(&sendbuf, &sendcounts, &displs, elementsPerProcess);
        }

        // Prepare receive buffer
        int recvcount = localElements * (d + 1);
        recvbuf = new u_char[recvcount];

        // Scatter element data to all processes
        scatterElementData(sendbuf, sendcounts, displs, recvbuf, recvcount, rank);

        // Unmarshall received data
        unmarshallElementData(recvbuf);

        // Clean up
        delete[] sendbuf;
        delete[] recvbuf;
        delete[] sendcounts;
        delete[] displs;
    }

    void marshallElementData(u_char** sendbuf, int** sendcounts, int** displs, int elementsPerProcess) {
        *sendbuf = new u_char[totalElements * (d + 1)];
        *sendcounts = new int[numProcesses];
        *displs = new int[numProcesses];

        // Pack element data and indexes
        int bufIndex = 0;
        for (int elemIndex = 0; elemIndex < totalElements; elemIndex++) {
            // Pack element data
            for (int dimIndex = 0; dimIndex < d; dimIndex++) {
                (*sendbuf)[bufIndex++] = elements[elemIndex][dimIndex];
            }
            // Pack element index
            (*sendbuf)[bufIndex++] = (u_char)elemIndex;
        }

        // Set up send counts and displacements
        for (int procIndex = 0; procIndex < numProcesses; procIndex++) {
            (*displs)[procIndex] = procIndex * elementsPerProcess * (d + 1);
            (*sendcounts)[procIndex] = elementsPerProcess * (d + 1);

            // Last process gets any remaining elements
            if (procIndex == numProcesses - 1) {
                (*sendcounts)[procIndex] = bufIndex - ((numProcesses - 1) * elementsPerProcess * (d + 1));
            }
        }
    }

    void scatterElementData(u_char* sendbuf, int* sendcounts, int* displs,
                            u_char* recvbuf, int recvcount, int rank) {
        MPI_Scatterv(
                sendbuf, sendcounts, displs, MPI_UNSIGNED_CHAR,
                recvbuf, recvcount, MPI_UNSIGNED_CHAR,
                ROOT_PROCESS, MPI_COMM_WORLD
        );
    }

    void unmarshallElementData(u_char* recvbuf) {
        partition = new Element[localElements];
        elementIds = new int[localElements];

        int bufIndex = 0;
        for (int elemIndex = 0; elemIndex < localElements; elemIndex++) {
            // Unpack element data
            for (int dimIndex = 0; dimIndex < d; dimIndex++) {
                partition[elemIndex][dimIndex] = recvbuf[bufIndex++];
            }
            // Unpack element index
            elementIds[elemIndex] = (int)recvbuf[bufIndex++];
        }
    }

    void initializeClusterCentroids() {
        std::vector<int> seeds;
        std::vector<int> candidates(totalElements);
        std::iota(candidates.begin(), candidates.end(), 0);

        // Randomly select k elements as initial centroids
        auto random = std::mt19937{std::random_device{}()};
        std::sample(candidates.begin(), candidates.end(), back_inserter(seeds), k, random);

        // Set up initial clusters
        for (int i = 0; i < k; i++) {
            clusters[i].centroid = elements[seeds[i]];
            clusters[i].elements.clear();
        }
    }

    void broadcastClusterCentroids(int rank) {
        logDebug(rank, " broadcasting centroids\n");

        int count = k * d;
        u_char* buffer = new u_char[count];

        // Marshall centroids on root
        if (rank == ROOT_PROCESS) {
            marshallCentroids(buffer);
            logDebug(rank, " sending centroids\n");
        }

        // Broadcast to all processes
        MPI_Bcast(buffer, count, MPI_UNSIGNED_CHAR, ROOT_PROCESS, MPI_COMM_WORLD);

        // Unmarshall centroids on non-root processes
        if (rank != ROOT_PROCESS) {
            unmarshallCentroids(buffer);
            logDebug(rank, " received centroids\n");
        }

        delete[] buffer;
    }

    void marshallCentroids(u_char* buffer) {
        int bufIndex = 0;
        for (int clusterIndex = 0; clusterIndex < k; clusterIndex++) {
            for (int dimIndex = 0; dimIndex < d; dimIndex++) {
                buffer[bufIndex++] = clusters[clusterIndex].centroid[dimIndex];
            }
        }
    }

    void unmarshallCentroids(const u_char* buffer) {
        int bufIndex = 0;
        for (int clusterIndex = 0; clusterIndex < k; clusterIndex++) {
            for (int dimIndex = 0; dimIndex < d; dimIndex++) {
                clusters[clusterIndex].centroid[dimIndex] = buffer[bufIndex++];
            }
        }
    }

    void calculateElementDistances() {
        for (int i = 0; i < localElements; i++) {
            logDebug("Calculating distances for element ", i, ": ");

            for (int j = 0; j < k; j++) {
                dist[i][j] = distance(clusters[j].centroid, partition[i]);
                logDebug(dist[i][j], " ");
            }

            logDebug("\n");
        }
    }

    void updateLocalClusters() {
        // Reset all clusters
        for (int j = 0; j < k; j++) {
            clusters[j].centroid = Element{};
            clusters[j].elements.clear();
        }

        // Assign each element to its closest cluster
        for (int i = 0; i < localElements; i++) {
            // Find closest centroid
            int closestCluster = findClosestCluster(i);

            // Update centroid with this element
            updateClusterCentroid(closestCluster, i);

            // Add element to the cluster
            clusters[closestCluster].elements.push_back(i);
        }
    }

    int findClosestCluster(int elementIndex) {
        int closestCluster = 0;
        for (int j = 1; j < k; j++) {
            if (dist[elementIndex][j] < dist[elementIndex][closestCluster]) {
                closestCluster = j;
            }
        }
        return closestCluster;
    }

    void updateClusterCentroid(int clusterIndex, int elementIndex) {
        accum(
                clusters[clusterIndex].centroid,
                clusters[clusterIndex].elements.size(),
                partition[elementIndex],
                1
        );
    }

    void mergeClusterResults(int rank) {
        int sendCount = k * (d + 1);
        int recvCount = numProcesses * sendCount;
        u_char* sendbuf = new u_char[sendCount];
        u_char* recvbuf = nullptr;

        // Marshall local cluster data
        marshallLocalClusters(sendbuf);

        // Gather all cluster data to root
        if (rank == ROOT_PROCESS) {
            recvbuf = new u_char[recvCount];
        }

        MPI_Gather(
                sendbuf, sendCount, MPI_UNSIGNED_CHAR,
                recvbuf, sendCount, MPI_UNSIGNED_CHAR,
                ROOT_PROCESS, MPI_COMM_WORLD
        );

        // On root, merge cluster results
        if (rank == ROOT_PROCESS) {
            mergeClustersOnRoot(recvbuf);
            delete[] recvbuf;
        }

        delete[] sendbuf;
    }

    void marshallLocalClusters(u_char* buffer) {
        int bufIndex = 0;
        for (int clusterIndex = 0; clusterIndex < k; clusterIndex++) {
            // Pack centroid
            for (int dimIndex = 0; dimIndex < d; dimIndex++) {
                buffer[bufIndex++] = clusters[clusterIndex].centroid[dimIndex];
            }
            // Pack cluster size
            buffer[bufIndex++] = (u_char)clusters[clusterIndex].elements.size();
        }
    }

    void mergeClustersOnRoot(u_char* recvbuf) {
        // Track sizes for proper averaging
        std::array<int, k> clusterSizes;
        for (int clusterIndex = 0; clusterIndex < k; clusterIndex++) {
            clusterSizes[clusterIndex] = clusters[clusterIndex].elements.size();
        }

        // Process data from each process
        int bufIndex = 0;
        for (int procIndex = 0; procIndex < numProcesses; procIndex++) {
            for (int clusterIndex = 0; clusterIndex < k; clusterIndex++) {
                // Extract centroid
                Element centroid = Element{};
                for (int dimIndex = 0; dimIndex < d; dimIndex++) {
                    centroid[dimIndex] = recvbuf[bufIndex++];
                }

                // Extract cluster size
                int size = (int)recvbuf[bufIndex++];

                // Average with existing centroid
                if (size > 0) {
                    accum(
                            clusters[clusterIndex].centroid,
                            clusterSizes[clusterIndex],
                            centroid, size
                    );
                    clusterSizes[clusterIndex] += size;
                }
            }
        }
    }

    void gatherClusterAssignments(int rank) {
        // Count how many bytes we need to send
        int elementsToSend = 0;
        for (const auto& cluster : clusters) {
            elementsToSend += cluster.elements.size();
        }

        int sendcount = elementsToSend + k;  // Add k for cluster sizes
        u_char* sendbuf = new u_char[sendcount];
        u_char* recvbuf = nullptr;
        int* recvcounts = nullptr;
        int* displs = nullptr;

        // Marshall cluster assignments
        marshallClusterAssignments(sendbuf);

        // Set up receive parameters on root
        if (rank == ROOT_PROCESS) {
            prepareClusterGatherBuffers(&recvbuf, &recvcounts, &displs);
        }

        // Gather all assignments to root
        MPI_Gatherv(
                sendbuf, sendcount, MPI_UNSIGNED_CHAR,
                recvbuf, recvcounts, displs, MPI_UNSIGNED_CHAR,
                ROOT_PROCESS, MPI_COMM_WORLD
        );

        // Process gathered assignments on root
        if (rank == ROOT_PROCESS) {
            processGatheredAssignments(recvbuf);
            delete[] recvbuf;
            delete[] recvcounts;
            delete[] displs;
        }

        delete[] sendbuf;
    }

    void marshallClusterAssignments(u_char* buffer) {
        int bufIndex = 0;

        for (int clusterIndex = 0; clusterIndex < k; clusterIndex++) {
            // Store cluster size
            buffer[bufIndex++] = (u_char)clusters[clusterIndex].elements.size();

            // Store element IDs
            for (int& elemIndex : clusters[clusterIndex].elements) {
                buffer[bufIndex++] = (u_char)elementIds[elemIndex];
            }
        }
    }

    virtual double distance(const Element& a, const Element& b) const = 0;
};