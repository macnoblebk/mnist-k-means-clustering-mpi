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

};