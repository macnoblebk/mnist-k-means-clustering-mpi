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


};