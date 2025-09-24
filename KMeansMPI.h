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

    struct Cluster {
        Element centroid;                // Center (mean) of the elements in the cluster
        std::vector<int> elements;       // Indices of elements belonging to this cluster

        friend bool operator==(const Cluster& left, const Cluster& right) {
            return left.centroid == right.centroid;
        }
    };

    using Clusters = std::array<Cluster, k>;

    KMeansMPI() : isVerboseMode(false) {}

    void setVerboseMode(bool verbose) {
        isVerboseMode = verbose;
    }

    virtual const Clusters& getClusters() {
        return clusters;
    }

    virtual void fit(const Element* data, int nData) {
        elements = data;
        totalElements = nData;
        fitWork(ROOT_PROCESS);
    }

};