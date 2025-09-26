/**
 * @file MNISTKMeansMPI.h
 * @author Mac-Noble Brako-Kusi
 * @date 03-07-2025
 * @see "Seattle University, CPSC5600, Winter 2025"
 */

#pragma once
#include "KMeansMPI.h"
#include "MNISTImage.h"

/**
 * @class MNISTKMeansMPI
 * @brief Concrete MNIST k-means class with MPI parallelization
 * @tparam k the number of clusters for k-means (typically 10 for digits 0-9)
 * @tparam d the dimensionality of an MNIST image (typically 784 for 28x28 pixels)
 *
 * This class extends the generic parallel K-means implementation to work specifically
 * with MNIST image data. It distributes the workload across multiple processes
 * using MPI for improved performance on large datasets.
 */
 template<int k, int d>
class MNISTKMeansMPI : public KMeansMPI<k, d> {
public:
    /**
    * @brief Run k-means clustering on MNIST images in parallel
    *
    * This method takes an array of MNIST images and adapts it to the format
    * expected by the base KMeansMPI class. It distributes the workload across
    * multiple processes using MPI for faster clustering.
    *
    * @param images pointer to the MNIST image data
    * @param n the number of images to cluster
    */
    void fit(MNISTImage* images, int n) {
        // Reinterpret the MNISTImage array as array of byte arrays
        KMeansMPI<k, d>::fit(reinterpret_cast<std::array<u_char, d>*>(images), n);
    }

protected:
    using Element = std::array<u_char, d>; // Type alias for the element type used in the KMeansMPI base class

    /**
     * @brief Calculate the Euclidean distance between two MNIST images
     * This method overrides the abstract distance method from the KMeansMPI base class
     * to provide a specific implementation for MNIST images. It converts the raw
     * element data to MNISTImage objects and uses their euclideanDistance method.
     * @param a one MNIST image represented as an array of bytes
     * @param b another MNIST image represented as an array of bytes
     * @return Euclidean distance between the two images
     */
    double distance(const Element& a, const Element& b) const {
        // Convert raw element data to MNISTImage objects to use their distance method
        return MNISTImage(a).euclideanDistance(MNISTImage(b));
    }
};