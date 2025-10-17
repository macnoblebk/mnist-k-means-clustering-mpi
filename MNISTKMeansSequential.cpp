

#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <random>
#include "MNISTKMeans.h"
#include "mpi.h"


// Constants for the K-means algorithm and data loading
const int K = 10;                // Number of clusters
const int ROOT = 0;             // Root process ID
const int IMAGE_LIMIT = 1000;  // Maximum number of images to process

// File paths for MNIST dataset files
const std::string MNIST_IMAGES_FILEPATH = "./emnist-mnist-test-images-idx3-ubyte";
const std::string MNIST_LABELS_FILEPATH = "./emnist-mnist-test-labels-idx1-ubyte";


bool readMNISTImages(MNISTImage** images, int* n) {
    std::ifstream file(MNIST_IMAGES_FILEPATH, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open MNIST images file." << std::endl;
        return false;
    }

    // MNIST header contains magic number and dimensions
    uint32_t magicNumber = 0; // Magic number to verify file format
    uint32_t images_n = 0;    // Number of images in file
    uint32_t rows_n = 0;      // Number of rows per image
    uint32_t cols_n = 0;      // Number of columns per image

    // Read header values
    file.read((char*)&magicNumber, sizeof(magicNumber));
    file.read((char*)&images_n, sizeof(images_n));
    file.read((char*)&rows_n, sizeof(rows_n));
    file.read((char*)&cols_n, sizeof(cols_n));

    // Convert from big-endian to host endianness
    magicNumber = swapEndian(magicNumber);
    images_n = swapEndian(images_n);
    rows_n = swapEndian(rows_n);
    cols_n = swapEndian(cols_n);

    // Verify magic number and dimensions
    if (magicNumber != 2051 || rows_n != MNISTImage::getNumRows() || cols_n != MNISTImage::getNumCols()) {
        std::cerr << "Error: Invalid MNIST image file format." << std::endl;
        return false;
    }

    // Determine how many images to read
    int imagesToRead = std::min(static_cast<int>(images_n), IMAGE_LIMIT);

    // Allocate memory for images and read data
    try {
        MNISTImage* imagesData = new MNISTImage[imagesToRead];

        for (int i = 0; i < imagesToRead; i++) {
            std::array<u_char, MNISTImage::getNumPixels()> imageData;
            file.read(reinterpret_cast<char*>(imageData.data()), MNISTImage::getNumPixels());

            if (file.fail()) {
                delete[] imagesData;
                std::cerr << "Error: Failed to read image data." << std::endl;
                return false;
            }

            imagesData[i] = MNISTImage(imageData); // Create image object from raw data
        }

        // Set output parameters
        *images = imagesData;
        *n = imagesToRead;
        return true;
    } catch (const std::bad_alloc& e) {
        std::cerr << "Error: Memory allocation failed - " << e.what() << std::endl;
        return false;
    }
}




bool readMNISTLabels(u_char** labels, int* n) {
    std::ifstream file(MNIST_LABELS_FILEPATH, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open MNIST labels file." << std::endl;
        return false;
    }

    // MNIST labels header
    uint32_t magicNumber = 0; // Magic number to verify file format
    uint32_t labels_n = 0;    // Number of labels in file

    // Read header values
    file.read((char*)&magicNumber, sizeof(magicNumber));
    file.read((char*)&labels_n, sizeof(labels_n));

    // Convert from big-endian to host endianness
    magicNumber = swapEndian(magicNumber);
    labels_n = swapEndian(labels_n);

    // Verify magic number
    if (magicNumber != 2049) {
        std::cerr << "Error: Invalid MNIST label file format." << std::endl;
        return false;
    }

    // Determine how many labels to read
    int labelsToRead = std::min(static_cast<int>(labels_n), IMAGE_LIMIT);

    // Allocate memory for labels and read data
    try {
        u_char* labelsData = new u_char[labelsToRead];

        for (int i = 0; i < labelsToRead; i++) {
            file.read((char*)&labelsData[i], 1); // Each label is a single byte

            if (file.fail()) {
                delete[] labelsData;
                std::cerr << "Error: Failed to read label data." << std::endl;
                return false;
            }
        }

        // Set output parameters
        *labels = labelsData;
        *n = labelsToRead;
        return true;
    } catch (const std::bad_alloc& e) {
        std::cerr << "Error: Memory allocation failed - " << e.what() << std::endl;
        return false;
    }
}


uint32_t swapEndian(uint32_t value) {
    return ((value & 0xFF) << 24) |
           ((value & 0xFF00) << 8) |
           ((value & 0xFF0000) >> 8) |
           ((value & 0xFF000000) >> 24);
}


double calculateErrorRate(
        const MNISTKMeans<K, MNISTImage::getNumPixels()>::Clusters& clusters,
        const u_char* labels) {

    int totalImages = 0;
    int misclassifiedImages = 0;

    // Process each cluster
    for (size_t clusterIdx = 0; clusterIdx < clusters.size(); clusterIdx++) {
        const auto& cluster = clusters[clusterIdx];

        if (cluster.elements.empty()) continue;

        // Count frequency of each digit (0-9) in this cluster
        std::array<int, 10> digitCounts = {0};
        for (int elemIdx : cluster.elements) {
            int digit = labels[elemIdx];
            digitCounts[digit]++;
        }

        // Find the majority label for this cluster
        int majorityDigit = 0;
        int maxCount = 0;
        for (int digit = 0; digit < 10; digit++) {
            if (digitCounts[digit] > maxCount) {
                maxCount = digitCounts[digit];
                majorityDigit = digit;
            }
        }

        // Count misclassified images (those not matching the majority label)
        for (int elemIdx : cluster.elements) {
            totalImages++;
            if (labels[elemIdx] != majorityDigit) {
                misclassifiedImages++;
            }
        }
    }

    // Calculate error rate
    double errorRate = (double)misclassifiedImages / totalImages;

    // Display results
    std::cout << "\nClustering Error Analysis:" << std::endl;
    std::cout << "Total images: " << totalImages << std::endl;
    std::cout << "Misclassified images: " << misclassifiedImages << std::endl;
    std::cout << "Error rate: " << (errorRate * 100.0) << "%" << std::endl;
    std::cout << "Accuracy: " << ((1.0 - errorRate) * 100.0) << "%" << std::endl;

    return errorRate;
}