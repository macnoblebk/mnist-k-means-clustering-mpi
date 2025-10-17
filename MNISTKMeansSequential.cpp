

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


int main(void) {
    // Pointers to store the loaded image and label data
    MNISTImage* images = nullptr;
    u_char* labels = nullptr;

    // Initialize the k-means algorithm with 10 clusters and 784 dimensions (28x28 pixels)
    MNISTKMeans<K, MNISTImage::getNumPixels()> kMeans;

    // Load the MNIST dataset
    int images_n;  // Number of images loaded
    int labels_n;  // Number of labels loaded

    // Load image data with error checking
    if (!readMNISTImages(&images, &images_n)) {
        std::cerr << "Failed to read MNIST images" << std::endl;
        return 1;
    }

    // Load label data with error checking
    if (!readMNISTLabels(&labels, &labels_n)) {
        std::cerr << "Failed to read MNIST labels" << std::endl;
        delete[] images; // Clean up already allocated image data
        return 1;
    }

    // Verify data alignment
    if (images_n != labels_n) {
        std::cerr << "Error: Number of images (" << images_n << ") doesn't match number of labels ("
                  << labels_n << ")" << std::endl;
        delete[] images;
        delete[] labels;
        return 1;
    }

    // Run the k-means clustering algorithm on the loaded images
    kMeans.fit(images, images_n);

    // Get the resulting clusters after algorithm convergence
    MNISTKMeans<K, MNISTImage::getNumPixels()>::Clusters clusters = kMeans.getClusters();

    // Output the clustering results to console and generate HTML visualization
    printClusters(clusters, labels);
    calculateErrorRate(clusters, labels);

    // Generate HTML visualization
    std::string filename = "kmeans_mnist_seq.html";
    toHTML(clusters, images, filename);
    std::cout << "\nTry displaying visualization file, " << filename << ", in a web browser!\n";

    // Clean up allocated memory
    delete[] images;
    delete[] labels;
    return 0;
}


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


void printClusters(
        const MNISTKMeans<K, MNISTImage::getNumPixels()>::Clusters& clusters,
        const u_char* labels
) {
    std::cout << "\nMNIST labels report: showing clusters...\n";

    // Iterate through each cluster
    for (size_t i = 0; i < clusters.size(); i++) {
        std::cout << "\ncluster #" << i + 1 << ":\n";

        // Print all label values for elements in this cluster
        for (int j: clusters[i].elements)
            std::cout << (int)labels[j] << " ";

        std::cout << std::endl;
    }
}

std::string htmlRandomBackground() {
    // Use static random generator
    static std::mt19937 rng(std::random_device{}());

    // Use a distribution that generates lighter colors for better readability
    static std::uniform_int_distribution<> distrib(200, 240);

    // Generate pastel color
    int r = distrib(rng);
    int g = distrib(rng);
    int b = distrib(rng);

    // Format as a hex color string
    char buffer[7];
    snprintf(buffer, sizeof(buffer), "%.6x", (r << 16) | (g << 8) | b);

    return {buffer};
}

void htmlCell(std::ofstream& f, const MNISTImage& image) {
    f << "<table style=\"border-collapse:collapse\">\n";

    // For each pixel row
    for (int row = 0; row < MNISTImage::getNumRows(); row++) {
        f << "<tr>\n";

        // For each pixel column
        for (int col = 0; col < MNISTImage::getNumCols(); col++) {
            // Create a cell with background color based on pixel value
            f << "<td class=\"pixel\" style=\"background:#"
              << image.pixelToHex(row, col) << ";\"></td>\n";
        }

        f << "</tr>\n";
    }

    f << "</table>\n";
}

void toHTML(
        const MNISTKMeans<K, MNISTImage::getNumPixels()>::Clusters& clusters,
        const MNISTImage* images,
        const std::string& filename
) {
    // Open output file with error checking
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error: Unable to create HTML output file: " << filename << std::endl;
        return;
    }

    // Create proper HTML document with styling
    f << "<!DOCTYPE html>\n"
      << "<html>\n<head>\n"
      << "<title>MNIST K-Means Clustering Results </title>\n"
      << "<style>\n"
      << "body { background-color: #" << htmlRandomBackground() << "; font-family: Arial, sans-serif; }\n"
      << "table { border-collapse: collapse; margin: 20px; }\n"
      << ".cluster-container { vertical-align: top; padding: 10px; }\n"
      << ".centroid { border: 2px solid #f00; margin-bottom: 10px; }\n"
      << ".digit-image { margin: 3px; }\n"
      << ".pixel { width: 5px; height: 5px; }\n"
      << "</style>\n</head>\n<body>\n"
      << "<h1>MNIST Clustering Results - Sequential Algorithm</h1>\n"
      << "<table><tr>\n";

    // For each cluster, create a column in the table
    for (int clusterIdx = 0; clusterIdx < K; clusterIdx++) {
        const auto& cluster = clusters[clusterIdx];

        f << "<td class=\"cluster-container\">\n"
          << "<h3>Cluster " << (clusterIdx + 1) << "</h3>\n"
          << "<p>" << cluster.elements.size() << " images</p>\n"
          << "<div class=\"centroid\">";

        // First display the centroid of the cluster
        htmlCell(f, cluster.centroid);
        f << "</div>\n";

        // Show sample of images in this cluster
        const int maxImagesToShow = std::min(20, static_cast<int>(cluster.elements.size()));
        for (int i = 0; i < maxImagesToShow; i++) {
            f << "<div class=\"digit-image\">";
            htmlCell(f, images[cluster.elements[i]]);
            f << "</div>\n";
        }

        // Show message if some images are hidden
        if (static_cast<int>(cluster.elements.size()) > maxImagesToShow) {
            f << "<p>(" << (static_cast<int>(cluster.elements.size()) - maxImagesToShow)
              << " more images not shown)</p>\n";
        }

        f << "</td>\n";
    }

    // Write HTML footer
    f << "</tr></table>\n</body>\n</html>\n";

    // Ensure the file is written
    f.flush();

    if (f.fail()) {
        std::cerr << "Error: Failed to write HTML visualization to " << filename << std::endl;
    }
}