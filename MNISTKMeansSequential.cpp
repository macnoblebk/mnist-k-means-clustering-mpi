

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