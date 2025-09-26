/**
 * @file MNISTImage.h
 * @author Mac-Noble Brako-Kusi
 * @date 03-07-2025
 * @see "Seattle University, CPSC5600, Winter 2025"
 */

#pragma once
#include <array>
#include <string>
#include <cmath>

/**
 * @class MNISTImage
 * @brief A wrapper class for handling MNIST handwritten digit images
 *
 * This class encapsulates a single 28x28 pixel grayscale image from the MNIST dataset.
 * It provides methods for accessing pixel data and comparing images.
 */
class MNISTImage {
private:
    static const int ROWS_N = 28;    // Number of rows in the MNIST image

    static const int COLS_N = 28;  // Number of columns in the MNIST image

    static const int PIXELS_N = ROWS_N * COLS_N; // Total number of pixels in the MNIST image

    using Pixels = std::array<u_char, PIXELS_N>;  // Type alias for the pixel data storage

    Pixels pixels;  // The actual pixel data of the image
public:
    /**
     * @brief Default constructor that creates an empty image
     */
    MNISTImage() {}

    /**
     * @brief Constructor that initializes the image with pixel data
     * @param pixels Array of pixel values representing the MNIST image
     */
    MNISTImage(const Pixels pixels) noexcept;

    /**
     * @brief Converts a pixel value to a hexadecimal color string
     * @param row The row index of the pixel (0-27)
     * @param col The column index of the pixel (0-27)
     * @return String representation of the pixel as a hex color code
     */
    std::string pixelToHex(int row, int col) const noexcept;

    /**
    * @brief Calculates the Euclidean distance between this image and another
    * @param other The image to compare with
    * @return The Euclidean distance in the 784-dimensional pixel space
    */
    double euclideanDistance(const MNISTImage& other) const;

    /**
     * @brief Gets the value of a pixel at the specified position
     * @param row The row index of the pixel (0-27)
     * @param col The column index of the pixel (0-27)
     * @return The grayscale value of the pixel (0-255)
     */
    u_char getPixel(int row, int col) const;

     /**
      * @brief Gets the number of rows in an MNIST image
      * @return The number of rows (28)
      */
    constexpr static int getNumRows() noexcept { return ROWS_N; }

    /**
     * @brief Gets the number of columns in an MNIST image
     * @return The number of columns (28)
     */
    constexpr static int getNumCols() noexcept { return COLS_N; }

    /**
     * @brief Gets the total number of pixels in an MNIST image
     * @return The total number of pixels (784)
     */
    constexpr static int getNumPixels() noexcept { return PIXELS_N; }
};