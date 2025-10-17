/**
 * @file MNISTImage.cpp
 * @author Mac-Noble Brako-Kusi
 * @date 03-07-2025
 * @see "Seattle University, CPSC5600, Winter 2025"
 */


#include "MNISTImage.h"
/**
 * @brief Constructor that initializes the image with pixel data
 * @param pixels Array of pixel values representing the MNIST image
 */
MNISTImage::MNISTImage(const Pixels pixels) noexcept: pixels(pixels) {}

/**
 * @brief Converts a pixel value to a hexadecimal color string
 * @param row The row index of the pixel (0-27)
 * @param col The column index of the pixel (0-27)
 * @return String representation of the pixel as a hex color code
 */
std::string MNISTImage::pixelToHex(int row, int col) const noexcept {
u_char pixel = pixels[ROWS_N * row + col];
char buffer[7];
snprintf(buffer, sizeof(buffer), "%.6x", pixel << 16 | pixel << 8 | pixel);
return {buffer};
}


/**
 * @brief Calculates the Euclidean distance between this image and another
 * @param other The image to compare with
 * @return The Euclidean distance in the 784-dimensional pixel space
 */
double MNISTImage::euclideanDistance(const MNISTImage &other) const {
    double sum = 0.0;
    for (int i = 0; i < PIXELS_N; i++) {
        double difference = static_cast<double>(pixels[i]) - static_cast<double>(other.pixels[i]);
        sum += difference * difference;
    }
    return std::sqrt(sum);
}

/**
 * @brief Gets the value of a pixel at the specified position
 * @param row The row index of the pixel (0-27)
 * @param col The column index of the pixel (0-27)
 * @return The grayscale value of the pixel (0-255)
 */
u_char MNISTImage::getPixel(int row, int col) const {
    return pixels[ROWS_N * row + col];
}