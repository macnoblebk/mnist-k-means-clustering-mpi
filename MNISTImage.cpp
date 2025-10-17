


#include "MNISTImage.h"

MNISTImage::MNISTImage(const Pixels pixels) noexcept: pixels(pixels) {}


std::string MNISTImage::pixelToHex(int row, int col) const noexcept {
u_char pixel = pixels[ROWS_N * row + col];
char buffer[7];
snprintf(buffer, sizeof(buffer), "%.6x", pixel << 16 | pixel << 8 | pixel);
return {buffer};
}


double MNISTImage::euclideanDistance(const MNISTImage &other) const {
    double sum = 0.0;
    for (int i = 0; i < PIXELS_N; i++) {
        double difference = static_cast<double>(pixels[i]) - static_cast<double>(other.pixels[i]);
        sum += difference * difference;
    }
    return std::sqrt(sum);
}