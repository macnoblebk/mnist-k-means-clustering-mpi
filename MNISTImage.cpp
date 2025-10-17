


#include "MNISTImage.h"

MNISTImage::MNISTImage(const Pixels pixels) noexcept: pixels(pixels) {}


std::string MNISTImage::pixelToHex(int row, int col) const noexcept {
u_char pixel = pixels[ROWS_N * row + col];
char buffer[7];
snprintf(buffer, sizeof(buffer), "%.6x", pixel << 16 | pixel << 8 | pixel);
return {buffer};
}
