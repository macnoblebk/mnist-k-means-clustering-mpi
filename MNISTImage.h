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

class MNISTImage {
private:
    static const int ROWS_N = 28;

    static const int COLS_N = 28;

    static const int PIXELS_N = ROWS_N * COLS_N;

    using Pixels = std::array<u_char, PIXELS_N>;

    Pixels pixels;
public:
    MNISTImage() {}

    MNISTImage(const Pixels pixels) noexcept;

    std::string pixelToHex(int row, int col) const noexcept;

    double euclideanDistance(const MNISTImage& other) const;

    u_char getPixel(int row, int col) const;

    constexpr static int getNumRows() noexcept { return ROWS_N; }

    constexpr static int getNumCols() noexcept { return COLS_N; }

    constexpr static int getNumPixels() noexcept { return PIXELS_N; }
};