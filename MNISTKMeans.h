/**
 * @file MNISTKMeans.h
 * @author Mac-Noble Brako-Kusi
 * @date 03-07-2025
 * @see "Seattle University, CPSC5600, Winter 2025"
 */

#pragma once
#include "KMeans.h"
#include "MNISTImage.h"


template<int k, int d>
class MNISTKMeans : public KMeans<k, d> {
public:
    void fit(MNISTImage* images, int n) {
        // Reinterpret the MNISTImage array as array of element type expected by KMeans
        KMeans<k, d>::fit(reinterpret_cast<std::array<u_char, d>*>(images), n);
    }

protected:
    // Type alias for readability
    using Element = std::array<u_char, d>;


    double distance(const Element& a, const Element& b) const {
        // Convert raw element data to MNISTImage objects to use their distance method
        return MNISTImage(a).euclideanDistance(MNISTImage(b));
    }
};