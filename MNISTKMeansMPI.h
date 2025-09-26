template<int k, int d>
class MNISTKMeansMPI : public KMeansMPI<k, d> {
public:
    void fit(MNISTImage* images, int n) {

        KMeansMPI<k, d>::fit(reinterpret_cast<std::array<u_char, d>*>(images), n);
    }

protected:
    using Element = std::array<u_char, d>;

    double distance(const Element& a, const Element& b) const {
        // Convert raw element data to MNISTImage objects to use their distance method
        return MNISTImage(a).euclideanDistance(MNISTImage(b));
    }
};