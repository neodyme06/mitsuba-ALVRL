#include <mitsuba/core/plugin.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/render/util.h>

MTS_NAMESPACE_BEGIN

class Rms : public Utility {
public:
    int run(int argc, char **argv) {
        if (!(argc == 4 || argc == 5 || argc == 6)) {
            cout << "Compute the pixel rms of the two given images (after gamma correction)" << endl;
            cout << "Syntax: mtsutil rms <gamma> <image.exr> <reference.exr> [robust fraction] [if this argument is present (no matter what it is) -> do a relative rms -- and yes, that's a quick hack and should be done with a flag :P]" << endl;
            return -1;
        }
        char *end_ptr = NULL;
        double gamma = strtod(argv[1], &end_ptr);
        if (*end_ptr != '\0')
            SLog(EError, "Could not parse gamma value '%s'", argv[1]);
        Log(EInfo, "Opening file %s", argv[2]);
        ref<FileStream> aFile = new FileStream(argv[2], FileStream::EReadOnly);
        Log(EInfo, "Opening file %s", argv[3]);
        ref<FileStream> bFile = new FileStream(argv[3], FileStream::EReadOnly);

        ref<Bitmap> aBitmap = new Bitmap(Bitmap::EOpenEXR, aFile);
        ref<Bitmap> bBitmap = new Bitmap(Bitmap::EOpenEXR, bFile);

        /* A few sanity checks */
        if (aBitmap->getPixelFormat() != bBitmap->getPixelFormat())
            Log(EError, "Error: Input bitmaps have a different pixel format!");
        if (aBitmap->getComponentFormat() != bBitmap->getComponentFormat())
            Log(EError, "Error: Input bitmaps have a different component format!");
        if (aBitmap->getSize() != bBitmap->getSize())
            Log(EError, "Error: Input bitmaps have a different size!");

        size_t nEntries =
            (size_t) aBitmap->getSize().x *
            (size_t) aBitmap->getSize().y *
            aBitmap->getChannelCount();

        size_t numItemsToDrop = 0; // This isn't very kosher, but sometimes vaguely useful
        if (argc >= 5) {
            char *end_ptr = NULL;
            double robustFraction = strtod(argv[4], &end_ptr); // cut away this fraction at both extremes of the deviations
            if (*end_ptr != '\0')
                SLog(EError, "Could not parse robust fraction '%s'", argv[4]);
            numItemsToDrop = 0.5 + nEntries * robustFraction;
            if (2 * numItemsToDrop >= nEntries) {
                SLog(EError, "robustFraction: dropping more elements than there are available!");
            }
        }

        bool relativeRMS = false;
        if (argc >= 6)
            relativeRMS = true;

        // save deviations and sort squared deviations before adding, for numerical stability
        std::vector<double> diffs(nEntries);

        for (size_t i = 0; i < nEntries; i++) {
            double sample, reference;
            switch (aBitmap->getComponentFormat()) {
                case Bitmap::EFloat16: {
                            half *aData = aBitmap->getFloat16Data();
                            half *bData = bBitmap->getFloat16Data();
                            sample = aData[i];
                            reference = bData[i];
                        }
                        break;
                case Bitmap::EFloat32: {
                            float *aData = aBitmap->getFloat32Data();
                            float *bData = bBitmap->getFloat32Data();
                            sample = aData[i];
                            reference = bData[i];
                        }
                        break;
                case Bitmap::EUInt32: {
                            uint32_t *aData = aBitmap->getUInt32Data();
                            uint32_t *bData = bBitmap->getUInt32Data();
                            sample = aData[i];
                            reference = bData[i];
                        }
                        break;
                default:
                    Log(EError, "Unsupported component format!");
                    return -1;
            }
            sample = std::pow(sample, 1.0/gamma);
            reference = std::pow(reference, 1.0/gamma);
            if (relativeRMS)
                diffs[i] = reference == 0 ? 0 : (sample - reference) / reference; // mask off zero-reference areas for relative RMS
            else
                diffs[i] = sample - reference;
        }

        if (numItemsToDrop > 0) {
            // stort first so we can drop the extremal values on either side
            std::sort(diffs.begin(), diffs.end());
        }
        // get squared deviations
        for (size_t i = numItemsToDrop; i < nEntries - numItemsToDrop; i++) {
            diffs[i] = diffs[i]*diffs[i];
        }
        // For numerical stability: sort squared differences before accumulation
        std::sort(diffs.begin() + numItemsToDrop, diffs.end() - numItemsToDrop);
        double acc = 0;
        for (size_t i = numItemsToDrop; i < nEntries - numItemsToDrop; i++) {
            acc += diffs[i];
        }
        std::cout << std::sqrt(acc / (nEntries - 2*numItemsToDrop)) << endl;

        return 0;
    }

    MTS_DECLARE_UTILITY()
};

MTS_EXPORT_UTILITY(Rms, "Compute the root mean squared difference between two EXR images")
MTS_NAMESPACE_END

