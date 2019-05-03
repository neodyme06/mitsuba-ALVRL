#include <mitsuba/core/plugin.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/bidir/rsampler.h>
#include <mitsuba/core/warp.h>
#include <cmath>
#include <numeric>
#include <utility>
#include <list>
#include <boost/heap/priority_queue.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/timer/timer.hpp>
using boost::timer::cpu_timer;
using boost::timer::cpu_times;

MTS_NAMESPACE_BEGIN

class Preprocessor: public Object{

public:
    Preprocessor(bool globalCluster, bool localRefinement, Float targetNumSlices,
            uint32_t neighbourCount, Float neighbourWeight,
            Float globalUndersampling, Float localUndersampling, Float fallBackUndersampling,
            Float depthCorrection, Float sliceCurvatureFactor, int workerCount) :
                m_targetNumSlices(targetNumSlices),
                m_sliceCurvatureFactor(sliceCurvatureFactor),
                m_neighbourCount(neighbourCount),
                m_neighbourWeight(neighbourWeight),
                m_globalCluster(globalCluster),
                m_localRefinement(localRefinement),
                m_globalUndersampling(globalUndersampling),
                m_localUndersampling(localUndersampling),
                m_fallBackUndersampling(fallBackUndersampling),
                m_depthCorrection(depthCorrection),
                m_workerCount(workerCount) {
        m_globalPixelUndersampling = -1; /* sentinel */
        if (m_targetNumSlices < 1) {
            Log(EError, "Invalid target number of slices!");
        }
    }

//////////////////////////////////////////////////////////////////////////
/*                  SECOND STEP: CLUSTER VRLS PER SLICE:                */
//////////////////////////////////////////////////////////////////////////

    struct GatherPoint {
        Point2 pixel;
        Point position;
        Point direction;
        GatherPoint() {}
        GatherPoint(const Point2 &pix, const Point &pos, const Point &dir) :
            pixel(pix), position(pos), direction(dir) {}
    };
    struct Slice {
        std::vector<GatherPoint> gatherPoints;
        Point positionCentroid;
        Point directionCentroid;
        Slice(const std::vector<GatherPoint> &gatherPts,
                const Point &posCentroid,
                const Point &dirCentroid) :
                    gatherPoints(gatherPts),
                    positionCentroid(posCentroid),
                    directionCentroid(dirCentroid) {}
        size_t numPixels() const {
            return gatherPoints.size();
        }
        std::vector<Point2> sampleRepresentativePixels(
                Float targetUndersampling, Sampler *sampler) const {
            std::vector<Point2> pixels;
            size_t targetNum = 0.5 + numPixels() / targetUndersampling;
            if (targetNum < 2) {
                // Need at least two representatives per slice for variance calculation!
                targetNum = std::min((size_t)2, numPixels());
                SLog(EWarn, "sampleRepresentativePixels: targetNum was too low, "
                        "increased to %d (numPix: %d)", targetNum, numPixels());
            } else {
                SLog(EDebug, "sampleRepresentativePixels: target: %d pixels out of %d,"
                        "(undersampling %f)",
                        targetNum, numPixels(),
                        ((Float)numPixels()) / targetNum);
            }
            if (numPixels() <= targetNum) {
                // every pixel in this slices is selected
                pixels.resize(numPixels());
                for (size_t i = 0; i < numPixels(); i++) {
                    pixels[i] = gatherPoints[i].pixel;
                }
                return pixels;
            }

            std::vector<uint32_t> indices;
            if (numPixels() <= 2*targetNum) {
                // shuffle gatherpoints, take targetNum
                indices.resize(numPixels());
                for (size_t i = 0; i < numPixels(); i++)
                    indices[i] = i;
                for (size_t i = numPixels()-1; i > 0; i--)
                    std::swap(indices[i], indices[(i+1) * sampler->next1D()]);
            } else {
                // sample indices, repeat in case of duplicate
                indices.resize(targetNum);
                size_t n = 0;
                while (n < targetNum) {
                    bool unique;
                    do {
                        indices[n] = sampler->next1D() * numPixels();
                        unique = true;
                        for (size_t i = 0; i < n; i++) {
                            if (indices[i] == indices[n]) {
                                unique = false;
                                break;
                            }
                        }
                    } while (!unique);
                    n++;
                }
            }
            pixels.resize(targetNum);
            for (size_t i = 0; i < targetNum; i++)
                pixels[i] = gatherPoints[indices[i]].pixel;
            return pixels;
        }
    };

    /**
     * Fallback cluster: used for pixels that have no associated gather
     * point (no surface got hit), and for slices whose gather points
     * (icluding neighbours) got no contribution in *any* VRL.
     *
     * TODO: the (slice) order in which the elements of R are given, *must*
     * be the same as the one that was returned by sampleSliceMapping() --
     * rewrite all this to avoid these ugly dependencies
     */
    std::vector<std::vector<uint32_t> > buildClusters(
            const std::vector<std::vector<std::vector<VrlContribution> > > &R,
            std::vector<std::vector<Float> > &clusterweight,
            std::vector<uint32_t> &globalCluster, std::vector<Float> &globalWeights,
            std::vector<uint32_t> &fallBackCluster, std::vector<Float> &fallBackWeights,
            Sampler *sampler) {
        std::vector< std::vector<uint32_t> > globalVrlsPerCluster;
        std::vector< std::vector<uint32_t> > result;
        cpu_times elapsed;
        float elapsedCpu, elapsedWall;


        // TODO blergh memory usage -- vector of references?
        size_t numGatherPoints = 0;
        for (size_t i = 0; i < R.size(); i++)
            numGatherPoints += R[i].size();
        std::vector<std::vector<VrlContribution> > Rflat(numGatherPoints);
        size_t n = 0;
        for (size_t i = 0; i < R.size(); i++) {
            for (size_t j = 0; j < R[i].size(); j++)
                Rflat[n + j] = R[i][j];
            n += R[i].size();
        }


        cpu_timer timer;
        cluster(Rflat, globalVrlsPerCluster, sampler); //global cluster
        Log(EInfo, "Global cluster found, sampling representatives...");
        if (m_globalPixelUndersampling < 0) {
            Log(EError, "Invalid pixel undersampling. "
                    "Did you forget to call buildSlices first?");
        }
        boost::numeric::ublas::vector<double> dummyLocalityWeights(
                numGatherPoints, 1.0/numGatherPoints);
        Clustering globalClustering(globalVrlsPerCluster, Rflat,
                dummyLocalityWeights, m_globalPixelUndersampling);
        globalClustering.sampleRepresentatives(globalCluster, globalWeights, sampler);
        elapsed = timer.elapsed();
        elapsedCpu = (elapsed.system + elapsed.user) * 1e-9;
        elapsedWall = elapsed.wall * 1e-9;
        Log(EInfo, "Global cluster build: %e wall, %e cpu", elapsedWall, elapsedCpu);

        timer = cpu_timer(); // new timer (resets clock)
        Log(EInfo, "Refining for fallback clustering");
        if (!globalClustering.refine(m_fallBackUndersampling, sampler))
            Log(EError, "couldn't refine global clustering! (but all VRLs should be non-zero!)");
        globalClustering.sampleRepresentatives(fallBackCluster, fallBackWeights, sampler);
        Log(EInfo, "FallBack clustering with %d clusters (undersampling %f)",
                fallBackCluster.size(), R[0].size() / ((Float) fallBackCluster.size()));
        elapsed = timer.elapsed();
        elapsedCpu = (elapsed.system + elapsed.user) * 1e-9;
        elapsedWall = elapsed.wall * 1e-9;
        Log(EInfo, "Fallback cluster refinement: %e wall, %e cpu", elapsedWall, elapsedCpu);

        timer = cpu_timer(); // new timer (resets clock)
        Log(EInfo, "Local refinement...");
        result = refinePerSlice(globalVrlsPerCluster, R,
                fallBackCluster, fallBackWeights, clusterweight, sampler);
        elapsed = timer.elapsed();
        elapsedCpu = (elapsed.system + elapsed.user) * 1e-9;
        elapsedWall = elapsed.wall * 1e-9;
        Log(EInfo, "Local cluster refinement: %e wall, %e cpu", elapsedWall, elapsedCpu);

        return result;
    }

    std::vector< std::vector <uint32_t> > refinePerSlice(
            const std::vector< std::vector<uint32_t> > &vrlsPerCluster,
            const std::vector< std::vector< std::vector<VrlContribution> > > &R,
            const std::vector<uint32_t> &fallBackCluster,
            const std::vector<Float> &fallBackWeights,
            std::vector<std::vector<Float> > &clusterweight,
            Sampler *sampler) {
        size_t numSlices = R.size();

        std::vector< std::vector <uint32_t> > result(numSlices);
        clusterweight.resize(numSlices);

        /* Do the refinement */
        if (m_workerCount > 1) {
            std::vector<ClusterRefiner *> refiners(m_workerCount);
            for (int i = 0; i < m_workerCount; i++) {
                refiners[i] = new ClusterRefiner(i, m_workerCount,
                        vrlsPerCluster, R, fallBackCluster, fallBackWeights,
                        *this, sampler);
                refiners[i]->incRef();
                refiners[i]->start();
            }
            for (int i = 0; i < m_workerCount; i++) {
                refiners[i]->join();
            }
            for (int i = 0; i < m_workerCount; i++) {
                refiners[i]->merge(result, clusterweight);
                refiners[i]->decRef();
            }
            refiners.clear();
        } else {
            for (size_t i = 0; i < numSlices; i++) { // For each slice i
                refineSlice(i, vrlsPerCluster, fallBackCluster, fallBackWeights,
                        result[i], clusterweight[i], R, sampler);
            }
        }

        /* Print some info */
        uint32_t totalNumSelectedVrls = 0;
        for (uint32_t s = 0; s < result.size(); s++) { //slice
            totalNumSelectedVrls += result[s].size();
        }
        Float averageClustersPerSlice = ((Float) totalNumSelectedVrls) / numSlices;
        Float var = 0;
        for (size_t s = 0; s < numSlices; s++) { //slice
            Float dev = (result[s].size() - averageClustersPerSlice);
            var += dev*dev;
        }
        var /= numSlices;
        SLog(EInfo, "Average %f +/- %f clusters per slice (undersampling factor %f)",
                averageClustersPerSlice, sqrt(var), numVrls(R) / averageClustersPerSlice);

        return result;
    }

    void refineSlice(uint32_t i,
            const std::vector< std::vector<uint32_t> > &vrlsPerCluster,
            const std::vector<uint32_t> &fallBackCluster,
            const std::vector<Float> &fallBackWeights,
            std::vector<uint32_t> &reprVrls,
            std::vector<Float> &weights,
            const std::vector< std::vector< std::vector<VrlContribution> > > &R,
            Sampler *sampler) const {
        /* TODO: if slice has less pixels than 'reprPixelsPerSlice' ->
         * don't bother refining; just do every VRL? (we already calculated
         * it in R ... make sure that never happens more early on?) */
        std::vector< std::vector<VrlContribution> > L_i; // Local matrix for slice i
        boost::numeric::ublas::vector<double> localityWeights;
        L_i = getLocalMatrix(R, localityWeights, i);
        Clustering clustering(vrlsPerCluster, L_i,
                localityWeights, m_sliceUndersampling[i], m_depthCorrection);
        if (!m_localRefinement) {
            // Just sample the global cluster and be done with it
            clustering.sampleRepresentatives(reprVrls, weights, sampler);
            return;
        }
        // Try to do the refinement
        if (clustering.refine(m_localUndersampling, sampler)) {
            clustering.sampleRepresentatives(reprVrls, weights, sampler);
        } else {
            Log(EWarn, "Could not refine slice %d, using fall-back clustering!", i);
            reprVrls = fallBackCluster;
            weights = fallBackWeights;
        }
    }


    /// Helper class for refining: clustering with cost
    class Clustering {
    private:
        struct ClusterNode {
            ClusterNode(Float uvar, Float ivar, uint32_t b, uint32_t e)
                    : undersamplingVar(uvar), integrationVar(ivar), begin(b), end(e) {}
            const bool operator < (const ClusterNode &other) const {
                return undersamplingVar + integrationVar < other.undersamplingVar + other.integrationVar; }
            Float undersamplingVar; /// expected variance due to sampling only one VRL from the cluster
            Float integrationVar; /// expected variance to integrating the one sampled VRL from the cluster with the eye rays
            uint32_t begin; /// begin index of vrls of this cluster in m_vrls
            uint32_t end; /// end index of vrls of this cluster in m_vrls (non inclusive)
        };

    public:
        Clustering(const std::vector< std::vector<uint32_t> > &vrlsPerCluster,
                    const std::vector< std::vector<VrlContribution> > &M,
                    const boost::numeric::ublas::vector<double> &localityWeights,
                    Float pixelUndersampling, Float depthCorrection = 1)
                : m_M(M), m_localityWeights(localityWeights),
                m_pixelUndersampling(pixelUndersampling), m_depthCorrection(depthCorrection) {
            Float norm = boost::numeric::ublas::norm_1(m_localityWeights);
            if (fabs(norm - 1) > 1e-3) {
                SLog(EError, "Incorrect normalization in localityWeights: %f", norm);
            }
            if (m_pixelUndersampling <= 0 || m_pixelUndersampling > 1) {
                SLog(EError, "Invalid pixel undersampling: %f", m_pixelUndersampling);
            }
            calculateColumnWeigths(m_M, m_localityWeights, m_columnWeights);
            m_clusterUndersamplingVariance = 0;  // gets incremented with each addCluster below
            m_clusterVrlIntegrationVariance = 0; // gets incremented with each addCluster below
            uint32_t numClust = vrlsPerCluster.size();
            uint32_t numVrls = 0;
            for (uint32_t i = 0; i < numClust; i++) {
                uint32_t clusterSize = vrlsPerCluster[i].size();
                numVrls += clusterSize;
            }
            m_vrls.resize(numVrls);
            uint32_t begin = 0;
            for (uint32_t i = 0; i < numClust; i++) {
                uint32_t clusterSize = vrlsPerCluster[i].size();
                for (uint32_t j = 0; j < clusterSize; j++) {
                    uint32_t k = begin + j;
                    m_vrls[k] = vrlsPerCluster[i][j];
                }
                addCluster(begin, begin + clusterSize);
                begin += clusterSize;
            }
            calculateUnclusteredVariance(m_M, m_localityWeights, m_vrls.begin(), m_vrls.end(),
                    m_vrlTracingVariance, m_unclusteredVrlIntegrationVariance);
            if (!std::isfinite(m_vrlTracingVariance) || m_vrlTracingVariance <= 0)
                SLog(EWarn, "Vrl tracing variance is %e", m_vrlTracingVariance);
            if (!std::isfinite(m_unclusteredVrlIntegrationVariance)
                    || m_unclusteredVrlIntegrationVariance <= 0)
                SLog(EWarn, "Vrl tracing variance is %e", m_unclusteredVrlIntegrationVariance);
        }

        uint32_t numSingletonClusters() const {
            return m_singletonClusters.size();
        }
        uint32_t numMultiClusters() const {
            return m_pq.size();
        }
        uint32_t numClusters() const {
            return numSingletonClusters() + numMultiClusters();
        }

        /// Sample representatives from this clustering
        void sampleRepresentatives(
                std::vector<uint32_t> &reprVrls,
                std::vector<Float> &weights,
                Sampler *sampler) const {
            reprVrls.resize(numClusters());
            weights.resize(numClusters());
            int i = 0;
            // Singleton clusters
            for (std::list<uint32_t>::const_iterator iter = m_singletonClusters.begin();
                    iter != m_singletonClusters.end(); ++iter) {
                reprVrls[i] = *iter;
                weights[i] = 1;
                i++;
            }
            // Multi-VRL clusters
            for (boost::heap::priority_queue<ClusterNode>::const_iterator iter = m_pq.begin();
                    iter != m_pq.end(); ++iter) {
                Float prob;
                uint32_t j = weightedSample(m_columnWeights, sampler, &prob,
                                            iter->begin, iter->end, &m_vrls);
                reprVrls[i] = m_vrls[j];
                weights[i] = 1.0f / prob;
                i++;
            }
        }

        bool refine(Float undersampling, Sampler *sampler) {
            if (undersampling <= 0)
                return refineAdaptively(sampler, m_depthCorrection);
            else
                return refineFixedDepth(undersampling, sampler);
        }

        bool refineFixedDepth(Float undersampling, Sampler *sampler) {
            uint32_t targetClusters = 0.5 + numVrls() / undersampling;
            if (numClusters() >= targetClusters || numMultiClusters() <= 0)
                return true;
            /* TODO: if it is certain that we have to split every multi
             * cluster to singletons: do that immediately */
            while (numClusters() < targetClusters && numMultiClusters() > 0) {
                ClusterNode cn = popMultiCluster();
                if (!split(cn.begin, cn.end, sampler))
                    SLog(EError, "couldn't split cluster!");
            }
            return true;
        }

        // TODO: Parallelize: distribute slices over all available cores!
        bool refineAdaptively(Sampler *sampler, Float depthCorrection) {
            ref<ReplayableSampler> rplSampler;
            size_t initialSampleIndex = 0;
            if (depthCorrection != 1) {
                rplSampler = new ReplayableSampler(); // TODO: wasteful to generate new sampler all the time...?
                initialSampleIndex = rplSampler->getSampleIndex();
            }

            if (numMultiClusters() <= 0)
                return true;

            if (unclusteredVariance() == 0) {
                SLog(EDebug, "Zero unclustered variance vrl set in refine()! "
                        "Not refining cluster with %d singletons and %d multis!",
                        numSingletonClusters(), numMultiClusters());
                return false;
            }

            Float bestConstant = convergenceConstant();
            int numberOfSplits = 0;
            int bestNumberOfSplits = 0;
            makeRefinementSnapshot();
            while (numMultiClusters() > 0) {
                ClusterNode cn = popMultiCluster();

                if (depthCorrection == 1) {
                    if (!split(cn.begin, cn.end, sampler))
                        SLog(EError, "couldn't split cluster!");
                } else {
                    if (!split(cn.begin, cn.end, rplSampler.get()))
                        SLog(EError, "couldn't split cluster!");
                }

                numberOfSplits++;
                Float currConstant = convergenceConstant();

                if (currConstant < bestConstant) {
                    /* TODO: no need to keep making snapshots if we are
                     * always decreasing, only when we increase again
                     * afterwards... */
                    if (depthCorrection == 1) {
                        makeRefinementSnapshot();
                    }
                    bestConstant = currConstant;
                    bestNumberOfSplits = numberOfSplits;
                }

                if (lowerBoundOfFutureConvergenceConstants() >= bestConstant) {
                    /* We are *guaranteed* to have found the optimum */
                    break;
                }
            }
            restoreSnapshot();

            if (depthCorrection != 1) {
                // rewind, but go to different depth this time
                // (snapshot of initial state was already restored above)
                rplSampler->setSampleIndex(initialSampleIndex);
                int correctedNumberOfSplits = 0.5 + depthCorrection * bestNumberOfSplits;
                for (int i = 0; i < correctedNumberOfSplits; i++) {
                    if (numMultiClusters() == 0) {
                        SLog(EWarn, "depthCorrection in splitting needs to split deeper than there are available clusters!");
                        break;
                    }
                    ClusterNode cn = popMultiCluster();
                    if (!split(cn.begin, cn.end, rplSampler.get()))
                        SLog(EError, "couldn't split cluster in second pass!");
                }
                SLog(EDebug, "Finished refinement with depth corr (%d clusters), expected convergence gain factor: %f, determined ideal gain was %f, render/preprocess work: %f (unclustVar %e, clustVar %e: traceVar %e, unclustIntVar %e, clustUnderSampVar %e, clustIntVar %e)",
                        numClusters(),
                        unclusteredConvergenceConstant() / convergenceConstant(),
                        unclusteredConvergenceConstant() / bestConstant,
                        numClusters() / (numVrls() * m_pixelUndersampling),
                        unclusteredVariance(), clusteredVariance(),
                        m_vrlTracingVariance, m_unclusteredVrlIntegrationVariance,
                        m_clusterUndersamplingVariance, m_clusterVrlIntegrationVariance);
            } else {
                SLog(EDebug, "Finished refinement (%d clusters), expected convergence gain factor: %f, render/preprocess work: %f (unclustVar %e, clustVar %e: traceVar %e, unclustIntVar %e, clustUnderSampVar %e, clustIntVar %e)",
                        numClusters(),
                        unclusteredConvergenceConstant() / bestConstant,
                        numClusters() / (numVrls() * m_pixelUndersampling),
                        unclusteredVariance(), clusteredVariance(),
                        m_vrlTracingVariance, m_unclusteredVrlIntegrationVariance,
                        m_clusterUndersamplingVariance, m_clusterVrlIntegrationVariance);
            }

            return true;
        }

        Float unclusteredVariance() const {
            return m_vrlTracingVariance + m_unclusteredVrlIntegrationVariance;
        }
        Float clusteredVariance() const {
            return m_vrlTracingVariance + m_clusterUndersamplingVariance + m_clusterVrlIntegrationVariance;
        }

        /**
         * Assumption:
         * Preproccessing time ~ the number of VRLs times the pixel undersampling factor
         * Rendering time      ~ the number of clusters
         */
        Float convergenceConstant() const {
            Float c = (numVrls() * m_pixelUndersampling + numClusters()) * clusteredVariance();
            if (!std::isfinite(c) || c <= 0) {
                SLog(EError, "invalid convergence constant %f", c);
            }
            return c;
        }
        /// Future convergence constants (after more cluster splitting) are guaranteed to be above this value
        Float lowerBoundOfFutureConvergenceConstants() const {
            Float c = (numVrls() * m_pixelUndersampling + numClusters()) * unclusteredVariance();
            if (!std::isfinite(c) || c <= 0) {
                SLog(EError, "invalid lower bound on convergence constant %f", c);
            }
            return c;
        }
        Float unclusteredConvergenceConstant() const {
            Float c = numVrls() * unclusteredVariance();
            if (!std::isfinite(c) || c <= 0) {
                SLog(EError, "invalid unclustered convergence constant %f", c);
            }
            return c;
        }

        std::vector< std::vector<uint32_t> > getVrlsPerCluster() const {
            std::vector< std::vector<uint32_t> > vrlsPerCluster(numClusters());
            std::vector<uint32_t> singletonVector(1);
            uint32_t c = 0;
            for (std::list<uint32_t>::const_iterator iter = m_singletonClusters.begin(); iter != m_singletonClusters.end(); ++iter) {
                singletonVector[0] = *iter;
                vrlsPerCluster[c] = singletonVector;
                c++;
            }
            for (boost::heap::priority_queue<ClusterNode>::const_iterator iter = m_pq.begin(); iter != m_pq.end(); ++iter) {
                vrlsPerCluster[c].assign(&m_vrls[iter->begin],
                                         &m_vrls[iter->end]);
                c++;
            }
            if (c != numClusters())
                SLog(EError, "error in getVrlsPerCluster()");
            return vrlsPerCluster;
        }

    private:
        uint32_t numVrls() const {
            return Preprocessor::numVrls(m_M);
        }
        void addCluster(uint32_t begin, uint32_t end, Float undersampVar, Float integrationVar) {
#if 0
            Float var2 = clusterVariance(m_M, m_columnWeights, &(m_vrls[begin]), &(m_vrls[end]));
            if (fabs((var-var2)/(var+var2)) > 1e-3)
                SLog(EError, "trying to add cluster with inconsistent error: %e vs %e", var, var2);
#endif

            if (end == begin)
                SLog(EError, "Trying to add empty cluster!");

            if (end == begin + 1) {
                // Singleton cluster
                m_singletonClusters.push_front(m_vrls[begin]);
                if (undersampVar != 0)
                    SLog(EError, "Trying to add singleton cluster with non-zero undersampling variance: %e", undersampVar);
                m_clusterVrlIntegrationVariance += integrationVar;
            } else {
                // Multi-VRL cluster: add to priority queue
                ClusterNode cn(undersampVar, integrationVar, begin, end);
                m_pq.push(cn);
                m_clusterUndersamplingVariance += undersampVar;
                m_clusterVrlIntegrationVariance += integrationVar;
            }
        }
        void addCluster(uint32_t begin, uint32_t end, std::pair<Float,Float> varPair) {
            addCluster(begin, end, varPair.first, varPair.second);
        }
        void addCluster(uint32_t begin, uint32_t end) {
            addCluster(begin, end,
                    calculateClusterVariance(m_M, m_columnWeights, m_localityWeights, &(m_vrls[begin]), &(m_vrls[end])));
        }

        ClusterNode popMultiCluster() {
            ClusterNode cn = m_pq.top();
            m_pq.pop();
            m_clusterUndersamplingVariance -= cn.undersamplingVar;
            m_clusterVrlIntegrationVariance -= cn.integrationVar;
            return cn;
        }

        /// Split cluster i
        bool split(uint32_t begin, uint32_t end, Sampler *sampler) {
            uint32_t clusterSize = end - begin;

            if (clusterSize < 2)
                return false;

            /* Sample two centers with weights according to their luminance */
            uint32_t vrl1 = m_vrls[weightedSample(m_columnWeights, sampler, NULL, begin, end, &m_vrls)];
            // temporarily set luminance of first sample to zero so we don't sample it again;
            Float weight1 = m_columnWeights[vrl1];
            m_columnWeights[vrl1] = 0.0f;
            uint32_t vrl2 = m_vrls[weightedSample(m_columnWeights, sampler, NULL, begin, end, &m_vrls)];
            m_columnWeights[vrl1] = weight1; // restore norm of first sample

            /* These two (normalized) VRLs span a line in the 'column space' */
            using namespace boost::numeric::ublas;
            vector<Float> direction(numGatherPoints(m_M)); // direction on which to project
            vector<Float> vrl1col = extractMean<Float>(m_M, vrl1);
            Float vrl1len = norm_2(vrl1col);
            vector<Float> vrl2col = extractMean<Float>(m_M, vrl2);
            Float vrl2len = norm_2(vrl2col);
            vector<Float> diff = vrl2col - vrl1col;
            Float diffLen = norm_2(diff);
            if (vrl1len != 0 && vrl2len != 0 && diffLen != 0) {
                direction = diff / diffLen;
            } else {
                // Sample direction uniformly on the surface of the n-sphere
                do {
                    for (size_t i = 0; i < direction.size(); i++) {
                        direction[i] = warp::squareToStdNormal(sampler->next2D()).x; // bit wasteful on rng
                    }
                } while (norm_2(direction) == 0);
                direction = direction / norm_2(direction);
            }

            std::vector<std::pair<Float, uint32_t> > proj(end - begin); // projections along the above direction along with their vrl
            for (uint32_t j = begin; j < end; j++) {
                uint32_t vrl = m_vrls[j];
                vector<Float> normalizedCol = extractMean<Float>(m_M, vrl);
                Float projection;
                if (norm_2(normalizedCol) == 0) {
                    //SLog(EWarn, "Encountered a zero norm VRL in cluster!");
                    //projection = 2*sampler->next1D() - 1; // not correct distribution
                    projection = 0;
                } else {
                    normalizedCol /= norm_2(normalizedCol);
                    projection = inner_prod(direction, normalizedCol);
                }

                proj[j - begin] = std::pair<Float, uint32_t>(projection, vrl);
            }
            std::sort(proj.begin(), proj.end());

            /* Put the vrls in m_vrls in the sorted order */
            for (uint32_t j = begin; j < end; j++) {
                m_vrls[j] = proj[j - begin].second;
            }

            /* Find optimal split that minimizes total variance */
            std::vector<std::pair<Float,Float> > variancesFromStart(clusterSize), variancesFromEnd(clusterSize);
            std::pair<Float,Float> v1 = calculateClusterVariance(m_M, m_columnWeights, m_localityWeights,
                    m_vrls.begin() + begin,
                    m_vrls.begin() + end,
                    &variancesFromStart);
            std::pair<Float,Float> v2 = calculateClusterVariance(m_M, m_columnWeights, m_localityWeights,
                    m_vrls.rend() - end,
                    m_vrls.rend() - begin,
                    &variancesFromEnd);
            if ((v1.first == 0 && v2.first != 0) || (v1.first != 0 && v2.first == 0) ||
                    (v1.first != 0 && v2.first !=0 && fabs((v1.first - v2.first)/(v1.first + v2.first)) > 1e-3))
                SLog(EWarn, "Inconsistency in forward/reverse subsampled variance calculation! %e vs %e", v1.first, v2.first);
            if ((v1.second == 0 && v2.second != 0) || (v1.second != 0 && v2.second == 0) ||
                    (v1.second != 0 && v2.second !=0 && fabs((v1.second - v2.second)/(v1.second + v2.second)) > 1e-3))
                SLog(EWarn, "Inconsistency in forward/reverse vrl integral variance calculation! %e vs %e", v1.second, v2.second);
            Float bestVariance = std::numeric_limits<Float>::infinity();
            uint32_t bestIndex = UINT32_T_MAX;
            for (uint32_t i = 1; i < clusterSize; ++i) { // i is the *beginning* of the second cluster
                std::pair<Float, Float> varHead = variancesFromStart[i - 1];
                std::pair<Float, Float> varTail = variancesFromEnd[clusterSize - 1 - i];
                Float thisVar = varHead.first + varHead.second  +  varTail.first + varTail.second;
                //                  variance for 0 ... i-1      +  variance for i ... clusterSize-1
                if (thisVar < bestVariance) {
                    bestVariance = thisVar;
                    bestIndex = i;
                }
            }
            if (bestIndex == UINT32_T_MAX)
                SLog(EError, "Couldn't find best splitting index!");
            uint32_t splitIndex = begin + bestIndex;

            /* Add the new sub-clusters to the queue */
            addCluster(begin, splitIndex, variancesFromStart[bestIndex - 1]);
            addCluster(splitIndex, end, variancesFromEnd[clusterSize - 1 - bestIndex]);
            return true;
        }

        void makeRefinementSnapshot() {
            /* m_vrls are only shuffled in-place 'on a smaller granularity'
             * during refinement -> no need to back that up */
            m_shad_clusterUndersamplingVariance  = m_clusterUndersamplingVariance;
            m_shad_clusterVrlIntegrationVariance = m_clusterVrlIntegrationVariance;
            m_shad_pq                            = m_pq;
            m_shad_singletonClusters             = m_singletonClusters;
        }
        void restoreSnapshot() {
            m_clusterUndersamplingVariance  = m_shad_clusterUndersamplingVariance;
            m_clusterVrlIntegrationVariance = m_shad_clusterVrlIntegrationVariance;
            m_pq                            = m_shad_pq;
            m_singletonClusters             = m_shad_singletonClusters;
        }

        std::vector<uint32_t> m_vrls; /// vrl indices (these get swapped around to make contiguous subarrays per cluster)
        std::vector<Float> m_columnWeights; /// column weights (indexed by vrl) [essentially const]
        const std::vector< std::vector<VrlContribution> > &m_M; /// matrix used to calculate cluster cost (typically a local matrix L_i), indices [some_slice_or_pixel_index][vrl]
        const boost::numeric::ublas::vector<double> &m_localityWeights; /// matrix used to calculate cluster cost (typically a local matrix L_i), indices [some_slice_or_pixel_index][vrl]
        Float m_vrlTracingVariance; /// expected variance in Li due to imperfect importance sampling of the VRLs in the VRL tracer
        Float m_unclusteredVrlIntegrationVariance; /// expected variance in Li from the contributions of all VRL integrals (i.e. unclustered case)
        Float m_clusterUndersamplingVariance; /// expected variance in Li due to subsampling the VRL samples through the clustering
        Float m_clusterVrlIntegrationVariance; /// expected variance in Li due to integrating the subsampled VRLs with the eye rays
        Float m_pixelUndersampling; /// Undersampling in the pixels during preprocessing -- needed for convergenceConstant() (value is <= 1)
        Float m_depthCorrection;

        boost::heap::priority_queue<ClusterNode> m_pq;
        std::list<uint32_t> m_singletonClusters; /// Singleton clusters are not stored in m_pq but separately here -> this directly holds the vrl (no indirection through m_vrls, here!)

        // shadow variables for refinement snapshots
        Float m_shad_clusterUndersamplingVariance;
        Float m_shad_clusterVrlIntegrationVariance;
        boost::heap::priority_queue<ClusterNode> m_shad_pq;
        std::list<uint32_t> m_shad_singletonClusters;
    };

    class ClusterRefiner : public Thread {
    public:
        ClusterRefiner(int id, int workerCount,
                const std::vector< std::vector<uint32_t> > &vrlsPerCluster,
                const std::vector< std::vector< std::vector<VrlContribution> > > &R,
                const std::vector<uint32_t> &fallBackCluster,
                const std::vector<Float> &fallBackWeights,
                const Preprocessor &preprocessor,
                Sampler *sampler) :
                    Thread(formatString("Ref%i", id)),
                    m_vrlsPerCluster(vrlsPerCluster),
                    m_R(R),
                    m_fallBackCluster(fallBackCluster),
                    m_fallBackWeights(fallBackWeights),
                    m_prep(preprocessor) {
            setCritical(true);
            m_sampler = sampler->clone();
            size_t numSlices = m_R.size();
            m_minIndex = (  id   * numSlices)/workerCount;
            m_maxIndex = ((id+1) * numSlices)/workerCount;
        }
        void run() {
            m_reprVrls.resize(m_maxIndex - m_minIndex);
            m_weights.resize(m_maxIndex - m_minIndex);
            for (size_t i = m_minIndex; i < m_maxIndex; i++) {
                m_prep.refineSlice(i, m_vrlsPerCluster,
                        m_fallBackCluster, m_fallBackWeights,
                        m_reprVrls[i - m_minIndex],
                        m_weights[i - m_minIndex],
                        m_R, m_sampler);
            }
        }
        void merge(std::vector< std::vector<uint32_t> > &reprVrls,
                std::vector< std::vector<Float> > &weights) {
            for (size_t loc_i = 0; loc_i < m_maxIndex - m_minIndex; loc_i++) {
                size_t glob_i = loc_i + m_minIndex;
                reprVrls[glob_i] = m_reprVrls[loc_i];
                weights[glob_i] = m_weights[loc_i];
            }
        }
    private:
        const std::vector< std::vector<uint32_t> > &m_vrlsPerCluster;
        const std::vector< std::vector< std::vector<VrlContribution> > > &m_R;
        ref<Sampler> m_sampler;
        size_t m_minIndex, m_maxIndex;
        const std::vector<uint32_t> &m_fallBackCluster;
        const std::vector<Float> &m_fallBackWeights;
        const Preprocessor &m_prep;
        // results:
        std::vector< std::vector<uint32_t> > m_reprVrls;
        std::vector< std::vector<Float> > m_weights;
    };



    /// summedLuminance is the combined luminance of the current and neighbouring slices' total luminance
    // TODO: summedLuminance with some (inverse [squared?]) distance weights (distance of gather point), to account for lesser importance?
    inline std::vector< std::vector<VrlContribution> > getLocalMatrix(
            const std::vector< std::vector< std::vector<VrlContribution> > > R,
            boost::numeric::ublas::vector<double> &localityWeights,
            uint32_t i) const {
        std::vector< std::vector<VrlContribution> > result; // TODO: use references instead of copies of columns? (but if we transpose R later on, we'll need to explicitly chop and cut anyway)

        result.insert(result.end(), R[i].begin(), R[i].end());

        if (m_neighbourWeight <= 0) {
            // Don't add neighbouring slices
            localityWeights.resize(R[i].size());
            for (size_t k = 0; k < R[i].size(); k++) {
                localityWeights[k] = 1.0 / R[i].size();
            }
            return result;
        }

        // Add the neighbours
        std::vector<Float> neighbourWeights(m_localities[i].size());
        Float summedNeighbourWeight = 0;
        int j = 0;
        for (std::set<Loc>::const_iterator it=m_localities[i].begin(); it!=m_localities[i].end(); ++it) {
            result.insert(result.end(), R[(*it).first].begin(), R[(*it).first].end());
            neighbourWeights[j] = 1.0 / (*it).second; // TODO inverse power law? Something else?
            summedNeighbourWeight += neighbourWeights[j];
            j++;
        }

        Float sliceWeight = summedNeighbourWeight * (1 - m_neighbourWeight) / m_neighbourWeight;
        Float normalization = 1 / (sliceWeight + summedNeighbourWeight);

        std::vector<double> locWghts;
        for (size_t k = 0; k < R[i].size(); k++) {
            locWghts.push_back(sliceWeight * normalization / R[i].size());
        }
        j = 0;
        for (std::set<Loc>::const_iterator it=m_localities[i].begin(); it!=m_localities[i].end(); ++it) {
            for (size_t k = 0; k < R[(*it).first].size(); k++) {
                locWghts.push_back(neighbourWeights[j] * normalization / R[(*it).first].size());
            }
            j++;
        }
        // TODO more efficient
        localityWeights.resize(locWghts.size());
        for (size_t k = 0; k < locWghts.size(); k++) {
            localityWeights[k] = locWghts[k];
        }
        return result;
    }

    /**
     * Find a global clustering.
     *
     * If there are VRLs that have no contribution in R, then these are all
     * clustered in one extra, separate cluster.
     *
     * columnWeights: weight given to each vrl to be sampled within a
     * cluster as representative (not normalized)
     */
    void cluster(
            const std::vector< std::vector<VrlContribution> > &R,
            std::vector< std::vector<uint32_t> > &vrlsPerCluster,
            Sampler *sampler) {

        /* Select VRLs with non-zero luminance. Zero-luminance VRLs are
         * treated separately, as we can't compute meaningful distance
         * metrics for them. */
        uint32_t amountOfVrls = R[0].size();
        std::vector<uint32_t> nonZeroVrls;
        std::vector<uint32_t> zeroVrls;
        for (uint32_t i = 0; i < amountOfVrls; i++) {
            if (totalVrlContribution(R, i) != 0) {
                nonZeroVrls.push_back(i);
            } else {
                zeroVrls.push_back(i);
            }
        }
        uint32_t amountOfNonZeroVrls = nonZeroVrls.size();
        uint32_t amountOfZeroVrls = zeroVrls.size();

        /* Perform the actual clustering of the non-zero VRLs */
        vrlsPerCluster.clear();
        if (amountOfNonZeroVrls != 0) {
            if (m_globalCluster) {
                clusterRefinement(R, nonZeroVrls, vrlsPerCluster, sampler);
            } else {
                /* Global clustering is disabled */
                vrlsPerCluster.resize(1);
                vrlsPerCluster[0] = nonZeroVrls;
            }
        }

        /* Print some cluster statistics (of the non-zero VRLs) */
        Float averageVrlsPerSlice = ((Float) amountOfNonZeroVrls) / vrlsPerCluster.size();
        Float var = 0;
        for (uint32_t s = 0; s < vrlsPerCluster.size(); s++) { //slice
            Float dev = vrlsPerCluster[s].size() - averageVrlsPerSlice;
            var += dev*dev;
        }
        var /= vrlsPerCluster.size();
        Log(EInfo, "Average %f +/- %f non-zero VRLs in %d global clusters",
                averageVrlsPerSlice, sqrt(var), vrlsPerCluster.size());

        /* Handle the zero-luminance VRLs */
        if (amountOfZeroVrls != 0) {
#if 1
            // put zero-luminance VRLs all in a single cluster
            Log(EWarn, "Found %d zero-luminance VRLs, creating extra cluster for them", amountOfZeroVrls);
            vrlsPerCluster.push_back(zeroVrls);
#else
            // put zero-luminance VRLs in singleton clusters (more safe, but slow)
            Log(EWarn, "Found %d zero-luminance VRLs, creating singleton clusters for them", amountOfZeroVrls);
            for (uint32_t i = 0; i < amountOfZeroVrls; i++) {
                std::vector<uint32_t> singletonCluster(1);
                singleTonCluster[0] = zeroVrls[i];
                vrlsPerCluster.push_back(singleTonCluster);
            }
#endif
        }
    }
    void clusterRefinement( /// Global cluster through refinement
            const std::vector< std::vector<VrlContribution> > &R,
            const std::vector<uint32_t> &nonZeroVrls,
            std::vector< std::vector<uint32_t> > &vrlsPerCluster,
            Sampler *sampler) {
        // Put all vrls in one cluster and refine it
        std::vector< std::vector<uint32_t> > initialVrlsPerCluster(1);
        initialVrlsPerCluster[0] = nonZeroVrls;
        boost::numeric::ublas::vector<double> dummyLocalityWeights(numGatherPoints(R), 1.0/numGatherPoints(R));
        Clustering clustering(initialVrlsPerCluster, R, dummyLocalityWeights, m_globalPixelUndersampling);
        if (!clustering.refine(m_globalUndersampling, sampler))
            SLog(EError, "Couldn't refine global clustering!");
        vrlsPerCluster = clustering.getVrlsPerCluster();
    }

    /// Compute the norm of column c in the matrix R
    static inline Float computeNorm(const std::vector< std::vector< Float > > &R, size_t c) {
        Float accum = 0;
        for (size_t r = 0; r < R.size(); r++) {
            Float val = R[r][c];
            accum += val*val;
        }
        return math::safe_sqrt(accum);
    }

    /// Compute the sum of column c in the matrix M
    static inline Float columnSum(const std::vector< std::vector< Float > > &M,
            size_t c, size_t begin = SIZE_T_MAX, size_t end = SIZE_T_MAX) {
        begin = begin != SIZE_T_MAX ? begin : 0;
        end   = end   != SIZE_T_MAX ? end   : M.size();
        Float sum = 0;
        for (size_t r = begin; r < end; r++) {
            sum += M[r][c];
        }
        return sum;
    }
    /// Compute the sum of the VRL means of column c in the matrix M
    static inline Float totalVrlContribution(const std::vector< std::vector< VrlContribution > > &M,
            size_t c, size_t begin = SIZE_T_MAX, size_t end = SIZE_T_MAX) {
        begin = begin != SIZE_T_MAX ? begin : 0;
        end   = end   != SIZE_T_MAX ? end   : M.size();
        Float sum = 0;
        for (size_t r = begin; r < end; r++) {
            sum += M[r][c].mean;
        }
        return sum;
    }

    template<typename FloatType>
    static inline boost::numeric::ublas::vector<FloatType> extractMean(const std::vector<std::vector<VrlContribution> >& R, size_t i) {
        boost::numeric::ublas::vector<FloatType> column(R.size());
        for (size_t j = 0; j < R.size(); j++) {
            column[j] = R[j][i].mean;
        }
        return column;
    }
    template<typename FloatType>
    static inline void extractMean(const std::vector<std::vector<VrlContribution> >& R, size_t i, boost::numeric::ublas::vector<FloatType> &column) {
        column.resize(R.size());
        for (size_t j = 0; j < R.size(); j++) {
            column[j] = R[j][i].mean;
        }
    }

    template<typename FloatType>
    static inline boost::numeric::ublas::vector<FloatType> extractVar(const std::vector<std::vector<VrlContribution> >& R, size_t i) {
        boost::numeric::ublas::vector<FloatType> column(R.size());
        for (size_t j = 0; j < R.size(); j++) {
            column[j] = R[j][i].var;
        }
        return column;
    }
    template<typename FloatType>
    static inline void extractVar(const std::vector<std::vector<VrlContribution> >& R, size_t i, boost::numeric::ublas::vector<FloatType> &column) {
        column.resize(R.size());
        for (size_t j = 0; j < R.size(); j++) {
            column[j] = R[j][i].var;
        }
    }

    /**
     * Calculate the weight associated with a vrl (a column in M), this is
     * the weight of it being sampled as representative in a set of vrls
     * that constitute a cluster.
     * For safety reasons, a fraction of the total average weight is added
     * to all weights to protect zero-weight vrls. */
    static void calculateColumnWeigths(
            const std::vector< std::vector< VrlContribution > > &M,
            const boost::numeric::ublas::vector<double> &localityWeights,
            std::vector<Float> &columnWeights,
            Float safetyFraction = 1e-2) {
        columnWeights.resize(numVrls(M));
        for (size_t vrl = 0; vrl < numVrls(M); vrl++) {
            //columnWeights[vrl] = columnSum(M, vrl); // simple summed luminance
            //columnWeights[vrl] = computeNorm(M, vrl); // norm as in original MRCS paper
            boost::numeric::ublas::vector<double> mean = extractMean<double>(M, vrl);
            boost::numeric::ublas::vector<double> var  = extractVar<double>(M, vrl);
            boost::numeric::ublas::vector<double> x = element_prod(mean,mean) + var;
            columnWeights[vrl] = math::safe_sqrt(inner_prod(localityWeights, x)); // weighted RMS
            if (!std::isfinite(columnWeights[vrl])) {
                SLog(EError, "Invalid calculated average column weight %f", columnWeights[vrl]);
            }
        }
        Float averageWeight = std::accumulate(columnWeights.begin(), columnWeights.end(), 0.0f) / numVrls(M);
        if (averageWeight == 0)
            averageWeight = 1.0; // all columns have zero weight: force uniform sampling
        for (size_t vrl = 0; vrl < numVrls(M); vrl++) {
            columnWeights[vrl] += averageWeight * safetyFraction;
        }
    }

    /**
     * Compute the variance of the sum of all (unweighted, unclustered) VRL
     * contributions, i.e. the variance that would be expected when simply
     * summing all VRLs.
     *
     * This variance has a part tracerVariance that is due to the VRL
     * tracer, and a part vrlIntegrationVariance that is due to the VRL -
     * eye ray integrals.
     *
     * Each VRL is independent (in the full set), so the vrl tracer
     * variance is numVrls * Var(<single VRL sample>)
     */
    template<typename VrlIterator>
    static void calculateUnclusteredVariance(const std::vector< std::vector< VrlContribution > > &R,
            const boost::numeric::ublas::vector<double> &localityWeights,
            VrlIterator begin, VrlIterator end,
            Float &tracerVariance, Float &vrlIntegrationVariance) {
        using namespace boost::numeric::ublas;
        size_t n = 0;
        vector<double> mean(R.size(), 0);
        vector<double> M2(R.size(), 0); // squared differences
        vector<double> summedVars(R.size(), 0); // sum of inherent variances due to VRL integration
        for (VrlIterator it = begin; it != end; ++it) {
            n++;
            summedVars += extractVar<double>(R, *it);
            vector<double> x = extractMean<double>(R, *it);
            vector<double> delta = x - mean;
            mean += delta / n;
            M2 += element_prod(delta, x - mean);
        }
        if (n <= 1)
            SLog(EError, "Need at least 2 VRLs to estimate variance, but got %d!", n);
        vrlIntegrationVariance = inner_prod(localityWeights, summedVars);
        // note: M2/n == variance of single vrl; M2 == variance of sum of all vrls
        tracerVariance = inner_prod(localityWeights, M2) - vrlIntegrationVariance;
        if (norm_1(mean) == 0) {
            SLog(EDebug, "unclusteredVariance(): zero luminance set of vrls! (%d vrls)", n);
        }
    }

    /**
     * Compute the extra variance due to the subsampling from the
     * cluster i.e. due to only sampling one VRL within the range [begin,
     * end).
     *
     * The returned pair is:
     * (undersamplingOfVRLsVar, undersampledVRLintegrationVar)
     */
    template<typename VrlIterator>
    static std::pair<Float, Float> calculateClusterVariance(
            const std::vector< std::vector< VrlContribution > > &R,
            const std::vector<Float> &columnWeights,
            const boost::numeric::ublas::vector<double> &localityWeights,
            VrlIterator begin, VrlIterator end,
            std::vector<std::pair<Float,Float> > *incrementalVar = NULL) {
        if (begin == end)
            Log(EError, "taking variance of empty cluster!");

        using namespace boost::numeric::ublas;
        vector<double> x;
        vector<double> sum(R.size(), 0); // sum of columns (VRL means)
        double weightSum = 0;
        vector<double> M(R.size(), 0); // squared differences (VRL means)
        vector<double> sumVars(R.size(), 0); // sum of individual VRL variances, scaled by 1/weight; needs to be scaled with weightSum at the end: VRL integration variance is sum_i=1^n ( (sum_k=1^n w_k) / w_i * \xi_i^2 )
        int n = 0;
        for (VrlIterator it = begin; it != end; ++it) {
            uint32_t vrl = *it;

            // Accumulate variance due to cluster undersampling
            extractMean(R, vrl, x);
            double weight = columnWeights[vrl];
            if (!std::isfinite(weight) || weight <= 0) {
                SLog(EError, "Invalid weight in calculateClusterVariance(): %f, vrl %d", weight, vrl);
            }
            double newWeightSum = weightSum + weight;
            vector<double> newSum = sum + x;
            vector<double> tmpVec = weight*sum - weightSum*x;
            if (n > 0) {
                M = ((newWeightSum*newWeightSum)/(weightSum*weightSum)) * M
                        + (1.0/weight + 1.0/weightSum) * element_prod(tmpVec,tmpVec);
            }

            // Accumulate variance due to VRL integration
            sumVars += extractVar<double>(R, vrl) / weight;

            sum = newSum;
            weightSum = newWeightSum;

            if (incrementalVar) {
                if (n == 0) {
                    incrementalVar->at(n).first  = 0;
                    incrementalVar->at(n).second = inner_prod(localityWeights, sumVars*weightSum);
                } else {
                    incrementalVar->at(n).first  = inner_prod(localityWeights, M/weightSum);
                    incrementalVar->at(n).second = inner_prod(localityWeights, sumVars*weightSum);
                }
            }

            n++;
        }

        std::pair<Float, Float> result;
        result.first  = inner_prod(localityWeights, M/weightSum);
        result.second = inner_prod(localityWeights, sumVars*weightSum);
        if (!std::isfinite(result.first) || result.first < 0)
            SLog(EError, "invalid undersampled VRL cluster variance: %e", result.first);
        if (!std::isfinite(result.second) || result.second < 0)
            SLog(EError, "invalid undersampled VRL integration cluster variance: %e", result.second);

        return result;
    }


//////////////////////////////////////////////////////////////////////////
/*                    FIRST STEP: SLICE CREATION                        */
//////////////////////////////////////////////////////////////////////////

    /**
     * Returns a vector that maps each pixel in the image plane to its
     * slice index, or -1 if the gather point is at infinity. */
    std::vector<uint32_t> buildSlices(const Scene *scene) {
        //find a "gather point" for every pixel
        Vector2i imageSize = scene->getSensor()->getFilm()->getSize(); // film resolution

        std::vector<Point2> pixels;
        std::vector<Point> gatherPoints; // for each pixel: the hitpoint of an eye ray, or "(nan)^3" if nothing was hit
        std::vector<Point> directions; // for each pixel: a (scaled) surface normal or (nan)^3 [actually a vector, but point here for uniformity down the road]
        Float directionScale = distance(scene->getAABB().min, scene->getAABB().max) / 8 * m_sliceCurvatureFactor; // loosely based on Multidim lightcuts; Walter et al (2006)

        // fill pixels and gatherPoints
        for (int i = 0; i < imageSize.x; i++) {
            for (int j = 0; j < imageSize.y; j++) {
                Ray ray;
                Intersection its;
                Point2 pixel(i+0.5f, j+0.5f);
                pixels.push_back(pixel);
                scene->getSensor()->sampleRay(ray, pixel, Point2(0.0f), 0.0f); // TODO scenes that need aperture/time sample...

                if (scene->rayIntersect(ray, its)) {
                    // If we encounter a null interaction: try to keep going until we hit something more substantial
                    // TODO: or keep going until we find first medium?
                    const BSDF *bsdf = its.getBSDF();
                    if (bsdf == NULL)
                        SLog(EError, "expected non-null BSDF");

                    Point gatherPoint;
                    Vector direction;
                    while (true) {
                        gatherPoint = its.p;
                        direction = its.shFrame.n;
                        if (!(bsdf->getType() & BSDF::ENull)) {
                            break;
                        }
                        ray = Ray(ray, its.t + Epsilon, ray.maxt);
                        if (!scene->rayIntersect(ray, its))
                            break;
                        bsdf = its.getBSDF();
                        if (bsdf == NULL)
                            SLog(EError, "expected non-null BSDF");
                    }
                    gatherPoints.push_back(gatherPoint);
                    directions.push_back(Point(directionScale * direction));
                } else {
                    gatherPoints.push_back(
                            Point(std::numeric_limits<Float>::quiet_NaN()));
                    directions.push_back(
                            Point(std::numeric_limits<Float>::quiet_NaN()));
                }
            }
        }

        // stopping conditions
        uint32_t numPixels = imageSize.x * imageSize.y;
        int thresholdCount = std::max((uint32_t) (0.5 + (Float) numPixels / m_targetNumSlices), (uint32_t) 1);
        Log(EInfo, "Slices threshold count: %d (target num slices = %d)", thresholdCount, m_targetNumSlices);

        std::vector<uint32_t> pixelToSlice = getSlices(pixels, gatherPoints, directions);

        uint32_t numSlices = m_slices.size();
        Log(EInfo, "Built %d slices for %d pixels",
                numSlices, numPixels);

        return pixelToSlice;
    }

    /**
     * Returns a vector of length gatherPoints.size() that maps each
     * gather point to its slice index, or UINT32_T_MAX if the gather point is at
     * infinity */
    // TODO: for infinite gather points -> slice based on distances in image plane? (but we don't render infinite vrls yet, anyway...)
    std::vector<uint32_t> getSlices(const std::vector<Point2> &pixels, const std::vector<Point> &gatherPoints, const std::vector<Point> &directions) {
        std::vector<uint32_t> gatherPointToSlice(gatherPoints.size(), UINT32_T_MAX);
        std::vector<uint32_t> indices(gatherPoints.size());
        for (uint32_t i = 0; i < indices.size(); i++)
            indices[i] = i;

        // bring all indices of infinite gather points to the front, so they can be skipped
        // first find the first good point
        uint32_t firstGoodPointInd = 0;
        while (!gatherPoints[firstGoodPointInd].isFinite()) {
            firstGoodPointInd++;
            if (firstGoodPointInd >= gatherPoints.size())
                break;
        }
        // then bring the other bad points to the front by swapping them with the first good point
        for (uint32_t i = firstGoodPointInd + 1; i < indices.size(); i++) {
            if (!gatherPoints[i].isFinite()) {
                indices[i] = indices[firstGoodPointInd];
                indices[firstGoodPointInd] = i;
                firstGoodPointInd++;
            }
        }

        m_slices.clear();
        getSlicesPQ(pixels, gatherPoints, directions, indices, gatherPointToSlice, firstGoodPointInd, indices.size());

        return gatherPointToSlice;
    }

    /// 6D distance
    static Float sliceDistance(
            const Point &pos1, const Point &dir1,
            const Point &pos2, const Point &dir2) {
        return sqrt(distanceSquared(pos1, pos2) + distanceSquared(dir1, dir2));
    }
    Float sliceDistance(uint32_t i, uint32_t j) {
        return sliceDistance(
                m_slices[i].positionCentroid, m_slices[i].directionCentroid,
                m_slices[j].positionCentroid, m_slices[j].directionCentroid);
    }

    void buildLocalities() {
        uint32_t numSlices = m_slices.size();
        m_localities.resize(numSlices);

        if (numSlices <= m_neighbourCount) {
            Log(EWarn, "neighbour count (%d) not smaller than number of slices (%d)",
                    m_neighbourCount, numSlices);
            // every slice is in the neighbourhood of every other one
            for (uint32_t i = 0; i < m_localities.size(); i++) {
                for (uint32_t j = 0; j < m_localities.size(); j++) {
                    if (i != j) {
                        m_localities[i].insert(std::pair<uint32_t, Float>(
                                j, sliceDistance(i, j)));
                    }
                }
            }
            return;
        }

        Float distances[m_neighbourCount]; // will hold the m_neighbourCount smallest distances to ...
        uint32_t indices[m_neighbourCount]; // ... the slices with these corresponding slice indices

        uint32_t maxInd = 0;
        for (uint32_t i = 0; i < numSlices; i++) {
            for (uint32_t j = 0; j < m_neighbourCount; j++) {
                distances[j] = std::numeric_limits<Float>::infinity();
            }

            // find the slices with smallest distance to the current one
            for (uint32_t j = 0; j < numSlices; j++) {
                if (i == j)
                    continue;

                Float dist = sliceDistance(i, j);
                if (dist < distances[maxInd]) {
                    // smaller distance found -> replace the previous max with this entry
                    distances[maxInd] = dist;
                    indices[maxInd] = j;

                    // find new maximum distance index
                    for (uint32_t k = 0; k < m_neighbourCount; k++) {
                        if (distances[k] > distances[maxInd])
                            maxInd = k;
                    }
                }
            }

            // store the indices of the neighbouring slices
            for (uint32_t x = 0; x < m_neighbourCount; x++) {
                m_localities[i].insert(Loc(indices[x], distances[x]));
            }
        }
    }

    struct SliceNode {
        uint32_t minInd, maxInd;
        Float distance;
        unsigned char dim;
        Float split;
        Point positionCentroid, directionCentroid;
        SliceNode(uint32_t minI, uint32_t maxI,
            const std::vector<Point> &positions,
            const std::vector<Point> &directions,
            const std::vector<uint32_t> &indices)
                : minInd(minI), maxInd(maxI) {
            if (minInd >= maxInd) {
                SLog(EError, "trying to create empty SliceNode");
            }
            if (minInd + 1 == maxInd) {
                // singleton slice
                distance = 0;
                dim = 0;
                split = std::numeric_limits<Float>::quiet_NaN();
                positionCentroid  = Point(std::numeric_limits<Float>::quiet_NaN());
                directionCentroid = Point(std::numeric_limits<Float>::quiet_NaN());
                return;
            }

            /* 'proper' slice -> determine bounding box of gather points to
             * determine distance, split and centroid */
            Float posInf = std::numeric_limits<Float>::infinity();
            Float negInf = -1 * posInf;
            Point maxPos(negInf); Point minPos(posInf);
            Point maxDir(negInf); Point minDir(posInf);
            for (size_t i = minInd; i < maxInd; i++) {
                const Point position = positions[indices[i]];
                updateMin(position, minPos);
                updateMax(position, maxPos);
                const Point direction = directions[indices[i]];
                updateMin(direction, minDir);
                updateMax(direction, maxDir);
            }

            distance = sliceDistance(minPos, minDir, maxPos, maxDir);
            findSplit(maxPos, minPos, maxDir, minDir, dim, split);

            positionCentroid  = minPos + 0.5 * (maxPos - minPos);
            directionCentroid = minDir + 0.5 * (maxDir - minDir);
        }
        const bool operator < (const SliceNode &other) const { return distance < other.distance; }
    };

    // pixels: array of 2D pixel-plane samples
    // positions: array of corresponding 3D gather points (must be valid!)
    // directions: array of corresponding 3D direction 'points' (must be valid!)
    // indices: mapping towards indices in the above arrays; this array gets partitioned
    // minInd: index in 'indices': partition elements in 'indices' with index equal or greater than this
    // maxInd: index in 'indices': partition elements in 'indices' with index strictly less than this
    void getSlicesPQ (
            const std::vector<Point2> &pixels,
            const std::vector<Point> &positions,
            const std::vector<Point> &directions,
            std::vector<uint32_t> &indices,
            std::vector<uint32_t> &gatherPointToSlice,
            size_t minInd, size_t maxInd) {

        if (maxInd <= minInd)
            return;

        boost::heap::priority_queue<SliceNode> pq;
        pq.push(SliceNode(minInd, maxInd, positions, directions, indices));

        // split largest distance slice until we have our target number of slices (or until each slice is a singleton gatherpoint)
        while (pq.size() < m_targetNumSlices && pq.top().distance > 0) {
            SliceNode sn = pq.top();
            pq.pop();

            // classic partitioning scheme; small left, large right
            size_t lo = sn.minInd;
            size_t hi = sn.maxInd - 1;
            size_t i = lo - 1;
            size_t j = hi + 1;
            while (true) {
                // run towards right until we find a large element
                while (true) {
                    i++;
                    // stop if 'i' larger than split
                    if (isLarger(positions[indices[i]], directions[indices[i]], sn.dim, sn.split) || i == hi)
                        break;
                }
                while (true) {
                    j--;
                    // stop if 'j' smaller than split
                    if (!isLarger(positions[indices[j]], directions[indices[j]], sn.dim, sn.split) || j == lo)
                        break;
                }
                if (i >= j)
                    break;
                // swap
                uint32_t temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }

            pq.push(SliceNode(sn.minInd, j+1, positions, directions, indices));
            pq.push(SliceNode(j+1, sn.maxInd, positions, directions, indices));
        }

        // save the slices
        for (boost::heap::priority_queue<SliceNode>::const_iterator iter = pq.begin(); iter != pq.end(); ++iter) {
            const SliceNode &sn = *iter;
            std::vector<GatherPoint> gatherPoints(sn.maxInd - sn.minInd);
            for (size_t i = sn.minInd; i < sn.maxInd; i++) {
                gatherPoints[i - sn.minInd] = GatherPoint(
                        pixels[indices[i]],
                        positions[indices[i]],
                        directions[indices[i]]);
            }
            m_slices.push_back(Slice(
                    gatherPoints, sn.positionCentroid, sn.directionCentroid));

            // update gatherPointToSlice with this new slice
            uint32_t slice = m_slices.size() - 1;
            for (size_t i = sn.minInd; i < sn.maxInd; i++) {
                gatherPointToSlice[indices[i]] = slice;
            }
        }
    }

    static inline bool isLarger(const Point &point, const Point &dir, int dim, Float split) {
        switch (dim) {
            case 0: return point.x > split;
            case 1: return point.y > split;
            case 2: return point.z > split;
            case 3: return dir.x > split;
            case 4: return dir.y > split;
            case 5: return dir.z > split;
            default: SLog(EError, "isLarger: invalid split dimension"); return false;
        }
    }

    static inline void findSplit(const Point &maxPos, const Point &minPos, const Point &maxDir, const Point &minDir, unsigned char &dim, Float &split) {
        unsigned char dimPos, dimDir;
        Float splitPos, splitDir, extPos, extDir;
        findSplitPoint(maxPos, minPos, dimPos, splitPos, extPos);
        findSplitPoint(maxDir, minDir, dimDir, splitDir, extDir);
        if (extPos == 0 && extDir == 0) {
            SLog(EError, "findSplit: min equal to max!\nmaxPos %s\nminPos %s\nmaxDir %s\nminDir %s",
                    maxPos.toString().c_str(), minPos.toString().c_str(),
                    maxDir.toString().c_str(), minDir.toString().c_str());
        }
        if (extPos > extDir) {
            dim = dimPos;
            split = splitPos;
        } else {
            dim = 3 + dimDir; // direction dimensions come 'after' positional dimensions
            split = splitDir;
        }
    }

    static inline void findSplitPoint(const Point &max, const Point &min, unsigned char &dim, Float &split, Float &extent) {
        Float diffx = max.x - min.x;
        Float diffy = max.y - min.y;
        Float diffz = max.z - min.z;

        if (diffx < 0 || diffy < 0 || diffz < 0) {
            SLog(EError, "findSplitPoint: min not smaller than max!");
        }
        if (diffx == 0 && diffy == 0 && diffz == 0) {
            extent = 0;
            dim = 0;
            split = std::numeric_limits<Float>::quiet_NaN();
            return;
        }

        if (diffx > diffy) {
            if (diffx > diffz) {
                dim = 0;
                split = min.x + 0.5*diffx;
                extent = diffx;
            } else {
                dim = 2;
                split = min.z + 0.5*diffz;
                extent = diffz;
            }
        } else { //diffx <= diffy
            if (diffy > diffz) {
                dim = 1;
                split = min.y + 0.5*diffy;
                extent = diffy;
            } else {
                dim = 2;
                split = min.z + 0.5*diffz;
                extent = diffz;
            }
        }
    }

    static inline void updateMax(const Point &newPoint, Point &max) {
        if (newPoint.x > max.x) max.x = newPoint.x;
        if (newPoint.y > max.y) max.y = newPoint.y;
        if (newPoint.z > max.z) max.z = newPoint.z;
    }

    static void updateMin(const Point &newPoint, Point &min) {
        if (newPoint.x < min.x) min.x = newPoint.x;
        if (newPoint.y < min.y) min.y = newPoint.y;
        if (newPoint.z < min.z) min.z = newPoint.z;
    }

    /// Samples and returns representative pixels for each slice
    const std::vector< std::vector<Point2> > sampleSliceMapping(
            Float targetPixelUndersampling, Sampler *sampler) {
        size_t numSlices = m_slices.size();
        std::vector<std::vector<Point2> > reprPixels(numSlices);
        m_sliceUndersampling.resize(numSlices);

        size_t totalNumPixels = 0;
        size_t totalNumReprPixels = 0;
        for (size_t i = 0; i < numSlices; i++) {
            reprPixels[i] = m_slices[i].sampleRepresentativePixels(
                    targetPixelUndersampling, sampler);
            m_sliceUndersampling[i] = ((Float) reprPixels[i].size()) / m_slices[i].numPixels();
            totalNumReprPixels += reprPixels[i].size();
            totalNumPixels += m_slices[i].numPixels();
        }
        buildLocalities();

        m_globalPixelUndersampling = ((Float) totalNumReprPixels) / totalNumPixels;
        SLog(EInfo, "Sampled %d representatives for %d valid gatherPoints (undersampling %f)",
                totalNumReprPixels, totalNumPixels,
                1 / m_globalPixelUndersampling);

        return reprPixels;
    }

    /**
     * Performs a weighted sample according to the given positive weights.
     * If all weights are zero, sample uniformly. If begin and end are
     * given, sample only indices in the range [begin ... end-1].
     * If indexIndirection is given, then this is used to translate indices
     * into the weight array, the returned index is an index in this array
     * (i.e. not yet translated) */
    static size_t weightedSample(
            const std::vector<Float> &weights, Sampler *sampler,
            Float *prob = NULL, size_t begin = SIZE_T_MAX, size_t end = SIZE_T_MAX,
            const std::vector<uint32_t> *indexIndirection = NULL) {
        begin = begin != SIZE_T_MAX ? begin : 0;
        end   = end   != SIZE_T_MAX ? end   : weights.size();
        if (begin >= end) {
            SLog(EError, "Trying to take weighted sample of empty set!");
        }
        if (end == begin + 1) {
            if (prob)
                *prob = 1;
            return begin;
        }

        Float weightSum = 0.0f;
        for (size_t i = begin; i < end; i++) {
            weightSum += weights[indexIndirection ? indexIndirection->at(i) : i];
        }

        Float probability;
        size_t ind;
        if (weightSum <= 0) {
            // uniform sampling
            do {
                ind = begin + sampler->next1D() * (end - begin);
            } while (ind >= end);
            probability = 1.0 / (end - begin);
        } else {
            Float alpha = sampler->next1D() * weightSum;
            Float accum = 0.0f;
            ind = begin; // initialize to keep compiler happy (should always get reassigned below)
            for (size_t i = begin; i < end; i++) {
                accum += weights[indexIndirection ? indexIndirection->at(i) : i];
                if (accum >= alpha) {
                    ind = i;
                    break;
                }
            }
            probability = weights[indexIndirection ? indexIndirection->at(ind) : ind] / weightSum;
        }

        if (prob) {
            *prob = probability;
        }
        return ind;
    }
    static uint32_t numVrls(const std::vector< std::vector< VrlContribution > > &M) {
        return M[0].size();
    }
    static uint32_t numVrls(const std::vector< std::vector< std::vector< VrlContribution > > > &M) {
        return M[0][0].size();
    }
    static uint32_t numGatherPoints(const std::vector< std::vector< VrlContribution > > &M) {
        return M.size();
    }

    MTS_DECLARE_CLASS()

private:
    uint32_t m_targetNumSlices;
    Float m_sliceCurvatureFactor;

    uint32_t m_neighbourCount;
    Float m_neighbourWeight;

    bool m_globalCluster;
    bool m_localRefinement;

    Float m_globalUndersampling;
    Float m_localUndersampling;
    Float m_fallBackUndersampling;

    Float m_depthCorrection;

    int m_workerCount;

    std::vector<Slice> m_slices;
    std::vector<Float> m_sliceUndersampling; /// The number of representative pixels that were sampled per slice by sampleSliceMapping(), divided by the total number of pixels for that slice
    Float m_globalPixelUndersampling; /// The total number of representative pixels divided by the full number of pixels (over all slices)

    /* Per slice: a list of slice indices that are close to it, including
     * the distance. Used to build L_i */
    typedef std::pair< uint32_t, Float > Loc;
    std::vector< std::set< Loc > > m_localities;
};
MTS_IMPLEMENT_CLASS(Preprocessor, false, Object);
MTS_NAMESPACE_END

