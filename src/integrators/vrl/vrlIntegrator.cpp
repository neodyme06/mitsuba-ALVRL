#include <mitsuba/core/plugin.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/render/scene.h>
#include "vrlTracer.h"
#include "Preprocessor.cpp"
#include <cmath>
#include <iomanip>
#include <boost/timer/timer.hpp>
using boost::timer::cpu_timer;
using boost::timer::cpu_times;

MTS_NAMESPACE_BEGIN

/// Class that holds info of a clustering of a vrlVector
// (just quick and dirty public wrapper over std vectors atm for network serialization)
class vrlClusterInfo : public SerializableObject {
public:
    inline vrlClusterInfo() {
        m_slices.clear();
        m_selectedVrls.clear();
        m_clusterWeight.clear();
        m_gcVrls.clear();
        m_gcWeight.clear();
        m_fallBackVrls.clear();
        m_fallBackWeight.clear();
    }

    vrlClusterInfo(Stream *stream, InstanceManager *manager) {
        m_slices.resize(stream->readULong());
        for (size_t i = 0; i < m_slices.size(); i++) {
            m_slices[i] = stream->readUInt();
        }
        m_selectedVrls.resize(stream->readULong());
        for (size_t i = 0; i < m_selectedVrls.size(); i++) {
            m_selectedVrls[i].resize(stream->readULong());
            for (size_t j = 0; j < m_selectedVrls[i].size(); j++) {
                m_selectedVrls[i][j] = stream->readUInt();
            }
        }
        m_clusterWeight.resize(stream->readULong());
        for (size_t i = 0; i < m_clusterWeight.size(); i++) {
            m_clusterWeight[i].resize(stream->readULong());
            for (size_t j = 0; j < m_clusterWeight[i].size(); j++) {
                m_clusterWeight[i][j] = stream->readFloat();
            }
        }
        m_gcVrls.resize(stream->readULong());
        for (size_t i = 0; i < m_gcVrls.size(); i++) {
            m_gcVrls[i] = stream->readUInt();
        }
        m_gcWeight.resize(stream->readULong());
        for (size_t i = 0; i < m_gcWeight.size(); i++) {
            m_gcWeight[i] = stream->readFloat();
        }
        m_fallBackWeight.resize(stream->readULong());
        for (size_t i = 0; i < m_fallBackWeight.size(); i++) {
            m_fallBackWeight[i] = stream->readUInt();
        }
        m_fallBackWeight.resize(stream->readULong());
        for (size_t i = 0; i < m_fallBackWeight.size(); i++) {
            m_fallBackWeight[i] = stream->readFloat();
        }
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        stream->writeULong(m_slices.size());
        for (size_t i = 0; i < m_slices.size(); i++) {
            stream->writeUInt(m_slices[i]);
        }
        stream->writeULong(m_selectedVrls.size());
        for (size_t i = 0; i < m_selectedVrls.size(); i++) {
            stream->writeULong(m_selectedVrls[i].size());
            for (size_t j = 0; j < m_selectedVrls[i].size(); j++) {
                stream->writeUInt(m_selectedVrls[i][j]);
            }
        }
        stream->writeULong(m_clusterWeight.size());
        for (size_t i = 0; i < m_clusterWeight.size(); i++) {
            stream->writeULong(m_clusterWeight[i].size());
            for (size_t j = 0; j < m_clusterWeight[i].size(); j++) {
                stream->writeFloat(m_clusterWeight[i][j]);
            }
        }
        stream->writeULong(m_gcVrls.size());
        for (size_t i = 0; i < m_gcVrls.size(); i++) {
            stream->writeUInt(m_gcVrls[i]);
        }
        stream->writeULong(m_gcWeight.size());
        for (size_t i = 0; i < m_gcWeight.size(); i++) {
            stream->writeFloat(m_gcWeight[i]);
        }
        stream->writeULong(m_fallBackVrls.size());
        for (size_t i = 0; i < m_fallBackVrls.size(); i++) {
            stream->writeUInt(m_fallBackVrls[i]);
        }
        stream->writeULong(m_fallBackWeight.size());
        for (size_t i = 0; i < m_fallBackWeight.size(); i++) {
            stream->writeFloat(m_fallBackWeight[i]);
        }
    }

    std::string toString() const {
        return "vrlClusterInfo"; // TODO
    }
    std::vector<uint32_t> m_slices;
    std::vector<std::vector<uint32_t> > m_selectedVrls;
    std::vector<std::vector<Float> > m_clusterWeight;
    std::vector<uint32_t> m_gcVrls;  /// global cluster representatives
    std::vector<Float> m_gcWeight; /// global cluster weights
    std::vector<uint32_t> m_fallBackVrls;  /// fall-back clustering representatives
    std::vector<Float> m_fallBackWeight; /// fall-back clustering weights

    MTS_DECLARE_CLASS()
};



static StatsCounter statsVrlsPreprocess("VRL integrator",
        "Number of integrated VRLs during preprocessing", ENumberValue);
static StatsCounter statsVrlsRender("VRL integrator",
        "Number of integrated VRLs during rendering", ENumberValue);

class vrlIntegrator : public ProgressiveMonteCarloIntegrator {
public:
    MTS_DECLARE_CLASS()

    vrlIntegrator(const Properties &props) : ProgressiveMonteCarloIntegrator(props) {
        if (props.hasProperty("nc"))
            Log(EError, "Neighbourcount is now called 'neighbourCount' "
                    "instead of 'nc'!");

        /* Short VRLs as opposed to infinitely long VRLs? (The latter are
         * not fully tested) */
        m_shortVrls = props.getBoolean("shortVrls", true);
        /* Target number of VRLs (for the current integration pass) */
        m_vrlTargetNum = props.getInteger("vrlTargetNum", 500);

        /* Maximum depth of particles traced to generate the VRLs. */
        m_maxParticleDepth = props.getInteger("maxParticleDepth", -1);
        /* Depth at which a nonzero Russian Roulette stopping probability
         * gets forced for specular chains. */
        m_specRRdepth = props.getInteger("specularForcedRRdepth", 100);
        /* Initial specular throughput for Russian Roulette decisions */
        m_initialSpecularThroughput = props.getFloat("initialSpecularThroughput", 20);

        /* Number of samples for VRL volume to volume transport. */
        m_volVolSamples = props.getInteger("volVolSamples", 2);
        if (m_volVolSamples != 0 && m_volVolSamples < 2)
            Log(EError, "Need at least 2 volVolSamples for variance "
                    "estimate, but received: %d", m_volVolSamples);
        /* Number of samples for VRL volume to surface transport. */
        m_volSurfSamples = props.getInteger("volSurfSamples", 2);
        if (m_volSurfSamples != 0 && m_volSurfSamples < 2)
            Log(EError, "Need at least 2 volSurfSamples for variance "
                    "estimate, but received: %d", m_volSurfSamples);

        /* Perform an initial global clustering (that will be used as a
         * starting point for later local refinement if requested)? */
        m_globalCluster = props.getBoolean("globalCluster", false);
        /* VRL undersampling for initial global cluster before per-slice
         * refinement (Positive number N for a '1 in N' undersampling, 1 to
         * disable an initial global clustering, -1 for adaptive
         * refinement). */
        m_globalUndersampling = props.getFloat("globalUndersampling", -1);

        /* Perform a local refinement/clustering of VRLs? */
        m_localRefinement = props.getBoolean("localRefinement", true);
        /* VRL undersampling of local clusters. Positive number N for a
         * fixed '1 in N' undersampling, -1 for adaptive splitting into the
         * optimal number of clusters. */
        m_localUndersampling = props.getFloat("localUndersampling", -1);
        /* VRL undersampling for gather points (pixels) that don't hit any
         * geometry, and for slices that did not receive aany cortributing
         * VRLs. */
        m_fallBackUndersampling = props.getFloat("fallBackUndersampling", 5);

        /* Target number of slices (each slice gets its own local VRL
         * clustering if 'localRefinement' is active) */
        m_targetNumSlices = props.getInteger("targetNumSlices", 100);
        /* Target pixel undersampling when sampling representative pixels within a slice. */
        m_targetPixelUndersampling = props.getFloat("targetPixelUndersampling", 64);
        /* How much geometric curvature to take into account when grouping
         * pixels into slices. */
        m_sliceCurvatureFactor = props.getFloat("sliceCurvatureFactor", 0.5);

        /* Use information of this many neighbouring slices when
         * determining VRL clustering for the current slice. */
        m_neighbourCount = props.getInteger("neighbourCount", 0);
        /* Weight of those neighbours */
        m_neighbourWeight = props.getFloat("neighbourWeight", 0.0);

        /* Used to make more sense with older code, now 1 is most logical. */
        m_Rsamples = props.getInteger("Rsamples", 1);
        /* Factor to modify the calculated ideal cluster splitting depth
         * for adaptive refinement. */
        m_depthCorrection = props.getFloat("depthCorrection", 1);

        m_numVrlFalseColor = props.getBoolean("numVrlFalseColor", false);
        m_slicesFalseColor = props.getBoolean("slicesFalseColor", false);
        m_convergenceFalseColor = props.getBoolean("convergenceFalseColor", false);

        /* Load VRLs from a file instead of generating them. */
        m_vrlFile = props.getString("vrlFile", "");
        m_vrls = NULL;

        m_ci = new vrlClusterInfo();
    }

    vrlIntegrator(Stream *stream, InstanceManager *manager)
    : ProgressiveMonteCarloIntegrator(stream, manager) {
        m_volVolSamples = stream->readInt();
        m_volSurfSamples = stream->readInt();
        m_globalCluster = stream->readBool();
        m_localRefinement = stream->readBool();
        m_specRRdepth = stream->readInt();
        m_initialSpecularThroughput = stream->readFloat();
        m_shortVrls = stream->readBool();

        m_vrlsID = m_ciID = 0;
    }

    /* TODO: [comment when cleaning up the code years later:] is this
     * actually working? (feels like it was neglected when adding all other
     * features :P) */
    void serialize(Stream *stream, InstanceManager *manager) const {
        ProgressiveMonteCarloIntegrator::serialize(stream, manager);
        stream->writeInt(m_volVolSamples);
        stream->writeInt(m_volSurfSamples);
        stream->writeBool(m_globalCluster);
        stream->writeBool(m_localRefinement);
        stream->writeInt(m_specRRdepth);
        stream->writeFloat(m_initialSpecularThroughput);
        stream->writeBool(m_shortVrls);
    }

    bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job,
            int sceneResID, int cameraResID, int samplerResID) {
        if (!ProgressiveMonteCarloIntegrator::preprocess(scene, queue, job,
                sceneResID, cameraResID, samplerResID))
            return false;

        if (m_vrlFile != "") {
            if (scene->getMedia().size() != 1)
                SLog(EError, "When loading VRLs from a file, the scene "
                        "should (currently) contain exactly one medium, "
                        "which will be the medium where all VRLs will "
                        "'live'");
            const Medium *medium = scene->getMedia()[0].get();
            ref<FileStream> fs = new FileStream(m_vrlFile, FileStream::EReadOnly);
            m_vrls = new vrlVector(fs.get(), medium);
        }

        if (!m_globalCluster && !m_localRefinement) {
            m_prep = NULL;
        } else {
            ref<Scheduler> sched = Scheduler::getInstance();
            m_prep = new Preprocessor(m_globalCluster, m_localRefinement,
                    m_targetNumSlices, m_neighbourCount, m_neighbourWeight,
                    m_globalUndersampling, m_localUndersampling,
                    m_fallBackUndersampling, m_depthCorrection,
                    m_sliceCurvatureFactor, sched->getWorkerCount());
            Log(EInfo, "Building slices");
            m_ci->m_slices = m_prep->buildSlices(scene);
        }
        return true;
    }

    /// to be executed before a (progressive) rendering pass
    virtual bool prepass(const Scene *scene, Sampler *sampler) {
        std::vector< std::vector< std::vector<VrlContribution> > > R; // R[slice][sliceRepresentative][vrl]

        ref_vector<Emitter> ems = scene->getEmitters();
        ref<Scheduler> sched = Scheduler::getInstance();

        if (m_vrlFile == "") {
            //vrl tracing!
            ref<vrlTracer> tracer = new vrlTracer(sampler, m_maxParticleDepth, m_rrDepth);
            m_vrls = tracer->randomWalk(scene, m_vrlTargetNum, m_shortVrls);
        } else {
            /* else: use preloaded VRLs from ascii file (note: multiple
             passes with clustering will be suboptimal, as the stuff below
             is basically always the same [modulo vrl integration noise])
             */
            if (m_vrls == NULL)
                SLog(EError, "VRL filename given, but vrls were not loaded!");
        }

        if (m_globalCluster || m_localRefinement) {
            cpu_times elapsed;
            float elapsedWall, elapsedCpu;

            Log(EInfo, "Sampling slice mapping");
            cpu_timer timer;
            const std::vector< std::vector<Point2> > &representativePixels =
                    m_prep->sampleSliceMapping(m_targetPixelUndersampling, sampler);
            elapsed = timer.elapsed();
            elapsedCpu = (elapsed.system + elapsed.user) * 1e-9;
            elapsedWall = elapsed.wall * 1e-9;
            Log(EInfo, "Sampling slice mapping: %e wall, %e cpu", elapsedWall, elapsedCpu);

            Log(EInfo, "Building R");
            timer = cpu_timer(); // new timer (resets clock)
            R.resize(representativePixels.size());
            int workerCount = sched->getWorkerCount();
            if (workerCount > 1) {
                std::vector<Rbuilder *> builders(workerCount);
                for (int i = 0; i < workerCount; i++) {
                    builders[i] = new Rbuilder(i, workerCount, representativePixels,
                            *this, sampler, scene, m_Rsamples);
                    builders[i]->incRef();
                    builders[i]->start();
                }
                for (int i = 0; i < workerCount; i++) {
                    builders[i]->join();
                }
                for (int i = 0; i < workerCount; i++) {
                    builders[i]->merge(R);
                    builders[i]->decRef();
                }
                builders.clear();
            } else {
                Ray ray;
                for (size_t i = 0; i < representativePixels.size(); i++) {
                    R[i].resize(representativePixels[i].size());
                    for (size_t j = 0; j < representativePixels[i].size(); j++) {
                        scene->getSensor()->sampleRay(
                                ray, representativePixels[i][j], Point2(0.0f), 0.0f);
                        R[i][j] = getLiLuminanceVrlContributions(
                                scene, ray, sampler, m_Rsamples, &statsVrlsPreprocess);
                    }
                }
            }
            elapsed = timer.elapsed();
            elapsedCpu = (elapsed.system + elapsed.user) * 1e-9;
            elapsedWall = elapsed.wall * 1e-9;
            Log(EInfo, "Building R: %e wall, %e cpu", elapsedWall, elapsedCpu);

            Log(EInfo, "Building clusters");
            timer = cpu_timer(); // new timer (resets clock)
            m_ci->m_clusterWeight.clear();
            m_ci->m_selectedVrls = m_prep->buildClusters(
                    R, m_ci->m_clusterWeight,
                    m_ci->m_gcVrls, m_ci->m_gcWeight,
                    m_ci->m_fallBackVrls, m_ci->m_fallBackWeight,
                    sampler);
            elapsed = timer.elapsed();
            elapsedCpu = (elapsed.system + elapsed.user) * 1e-9;
            elapsedWall = elapsed.wall * 1e-9;
            Log(EInfo, "Building clusters: %e wall, %e cpu", elapsedWall, elapsedCpu);
        }

        m_vrlsID = sched->registerResource(m_vrls);
        m_ciID = sched->registerResource(m_ci);
        return true;
    }
    std::string passFileSuffix() {
        std::stringstream suffix;
        float preprocessVrls = statsVrlsPreprocess.getValue();
        float renderVrls = statsVrlsRender.getValue();
        suffix << std::fixed << std::scientific << std::setprecision(4)
                << "_prevrl" << preprocessVrls << "_renvrl" << renderVrls;
        return suffix.str();
    }

    void configureSampler(const Scene *scene, Sampler *sampler) {
        ProgressiveMonteCarloIntegrator::configureSampler(scene, sampler);
    }

    /// Specify globally shared resources: just the vrl collection
    void bindUsedResources(ParallelProcess *proc) const {
        ProgressiveMonteCarloIntegrator::bindUsedResources(proc);
        proc->bindResource("vrls", m_vrlsID);
        proc->bindResource("vrlClusterInfo", m_ciID);
    }

    /// Connect to globally shared resources
    void wakeup(ConfigurableObject *parent, std::map<std::string, SerializableObject *> &params) {
        ProgressiveMonteCarloIntegrator::wakeup(parent, params);
        if (params.find("vrls") != params.end())
            m_vrls = static_cast<vrlVector *>(params["vrls"]);
        if (params.find("vrlClusterInfo") != params.end())
            m_ci = static_cast<vrlClusterInfo *>(params["vrlClusterInfo"]);
    }

    Spectrum Li(const RayDifferential &ray, RadianceQueryRecord &rRec) const {
        size_t numVrls = 0;
        Spectrum Li = LiInternal(ray, rRec, NULL, m_globalCluster || m_localRefinement, 1,
                ray, Spectrum(m_initialSpecularThroughput), Spectrum(1.0),
                &statsVrlsRender, m_numVrlFalseColor, m_slicesFalseColor,
                m_convergenceFalseColor, &numVrls);
        return Li;
    }

private:
    /// Internal function for Li computation, which supports individual VRL contribution reporting
    /// samples: for sampling this same ray multiple times
    Spectrum LiInternal(
            const RayDifferential &ray,
            RadianceQueryRecord &rRec,
            std::vector<VrlContribution> *vrlContributions,
            bool clustered,
            size_t samples,
            const RayDifferential &cameraRay,
            Spectrum throughputWithEtaSq,
            Spectrum weight, /// multiply result by this factor (needed for consistent vrlContribution accumulation)
            StatsCounter *stats, // = &statsVrlsRender,
            bool numVrlFalseColor, // = false,
            bool slicesFalseColor, // = false,
            bool convergenceFalseColor, // = false,
            size_t *numVrls // = NULL, // all of the above false color stuff are ugly retrofitted hacks :P
        ) const {

        if (samples == 0)
            SLog(EError, "LiInternal() requested 0 samples");

        SLog(EDebug, "Intersecting %s", ray.toString().c_str());
        if (!rRec.rayIntersect(ray)) {
            if (rRec.medium) {
                SLog(EWarn, "Dropping contributions to an infinite eye ray!");
            }
            return Spectrum(0.0f); // TODO: ACTUALLY INTEGRATE OVER VRLS IF WE ARE IN A MEDIUM!! (over infinite eye ray)
        }
        SLog(EDebug, "Hit %s", rRec.its.toString().c_str());

        Spectrum LiDirect(0.0f);
        for (size_t i = 0; i < samples; i++) {
            if (clustered) {
                if (vrlContributions) {
                    Log(EError, "requested VRL contributions on clustered query!");
                }
                LiDirect += getClusteredVrlContributions(
                        ray, rRec, cameraRay, weight, stats, numVrls,
                        numVrlFalseColor, slicesFalseColor);
            } else {
                if (slicesFalseColor) {
                    Log(EError, "requested slices false color image without clustering!");
                }
                LiDirect += getVRLContributions(
                        ray, rRec, vrlContributions, weight, stats, numVrls, numVrlFalseColor);
            }
        }
        LiDirect /= samples;

        const BSDF *bsdf = rRec.its.getBSDF();

        if (!(bsdf->getType() & BSDF::EDelta))
            return LiDirect; // No specular chains

        Spectrum transmittance;
        if (rRec.medium) {
            MediumSamplingRecord mRec;
            rRec.medium->eval(Ray(ray, 0, rRec.its.t), mRec);
            transmittance = mRec.transmittance;
        } else {
            transmittance = Spectrum(1.0f);
        }
        if (transmittance.isZero())
            return LiDirect;

        if (slicesFalseColor)
            return LiDirect;

        /* Specular chains */
        Spectrum LiSpec(0.0f);
        RadianceQueryRecord rRec2;
        int compCount = bsdf->getComponentCount();
        for (int i = 0; i < compCount; i++) {
            unsigned int type = bsdf->getType(i);
            if (!(type & BSDF::EDelta))
                continue;

            BSDFSamplingRecord bRec(rRec.its, rRec.sampler, ERadiance);
            bRec.component = i;
            Spectrum bsdfWeight = bsdf->sample(bRec, Point2(0.5f));
            SLog(EDebug, "BSDF sampling record: %s", bRec.toString().c_str());
            if (bsdfWeight.isZero())
                continue;

            Spectrum throughputWithEtaSq2 = throughputWithEtaSq * transmittance * bsdfWeight * (bRec.eta*bRec.eta);

            Float maxRRprob;
            if (rRec.depth >= m_specRRdepth) {
                maxRRprob = 0.98;
            } else {
                maxRRprob = 1;
            }
            Float rrProb = std::min(maxRRprob, throughputWithEtaSq2.max());
            if (rrProb <= 0  ||  (rrProb < 1 && rRec.nextSample1D() > rrProb)) {
                continue;
            }
            throughputWithEtaSq2 /= rrProb;

            rRec2.recursiveQuery(rRec);
            RayDifferential ray2(rRec.its.p, rRec.its.toWorld(bRec.wo), ray.time);
            if (rRec.its.isMediumTransition()) {
                rRec2.medium = rRec.its.getTargetMedium(ray2.d);
            }

            SLog(EDebug, "Recursing!");
            Spectrum weightSpec;
            if (numVrlFalseColor) {
                weightSpec = weight;
            } else {
                weightSpec = weight * transmittance * bsdfWeight/rrProb;
            }

            LiSpec += LiInternal(ray2, rRec2, vrlContributions, clustered, samples, cameraRay,
                    throughputWithEtaSq2, weightSpec, stats,
                    numVrlFalseColor, slicesFalseColor, convergenceFalseColor, numVrls);
        }
        Spectrum Li = LiDirect + LiSpec;

        if (convergenceFalseColor) {
            // R,G,B -> Luminance, Num Vrls, 0
            Spectrum result;
            result[0] = Li.getLuminance();
            result[1] = *numVrls;
            result[2] = 0;
            return result;
        } else {
            return Li;
        }
    }

    /// Return the contributions of each vrl to the luminance of the rendered ray, as per Li(), hence including specular chains.
    std::vector<VrlContribution> getLiLuminanceVrlContributions(
            const Scene *scene, Ray &ray, Sampler *sampler, size_t samples,
            StatsCounter *stats) const {
        std::vector<VrlContribution> result(m_vrls->size(), VrlContribution(0, 0));
        RayDifferential rayd(ray);
        RadianceQueryRecord rRec(scene, sampler);
        rRec.newQuery(RadianceQueryRecord::ESensorRay, scene->getSensor()->getMedium());
        rRec.rayIntersect(rayd);
        LiInternal(rayd, rRec, &result, false, samples, rayd,
                Spectrum(m_initialSpecularThroughput), Spectrum(1.0),
                stats, false, false, false, NULL);
        return result;
    }


    Spectrum getClusteredVrlContributions(
            const RayDifferential &ray, RadianceQueryRecord &rRec,
            const RayDifferential &cameraRay, Spectrum weight, StatsCounter *stats,
            size_t *numVrls, bool numVrlFalseColor, bool slicesFalseColor) const {
        if (!slicesFalseColor && (!rRec.medium || rRec.medium->getSigmaS().isZero())) {
            return Spectrum(0.0f); // We are not in a scattering medium, hence visibility between any VRL will be zero!
        }

        // Determine slice number
        DirectionSamplingRecord slice_dirRec(cameraRay.d);
        PositionSamplingRecord slice_posiRec;
        slice_posiRec.p = cameraRay.o;

        Point2 slice_result;
        rRec.scene->getSensor()->getSamplePosition(slice_posiRec, slice_dirRec, slice_result);
        int x = slice_result.x;
        int y = slice_result.y;

        uint32_t sliceNumber = m_ci->m_slices[y + (getImageSizeY(rRec.scene)*x)];

        const std::vector<uint32_t> *vrls;
        const std::vector<Float> *weights;
        if (sliceNumber == UINT32_T_MAX) {
            // This slice got no contribution in the reduced matrix -> use fall-back cluster
            vrls = &m_ci->m_fallBackVrls;
            weights = &m_ci->m_fallBackWeight;
        } else {
            vrls = &m_ci->m_selectedVrls[sliceNumber];
            weights = &m_ci->m_clusterWeight[sliceNumber];
        }

        Spectrum Li;
        if (numVrlFalseColor) {
            Li = Spectrum(((Float) vrls->size()) / m_vrls->size());
        } else if (slicesFalseColor) {
            if (sliceNumber == UINT32_T_MAX)
                Li = Spectrum(0.5); // fallback clustering in gray
            else
                //Li.fromLinearRGB((sliceNumber % 7)/7.0, ((sliceNumber + 2) % 7)/7.0, ((sliceNumber + 4) % 7)/7.0);
                Li.fromLinearRGB(
                        ((sliceNumber + sliceNumber*sliceNumber)% 43)/43.0,
                        ((7*sliceNumber + 2*sliceNumber*sliceNumber + 7)% 41)/41.0,
                        ((23*sliceNumber + 5*sliceNumber*sliceNumber + sliceNumber*sliceNumber*sliceNumber + 17)% 53)/53.0);
        } else {
            Li = Spectrum(0.0f);
            for (size_t i = 0; i < vrls->size(); i++)
                Li += weights->at(i) * integrateVRL(ray, rRec, m_vrls->get(vrls->at(i)),
                        m_volVolSamples, m_volSurfSamples);
            Li /= m_vrls->getParticleCount();
        }

        if (stats)
            (*stats) += vrls->size();
        if (numVrls)
            (*numVrls) += vrls->size();

        return Li * weight;
    }


    /// integrate the eye ray against the vrl
    Spectrum integrateVRL(const RayDifferential &ray, RadianceQueryRecord &rRec, const VRL &vrl,
            int volVolSamples, int volSurfSamples, Float *contrib = NULL, Float *variance = NULL,
            Spectrum weight = Spectrum(1.0)) const {
        const Medium *eyeMedium = rRec.medium;
        const Medium *vrlMedium = vrl.m_medium.get();

        if (contrib)
            *contrib = 0;
        if (variance)
            *variance = 0;

        if (!eyeMedium || eyeMedium->getSigmaS().isZero()) {
            // We are not in a scattering medium -> visibility with the vrl will be zero
            return Spectrum(0.0f);
        }

        if (!vrlMedium) {
            SLog(EError, "VRL without associated medium!");
            return Spectrum(0.0f);
        }

        Point U; // The sampled position along the eye ray
        Point V; // The sampled position along the VRL
        Point S = vrl.m_start; // Starting position of the vrl
        Point E = ray.o; // Eye position
        Point Usurf = rRec.its.p; // The hit point of the eye ray at the surface (if it exists)
        Vector SV = normalize(vrl.m_end - vrl.m_start); // Direction along the vrl
        Vector VU; // Direction from sampled VRL position to sampled eye position
        Vector EU = ray.d; // Direction from the eye to the sampled eye position
        MediumSamplingRecord eyeMRec;
        MediumSamplingRecord vrlMRec;
        Float samplingPDF;
        Float time = rRec.its.time;

        Spectrum totalContribution(0.0f);
        Spectrum transmittanceUV;

        int negOne = -1;

        Float volVolSampleLum[volVolSamples]; // TODO: quick hack for variance, do it properly online
        for (int i = 0; i < volVolSamples; i++) {
            volVolSampleLum[i] = 0;
        }
        /* Volume to volume transport, L (V|D|S)* V V S* E */
        for (int sample = 0; sample < volVolSamples; sample++) {
            // get contribution to volume points of ray.
            //sample U and V
            samplingPDF = sampleUV(ray, rRec, vrl, U, V, rRec.sampler);
            if (distance(U, V) == 0) {
                Log(EWarn, "distance U to V is zero");
                continue;
            }
            VU = normalize(U - V);

            negOne = -1;
            transmittanceUV = rRec.scene->evalTransmittance(U, false, V, false, time, eyeMedium, negOne, rRec.sampler);
            negOne = -1;

            if (transmittanceUV.isZero()) {
                continue;
            }

            eyeMedium->eval(Ray(E, EU, 0, distance(E, U), time), eyeMRec);
            vrlMedium->eval(Ray(S, SV, 0, distance(S, V), time), vrlMRec);

            Spectrum contribution = weight;
            contribution *= vrl.m_power;                                    // VRL power
            contribution *= vrlMRec.sigmaS * eyeMRec.sigmaS / samplingPDF;  // scattering and sampling probabilities
            contribution *= 1/distanceSquared(U, V);                        // distance squared from U to V
            contribution *= vrlMRec.transmittance;                          // transmittance from S to V
            contribution *= transmittanceUV;                                // transmittance from V to U
            contribution *= eyeMRec.transmittance;                          // transmittance from U to E
            if (m_shortVrls)
                contribution /= vrlMRec.pdfFailure;

            // PHASE FUNCTIONS
            const PhaseFunction *phaseU = eyeMedium->getPhaseFunction();
            const PhaseFunction *phaseV = vrlMedium->getPhaseFunction();
            eyeMRec.p = U;
            vrlMRec.p = V;
            contribution *= phaseU->eval(PhaseFunctionSamplingRecord(eyeMRec, -VU, -EU));
            contribution *= phaseV->eval(PhaseFunctionSamplingRecord(vrlMRec, -SV, VU));

            if (!contribution.isValid()) {
                SLog(EWarn, "Dropping invalid VRL vol-to-vol contribution: %s!", contribution.toString().c_str());
            } else {
                totalContribution += contribution/volVolSamples;
                volVolSampleLum[sample] = contribution.getLuminance();
            }
        }
        Float mean = 0;
        Float M2 = 0;
        for (int i = 0; i < volVolSamples; i++) {
            Float delta = volVolSampleLum[i] - mean;
            mean += delta/(i+1);
            M2 += delta * (volVolSampleLum[i] - mean);
        }
        if (contrib && volVolSamples > 0)
            *contrib += mean;
        if (variance && volVolSamples > 0) // actually > 1...
            *variance += M2 / ((volVolSamples - 1) * volVolSamples); // M2/(n-1) estimates variance of samples and factor 1/n gets variance of mean


        /* Volume to surface transport, L (V|D|S)* V D S* E */
        U = Usurf;
        const BSDF *bsdf = rRec.its.getBSDF();

        // Precompute transittance from surface to eye
        Spectrum transmittanceEUsurf(0.0f);
        if (rRec.its.isValid()) {
            if (distance(Usurf, E) != 0) {
                vrlMedium->eval(Ray(ray, 0, distance(Usurf, E)), eyeMRec);
                transmittanceEUsurf = eyeMRec.transmittance;
            } else {
                Log(EWarn, "distance Usurf to eye is zero");
            }
        }

        Float volSurfSampleLum[volSurfSamples]; // TODO: quick hack for variance, do it properly online
        for (int i = 0; i < volSurfSamples; i++) {
            volSurfSampleLum[i] = 0;
        }
        // Sample volume to surface transport
        unsigned int bsdfType = BSDF::ESmooth; //request non-degenerate component
        if (!transmittanceEUsurf.isZero() && (bsdf->getType() & bsdfType)) {
            for (int sample = 0; sample < volSurfSamples; sample++) {

                samplingPDF = sampleV(ray, rRec, vrl, V, rRec.sampler);
                if (distance(U, V) == 0) {
                    Log(EWarn, "distance U to V is zero");
                    continue;
                }
                VU = normalize(U - V);

                negOne = -1;
                Spectrum transmittanceUV = rRec.scene->evalTransmittance(
                        U, true, V, false, time, eyeMedium, negOne, rRec.sampler);
                negOne = -1;
                vrlMedium->eval(Ray(S, SV, 0, distance(S, V), time), vrlMRec);

                Spectrum contribution = weight;
                contribution *= vrl.m_power;                            // VRL power
                contribution *= vrlMedium->getSigmaS() / samplingPDF;   // scattering and sampling probability
                contribution *= 1/distanceSquared(U, V);                // distance squared from U to V
                contribution *= vrlMRec.transmittance;                  // transmittance from S to V
                contribution *= transmittanceUV;                        // transmittance from V to U
                contribution *= transmittanceEUsurf;                    // transmittance from U to E
                if (m_shortVrls)
                    contribution /= vrlMRec.pdfFailure;

                // Phase function at V
                const PhaseFunction *phaseV = vrlMedium->getPhaseFunction();
                vrlMRec.p = V;
                contribution *= phaseV->eval(PhaseFunctionSamplingRecord(vrlMRec, -SV, VU));

                // Bsdf at U
                BSDFSamplingRecord bRec(rRec.its, rRec.its.toLocal(-VU));
                bRec.typeMask = bsdfType;
                contribution *= bsdf->eval(bRec);

                if (!contribution.isValid()) {
                    SLog(EWarn, "Dropping invalid VRL vol-to-surf contribution!");
                } else {
                    totalContribution += contribution / volSurfSamples;
                    volSurfSampleLum[sample] = contribution.getLuminance();
                }
            }
        }

        mean = 0;
        M2 = 0;
        for (int i = 0; i < volSurfSamples; i++) {
            Float delta = volSurfSampleLum[i] - mean;
            mean += delta/(i+1);
            M2 += delta * (volSurfSampleLum[i] - mean);
        }
        if (contrib && volSurfSamples > 0)
            *contrib += mean;
        if (variance && volSurfSamples > 0) // actually > 1 (but that was checked before)
            *variance += M2 / ((volSurfSamples - 1) * volSurfSamples);

        return totalContribution;
    }

    /*
     * Get contribution of VRLs to the given ray. Store the luminance
     * contributions of individual VRLs in vrlContributions if provided
     * (should be of correct size).
     */
    Spectrum getVRLContributions(const RayDifferential &ray, RadianceQueryRecord &rRec,
            std::vector<VrlContribution> *vrlContributions, Spectrum weight,
            StatsCounter *stats, size_t *numVrls, bool numVrlFalseColor) const {
        if (!rRec.medium || rRec.medium->getSigmaS().isZero()) {
            return Spectrum(0.0f); // We are not in a scattering medium, hence visibility between any VRL will be zero!
        }

        Spectrum Li;
        if (numVrlFalseColor) {
            Li = weight;
        } else {
            Li = Spectrum(0.0f);
            for (size_t i = 0; i < m_vrls->size(); i++) {
                Float normalization = 1.0 / m_vrls->getParticleCount();
                Float contribution, variance;
                Spectrum vrlContribution = integrateVRL(ray, rRec,
                        m_vrls->get(i), m_volVolSamples, m_volSurfSamples,
                        &contribution, &variance, weight);
                vrlContribution *= normalization;
                if (vrlContributions) {
                    (*vrlContributions)[i].mean += contribution * normalization;
                    (*vrlContributions)[i].var  += variance * normalization*normalization;
                }
                Li += vrlContribution;
            }
        }

        if (stats)
            (*stats) += m_vrls->size();
        if (numVrls)
            (*numVrls) += m_vrls->size();

        return Li;
    }


    /*
     * Sampling by the method of Kulla (see Kulla et al. 2011)
     */
    Float sampleUV(const RayDifferential& eyeRay, const RadianceQueryRecord& eyeRayQ,const VRL& vrl, Point& U, Point& V, Sampler* sampler) const {
        return sampleUVKulla(eyeRay, eyeRayQ, vrl, U, V, sampler);
        //return sampleUVuniform(eyeRay, eyeRayQ, vrl, U, V, sampler);
    }
    /*
     * sample v for eq 6
     */
    Float sampleV(const mitsuba::RayDifferential& eyeRay, const RadianceQueryRecord& eyeRayQ,
            const VRL& vrl, Point& V, Sampler* sampler) const {
        return KullaSampling(vrl.m_start, vrl.m_end, eyeRayQ.its.p, V, sampler);//Kulla sample along ray
        //return sampleVuniform(vrl.m_start, vrl.m_end, eyeRayQ.its.p, V, sampler);
    }

    Float sampleUVuniform(const RayDifferential &eyeRay, const RadianceQueryRecord &eyeRayQ,
            const VRL &vrl, Point &U, Point &V, Sampler *sampler) const {
        // TODO eyeRayQ potentially has no intersection /// intersection at infinty!!
        if (!eyeRayQ.its.isValid() || !std::isfinite(eyeRayQ.its.t))
            SLog(EError, "uniform sampling on infinite eye ray!");
        Point eyeStart = eyeRay.o;
        Point eyeEnd = eyeRayQ.its.p;
        Point vrlStart = vrl.m_start;
        Point vrlEnd = vrl.m_end;
        Float eyeLen = distance(eyeStart, eyeEnd);
        Float vrlLen = distance(vrlStart, vrlEnd);
        Point2 rnd = sampler->next2D();
        U = eyeStart + (eyeEnd - eyeStart) * rnd.x;
        V = vrlStart + (vrlEnd - vrlStart) * rnd.y;
        return 1 / (eyeLen * vrlLen);
    }
    Float sampleUVKulla(const RayDifferential& eyeRay, const RadianceQueryRecord& eyeRayQ,
            const VRL& vrl, Point& U, Point& V, Sampler* sampler) const {
        //As In Paper
        Float result = sampleVtoDistance(eyeRay, eyeRayQ, vrl, V, sampler->next1D());

        Float n, dist;
        if (eyeRayQ.its.t == std::numeric_limits<Float>::infinity())
            eyeRayQ.scene->getAABB().rayIntersect(eyeRay, n, dist);
        else
            dist = distance(eyeRayQ.its.p, eyeRay.o);
        Point A = eyeRay.o;
        Point B = eyeRay.o + (dist*eyeRay.d);

        result *= KullaSampling(A, B, V, U, sampler); //Kulla sample along ray

        return result;
    }

    Float sampleVuniform(const Point &vrlStart, const Point &vrlEnd,
            const Point &surfacePoint, Point& V, Sampler* sampler) const {
        // TODO eyeRayQ potentially has no intersection /// intersection at infinty!!
        Float vrlLen = distance(vrlStart, vrlEnd);
        Float rnd = sampler->next1D();
        V = vrlStart + (vrlEnd - vrlStart) * rnd;
        return 1 / vrlLen;
    }
    /*
     * Importance sampling according to distance, by Kulla et al.
     */
    Float KullaSampling(Point A, Point B, Point D, Point& result, Sampler* sampler) const{
        Vector dir = normalize(B - A);
        Float dotPr = dot(dir, D - A);
        Point I = A + (dotPr * dir);

        Float Dis = distance(D, I);
        Float angle_a = atan(distance(A, I)/Dis);
        Float angle_b = atan(distance(I, B)/Dis);

        //get the signs of the angles right
        if (dotPr > 0) {
            angle_a *= -1;
            if (distance(A, I) > distance(A, B))
                angle_b *= -1;
        }// else, both are positive

        Float uniform = sampler->next1D();

        Float t = Dis * tan(((1.0f-uniform) * angle_a) + (uniform * angle_b));

        Float pdf = Dis / ((angle_b - angle_a) * (Dis*Dis + t*t)); // pdf of sampling

        result = I + (t * dir); //Kulla sample along ray

        return pdf;
    }

    Float sampleVtoDistance(
            const mitsuba::RayDifferential& eyeRay, const RadianceQueryRecord& eyeRayQ,
            const VRL& vrl, Point& V, Float uniform) const {
        if (distance(vrl.m_start, vrl.m_end) == 0) {
            SLog(EWarn, "zero length vrl in sampleVtoDistance");
            V = vrl.m_start;
            return 1;
        }

        Float cosTheta = dot(normalize(eyeRay.d), normalize(vrl.m_end - vrl.m_start));
        Float sinTheta = math::safe_sqrt(1 - cosTheta*cosTheta);

        if (sinTheta < Epsilon) { // VRL and eye ray are (nearly) parallel
            // sample uniformly over the VRL
            V = vrl.m_start + uniform * (vrl.m_end - vrl.m_start);
            return 1 / distance(vrl.m_end, vrl.m_start);
        }

        Point Uh;
        Point Vh;
        Float h = getClosestPoints(eyeRay.o, eyeRayQ.its.p, vrl.m_start, vrl.m_end, Uh, Vh);

        Float V0c = -1 * distance(Vh, vrl.m_start);
        Float V1c = distance(Vh, vrl.m_end);

        Float newV = h * sinh(A(V0c, h, sinTheta) + (uniform * (A(V1c, h, sinTheta) - A(V0c, h, sinTheta))));
        newV = newV / sinTheta;

        Float result = 1.0f / sqrt(h*h + newV*newV*sinTheta*sinTheta);

        Float denom = (A(V1c, h, sinTheta) - A(V0c, h, sinTheta)) / sinTheta;

        newV += distance(Vh, vrl.m_start); //reparametrisation

        V = vrl.m_start + newV * (normalize(vrl.m_end - vrl.m_start));

        return result/denom;
    }

    Float A(Float x, Float h, Float sinTheta) const {
        return asinh((x / h) * sinTheta);
    }

    /**
     * Get closest points of two lines
     */
    Float getClosestPoints(Point S1P0, Point S1P1, Point S2P0, Point S2P1, Point& S1h, Point& S2h) const {
        Vector u = S1P1 - S1P0;
        Vector v = S2P1 - S2P0;
        Vector w = S1P0 - S2P0;
        Float  a = dot(u,u);         // always >= 0
        Float  b = dot(u,v);
        Float  c = dot(v,v);         // always >= 0
        Float  d = dot(u,w);
        Float  e = dot(v,w);
        Float  D = a*c - b*b;        // always >= 0
        Float  sc, sN, sD = D;       // sc = sN / sD, default sD = D >= 0
        Float  tc, tN, tD = D;       // tc = tN / tD, default tD = D >= 0

        // compute the line parameters of the two closest points
        if (D < Epsilon * u.lengthSquared() * v.lengthSquared()) { // the lines are almost parallel
            // TODO: *sample* some points along the lines to avoid bias!
            sN = 0.0;         // force using point P0 on segment S1
            sD = 1.0;         // to prevent possible division by 0.0 later
            tN = e;
            tD = c;
        } else {                 // get the closest points on the infinite lines
            sN = (b*e - c*d);
            tN = (a*e - b*d);
            if (sN < 0.0) {        // sc < 0 => the s=0 edge is visible
                sN = 0.0;
                tN = e;
                tD = c;
            }
            else if (sN > sD) {  // sc > 1  => the s=1 edge is visible
                sN = sD;
                tN = e + b;
                tD = c;
            }
        }

        if (tN < 0.0) {            // tc < 0 => the t=0 edge is visible
            tN = 0.0;
            // recompute sc for this edge
            if (-d < 0.0)
                sN = 0.0;
            else if (-d > a)
                sN = sD;
            else {
                sN = -d;
                sD = a;
            }
        } else if (tN > tD) {      // tc > 1  => the t=1 edge is visible
            tN = tD;
            // recompute sc for this edge
            if ((-d + b) < 0.0)
                sN = 0;
            else if ((-d + b) > a)
                sN = sD;
            else {
                sN = (-d +  b);
                sD = a;
            }
        }

        // finally do the division to get sc and tc
        sc = sN / sD;
        tc = tN / tD;

        // get the difference of the two closest points
        Vector   dP = w + (sc * u) - (tc * v);  // =  S1(sc) - S2(tc)

        S1h = S1P0 + sc * (S1P1 - S1P0);
        S2h = S2P0 + tc * (S2P1 - S2P0);

        return dP.length();   // return the closest distance
    }

    int getImageSizeY(const Scene *scene) const {
        return scene->getSensor()->getFilm()->getSize().y;
    }

    class Rbuilder : public Thread {
    public:
        Rbuilder(int id, int workerCount, const std::vector< std::vector<Point2> > &representativePixels,
                const vrlIntegrator &integrator, Sampler *sampler, const Scene *scene, int Rsamples)
                : Thread(formatString("Rbld%i", id)),
                m_Rsamples(Rsamples),
                m_reprPixels(representativePixels),
                m_integrator(integrator),
                m_scene(scene) {
                    setCritical(true);
                    m_sampler = sampler->clone();
                    size_t numSamples = m_reprPixels.size();
                    m_minIndex = (  id   * numSamples)/workerCount;
                    m_maxIndex = ((id+1) * numSamples)/workerCount;
                }
        void run() {
            m_results = std::vector< std::vector< std::vector< VrlContribution > > >(m_maxIndex - m_minIndex);
            Ray tempRay;
            for (size_t loc_i = 0; loc_i < m_maxIndex - m_minIndex; loc_i++) {
                size_t glob_i = loc_i + m_minIndex;
                m_results[loc_i].resize(m_reprPixels[glob_i].size());
                for (size_t j = 0; j < m_reprPixels[glob_i].size(); j++) {
                    m_scene->getSensor()->sampleRay(
                            tempRay, m_reprPixels[glob_i][j], Point2(0.0f), 0.0f); // get a ray for this sample
                    // TODO: should probably do this more cleanly, but for now made it a public function in the integrator
                    m_results[loc_i][j] = m_integrator.getLiLuminanceVrlContributions(
                            m_scene, tempRay, m_sampler, m_Rsamples, &statsVrlsPreprocess);
                }
            }
        }
        void merge(std::vector< std::vector< std::vector< VrlContribution > > > &R) {
            for (size_t loc_i = 0; loc_i < m_maxIndex - m_minIndex; loc_i++) {
                size_t glob_i = loc_i + m_minIndex;
                R[glob_i] = m_results[loc_i];
            }
        }

    private:
        int m_Rsamples;
        const std::vector< std::vector<Point2> > &m_reprPixels;
        const vrlIntegrator &m_integrator;
        std::vector< std::vector< std::vector <VrlContribution> > > m_results; // same indexing as R
        ref<Sampler> m_sampler;
        size_t m_minIndex, m_maxIndex; //handle from min_index up to, but not including max_index
        const Scene* m_scene;
    };

    /*
     * Variables
     */
    // Needed for rendering
    ref<vrlVector> m_vrls;
    ref<vrlClusterInfo> m_ci;
    int m_vrlsID;
    int m_ciID;
    int m_volVolSamples;
    int m_volSurfSamples;
    bool m_globalCluster;
    bool m_localRefinement;
    int m_specRRdepth; /// Depth at which an upper bound is placed on the RR probability, to avoid infinite perfectly specular reflections
    Float m_initialSpecularThroughput; /// This is a naturally adaptive way of introducing Russian Roulette in the specular chains
    bool m_numVrlFalseColor;
    bool m_slicesFalseColor;
    bool m_convergenceFalseColor;


    // Needed for preprocessor
    ref<Preprocessor> m_prep;
    int m_vrlTargetNum;
    int m_maxParticleDepth;
    int m_neighbourCount;
    Float m_neighbourWeight;
    int m_Rsamples;
    size_t m_targetNumSlices;
    Float m_targetPixelUndersampling;
    Float m_globalUndersampling;
    Float m_localUndersampling;
    Float m_fallBackUndersampling;
    Float m_depthCorrection;
    Float m_sliceCurvatureFactor;
    std::string m_vrlFile; /// load vrls from this file (unless == "")

    // Needed for both
    bool m_shortVrls;
};


MTS_IMPLEMENT_CLASS_S(vrlIntegrator, false, ProgressiveMonteCarloIntegrator);
MTS_IMPLEMENT_CLASS_S(vrlClusterInfo, false, SerializableObject);
MTS_EXPORT_PLUGIN(vrlIntegrator, "An implementation of Adaptive Lightslice for Virtual Ray Lights");
MTS_NAMESPACE_END

