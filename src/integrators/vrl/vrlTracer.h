#include "VRL.h"

MTS_NAMESPACE_BEGIN

/*
 * This class will shoot the actual VRLs.
 */
class vrlTracer: public Object{
public:
    vrlTracer(ref<Sampler> sampler, int maxDepth, int rrDepth) :
        m_maxDepth(maxDepth), m_rrDepth(rrDepth), m_sampler(sampler) { }

    ref<vrlVector> randomWalk(const Scene *scene,
            unsigned int vrlTargetNum, bool shortVrls) {
        m_vrls = new vrlVector();

        m_Vcount = 0;
        m_Scount = 0;
        m_InfCount = 0;
        m_rrCount = 0;
        m_other = 0;

        //m_NbOfVrls = NbOfParticles;

        Log(EInfo, "Tracing for %d VRLs", vrlTargetNum);
        //uint32_t prevNumVrls = 0;
        Float printFraction = ((Float) vrlTargetNum) / 100;
        Float nextPrintThreshold = printFraction;
        while (m_vrls->size() < vrlTargetNum) {
            traceOneParticle(scene, shortVrls);
            //Log(EDebug, "VRLs stored with this particle: %d", m_vrls->size() - prevNumVrls);
            //prevNumVrls = m_vrls->size();
            if (m_vrls->size() >= nextPrintThreshold) {
                Log(EDebug, "Stored %d / %d VRLs after %d particles",
                        m_vrls->size(), vrlTargetNum,
                        m_vrls->getParticleCount());
                nextPrintThreshold += printFraction;
            }
        }

        Log(EInfo, "Particles traced: %d", m_vrls->getParticleCount());
        Log(EInfo, "V count: %d, S count: %d", m_Vcount, m_Scount);
        Log(EInfo, "rr count: %d, inf count: %d", m_rrCount, m_InfCount);
        Log(EInfo, "other: %d", m_other);
        Log(EInfo, "VRLs generated: %d (target: %d, VRLs per particle: %f)",
                m_vrls->size(), vrlTargetNum, ((Float) m_vrls->size()) / m_vrls->getParticleCount());

        ref<vrlVector> result = m_vrls;
        m_vrls = NULL;

        return result;
    }

private:
    // Handle an emission event (i.e. first vertex of a sampled path)
    void handleEmission(const PositionSamplingRecord &pRec,
            const Medium *medium, const Spectrum &weight) {
        m_vrls->nextParticle();
        // start new vrl
        m_tempVrl = VRL(weight, pRec.p, medium);
    }

    // Handle a new scatter event
    void handleSurfaceScattering(const Intersection &its,
            const Medium *medium, const Spectrum &weight) {
        //store the current vrl
        endCurrentVrl(its.p);

        //start a new vrl
        m_tempVrl = VRL(weight, its.p, medium);
    }

    //Handle a new volume scatter event
    void handleMediumScattering(Point endCurrent, Point startNew,
            const Medium *medium, const Spectrum &weight) {
        // store the current vrl
        endCurrentVrl(endCurrent);

        // start a new vrl
        m_tempVrl = VRL(weight, startNew, medium);
    }

    void endCurrentVrl(const Point p) {
        if (distance(m_tempVrl.m_start, p) == 0)
            return; // zero length vrl!
        // store the current vrl
        m_tempVrl.m_end = p;
        m_vrls->put(m_tempVrl);
    }

    void traceOneParticle(const Scene *scene, bool shortVrls) {
        MediumSamplingRecord mRec;
        Intersection its;
        ref<const Sensor> sensor = scene->getSensor();
        bool needsTimeSample = sensor->needsTimeSample();
        PositionSamplingRecord pRec(
                sensor->getShutterOpen() + 0.5f*sensor->getShutterOpenTime());
        Ray ray;

        /* Sample an emission */
        if (needsTimeSample)
            pRec.time = sensor->sampleTime(m_sampler->next1D());

        const Emitter *emitter = NULL;
        const Medium *medium;

        Spectrum power;

        /* Sample the position and direction component separately to
         * generate emission events */
        power = scene->sampleEmitterPosition(pRec, m_sampler->next2D());
        emitter = static_cast<const Emitter *>(pRec.object);
        medium = emitter->getMedium();

        DirectionSamplingRecord dRec;
        power *= emitter->sampleDirection(dRec, pRec,
                emitter->needsDirectionSample() ? m_sampler->next2D() : Point2(0.5f));

        if (power.isZero())
            return;

        handleEmission(pRec, medium, power);

        ray.setTime(pRec.time);
        ray.setOrigin(pRec.p);
        ray.setDirection(dRec.d);


        int depth = 1;

        Spectrum throughput(1.0f); // unitless path throughput (used for russian roulette)
        Float eta = 1.0f;

        while (!throughput.isZero() && (depth <= m_maxDepth || m_maxDepth < 0)) {
            scene->rayIntersectAll(ray, its);

            /* ==================================================================== */
            /*                 Radiative Transfer Equation sampling                 */
            /* ==================================================================== */
            if (!its.isValid()) {
                m_InfCount++;
            }
            if (medium && medium->sampleDistance(Ray(ray, 0, its.t), mRec, m_sampler)) {
                m_Vcount++;
                /* Sample the integral
                 \int_x^y tau(x, x') [ \sigma_s \int_{S^2} \rho(\omega,\omega') L(x,\omega') d\omega' ] dx'
                 */

                throughput *= mRec.transmittance * mRec.sigmaS / mRec.pdfSuccess;

                PhaseFunctionSamplingRecord pRec(mRec, -ray.d, EImportance);

                throughput *= medium->getPhaseFunction()->sample(pRec, m_sampler);

                Point endPoint; // endpoint of current vrl
                if (shortVrls) {
                    endPoint = mRec.p;
                } else {
                    if (its.isValid()) {
                        endPoint = its.p;
                    } else {
                        SLog(EWarn, "Long VRLs currently only support "
                                "contained media, not infinite ones! Not "
                                "creating infinite VRL!");
                        break;
                    }
                }

                handleMediumScattering(endPoint, mRec.p, medium, throughput*power);
                ray.setOrigin(mRec.p);
                ray.setDirection(pRec.wo);
                ray.mint = 0;
            } else if (its.isValid()) { // surface interaction
                m_Scount++;

                if (medium)
                    throughput *= mRec.transmittance / mRec.pdfFailure;

                const BSDF *bsdf = its.getBSDF();

                BSDFSamplingRecord bRec(its, m_sampler, EImportance);
                bRec.typeMask = BSDF::EAll;

                Spectrum bsdfWeight = bsdf->sample(bRec, m_sampler->next2D());

                if (bsdfWeight.isZero()) {
                    m_other++;
                    endCurrentVrl(its.p);
                    break;
                }

                /* Prevent light leaks due to the use of shading normals -- [Veach, p. 158] */
                Vector wi = -ray.d, wo = its.toWorld(bRec.wo);
                Float wiDotGeoN = dot(its.geoFrame.n, wi),
                woDotGeoN = dot(its.geoFrame.n, wo);
                if (wiDotGeoN * Frame::cosTheta(bRec.wi) <= 0 ||
                    woDotGeoN * Frame::cosTheta(bRec.wo) <= 0) {
                    m_other++;
                    endCurrentVrl(its.p);
                    break;
                }

                throughput *= bsdfWeight;
                eta *= bRec.eta;

                if (its.isMediumTransition())
                    medium = its.getTargetMedium(woDotGeoN);

                handleSurfaceScattering(its, medium, throughput*power);

                ray.setOrigin(its.p);
                ray.setDirection(wo);
                ray.mint = Epsilon;
            } else { // infinite ray (no medium interaction sampled, and no valid intersection)
                break;
            }

            if (depth++ >= m_rrDepth) {
                /* Russian roulette: try to keep path weights equal to one,
                 Stop with at least some probability to avoid
                 getting stuck (e.g. due to total internal reflection) */
                Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                if (m_sampler->next1D() >= q) {
                    m_rrCount++;
                    break;
                }
                throughput /= q;
            }
        }
    }

    int m_NbOfVrls;
    int m_maxDepth;
    int m_rrDepth;
    int m_Vcount;
    int m_Scount;
    int m_InfCount;
    int m_rrCount;
    int m_other;
    ref<Sampler> m_sampler;
    ref<vrlVector> m_vrls;
    VRL m_tempVrl;

    MTS_DECLARE_CLASS() //declares it as a native mitsuba class...
};
MTS_IMPLEMENT_CLASS(vrlTracer, false, Object);
MTS_NAMESPACE_END
