#include <mitsuba/render/scene.h>

MTS_NAMESPACE_BEGIN

/*
 * Defines the result of a VRL integration with respect to an eye ray.
 */
struct VrlContribution {
    Float mean; /// mean contribution (luminance)
    Float var;  /// variance of the mean
    VrlContribution(Float m, Float v) : mean(m), var(v) {}
};

/*
 * Defines an actual VRL.
 */
class VRL : public SerializableObject{
public:

    //define an invalid VRL
    inline VRL() {
        m_medium = NULL;
        //m_start = Point(std::numeric_limits<double>::quiet_NaN());
        //m_end = Point(std::numeric_limits<double>::quiet_NaN());
    }

    inline VRL(Spectrum p, Point starting, Point end, const Medium *m):
            m_power(p), m_start(starting), m_end(end), m_medium(m) { }

    inline VRL(Spectrum p, Point starting, const Medium *m):
            m_power(p), m_start(starting), m_medium(m) {
        //m_end = Point(std::numeric_limits<double>::quiet_NaN());
    }

    VRL(Stream *stream, InstanceManager *manager) {
        m_start = Point(stream);
        m_end = Point(stream);
        m_power = Spectrum(stream);
        m_medium = static_cast<Medium *>(manager->getInstance(stream));
    }

    /// Parse ASCII version of VRL, place in given medium
    VRL(Stream *stream, const Medium *medium) :
            m_medium(medium) {
        Float r,g,b;
        std::stringstream ss(stream->readLine());
        ss >> m_start.x >> m_start.y >> m_start.z;
        ss >> m_end.x >> m_end.y >> m_end.z;
        ss >> r >> g >> b;
        m_power.fromLinearRGB(r, g, b);
        if (!m_power.isValid()) {
            Log(EError, "invalid parsed VRL power: %s", m_power.toString().c_str());
        }
    }

    ~VRL() {}

    void serialize(Stream *stream, InstanceManager *manager) const {
        m_start.serialize(stream);
        m_end.serialize(stream);
        m_power.serialize(stream);
        manager->serialize(stream, m_medium.get());
    }

    void serializeAscii(Stream *stream) const {
        Float r,g,b;
        m_power.toLinearRGB(r, g, b);
        std::ostringstream oss;
        oss << m_start.x << m_start.y << m_start.z;
        oss << m_end.x << m_end.y << m_end.z;
        oss << r << g << b;
        stream->writeLine(oss.str());
    }

    Ray getRay() {
        Point origin = m_start;
        float maxt = distance(m_start, m_end);
        Vector dir = normalize(m_start - m_end);
        return Ray(origin, dir, 0, maxt, 0);
    }

    void printVRL(int number) const{
        Log(EDebug, "%i - start: %f, %f, %f", number, m_start.x, m_start.y, m_start.z);
        Log(EDebug, "%i - end: %f, %f, %f", number, m_end.x, m_end.y, m_end.z);
        Log(EDebug, "%i - power: %f", number, m_power.getLuminance());
    }


    // The emitted radiance by the VRL at the starting point towards the end point
    Spectrum m_power;
    //startPosition
    Point m_start;
    //endPosition
    Point m_end;
    // pointer to the medium that surrounds the VRL.
    ref<const Medium> m_medium;

    MTS_DECLARE_CLASS()

};

/*
 * vrlVector: contains VRLs shot by a vrlTracer. Borrows heavily from PhotonVector
 */
class vrlVector : public SerializableObject{
public:
    vrlVector() {
        clear();
    }

    vrlVector(Stream *stream, InstanceManager *manager) {
        size_t numVrls = (size_t) stream->readUInt();
        m_numParticles = (size_t) stream->readUInt();
        m_vrls.resize(numVrls);
        for (size_t i = 0; i < numVrls; i++)
            m_vrls[i] = VRL(stream, manager);
    }

    /// Parse ASCII version of vrlVector, place in given medium
    vrlVector(Stream *stream, const Medium *medium) {
        try {
            while (true)
                put(VRL(stream, medium));
        } catch (std::exception &e) {
            // TODO sanity check to verify that we are at the end of the file
        };
        m_numParticles = size();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        stream->writeUInt((uint32_t) m_vrls.size());
        stream->writeUInt((uint32_t) m_numParticles);
        for (size_t i=0; i< m_vrls.size(); ++i)
            m_vrls[i].serialize(stream, manager);
    }

    void serializeAscii(Stream *stream) const {
        for (size_t i = 0; i < size(); i++)
            m_vrls[i].serializeAscii(stream);
    }

    ~vrlVector() {}

    inline void nextParticle() {
        m_numParticles++;
    }

    inline void put(const VRL &v) {
        if (v.m_medium == NULL || v.m_medium->getSigmaS().isZero())
            return; //don't put it in if it hasn't got a scattering medium.
        if (v.m_power.isZero())
            return; //or if it carries no power
        //if (std::isnan(v.m_start.x) || std::isnan(v.m_end.x))
        //  Log(EError, "Tried to put an invalid VRL in vrlVector!");
        if (distance(v.m_start, v.m_end) == 0)
            return; //or if it has zero length
        m_vrls.push_back(v);
    }

    inline size_t size() const {
        return m_vrls.size();
    }

    inline size_t getParticleCount() const {
        return m_numParticles;
    }

    inline void clear() {
        m_vrls.clear();
        m_numParticles = 0;
    }

    inline const VRL get(int i) const {
        return m_vrls[i];
    }

    inline const VRL &operator[](size_t index) const {
        return m_vrls[index];
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "vrlVector[size=" << m_vrls.size() << "]";
        return oss.str();
    }


    MTS_DECLARE_CLASS()

private:
    std::vector<VRL> m_vrls;
    size_t m_numParticles; // Number of particles that were traced to create the VRLs

};

MTS_IMPLEMENT_CLASS_S(vrlVector, false, SerializableObject)
MTS_IMPLEMENT_CLASS_S(VRL, false, SerializableObject)
MTS_NAMESPACE_END

