#include <xrslam/core/persistent_relocalization_map_io.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace xrslam {
namespace {

// Version 1: explicit byte order, float IEEE-754, bounded collections,
// FNV-1a corruption check, retained session/map frame provenance.
constexpr char MAGIC[8] = {'X','R','S','L','A','M','M','P'};
constexpr std::uint32_t VERSION = 2; // v1 still readable, but has no map alignment
constexpr std::size_t MAX_FILE = 512u * 1024u * 1024u;
constexpr std::uint64_t MAX_SESSIONS = 10000;
constexpr std::uint64_t MAX_KEYFRAMES = 2000000;
constexpr std::uint64_t MAX_OBSERVATIONS = 5000000;
constexpr std::uint64_t MAX_DESCRIPTOR_SCALARS = 100000000;

struct Writer {
    std::vector<std::uint8_t> bytes;
    void u8(std::uint8_t v) { bytes.push_back(v); }
    void u32(std::uint32_t n) {
        for (int i = 0; i < 4; ++i) u8(static_cast<std::uint8_t>(n >> (8*i)));
    }
    void u64(std::uint64_t n) {
        for (int i = 0; i < 8; ++i) u8(static_cast<std::uint8_t>(n >> (8*i)));
    }
    void f32(float v) {
        if (!std::isfinite(v)) throw std::runtime_error("nonfinite descriptor");
        std::uint32_t u; static_assert(sizeof(u)==sizeof(v),"float32");
        std::memcpy(&u,&v,sizeof(u));u32(u);
    }
    void f64(double v) {
        if (!std::isfinite(v)) throw std::runtime_error("nonfinite map geometry");
        std::uint64_t u; static_assert(sizeof(u)==sizeof(v),"float64");
        std::memcpy(&u,&v,sizeof(u));u64(u);
    }
    void str(const std::string & s) {
        if (s.empty()||s.size()>1024) throw std::runtime_error("invalid map identity");
        u32(static_cast<std::uint32_t>(s.size()));
        bytes.insert(bytes.end(),s.begin(),s.end());
    }
    void pose(const PoseState &p) {
        f64(p.q.x());f64(p.q.y());f64(p.q.z());f64(p.q.w());
        for(int i=0;i<3;++i)f64(p.p[i]);
    }
};

struct Reader {
    const std::vector<std::uint8_t> &b;
    std::size_t at=0, end=0;
    Reader(const std::vector<std::uint8_t> &bytes,std::size_t stop)
      :b(bytes),end(stop) {}
    std::uint8_t u8(){
        if(at>=end) throw std::runtime_error("truncated map archive");
        return b[at++];
    }
    std::uint32_t u32(){
        std::uint32_t r=0;
        for(int i=0;i<4;++i)r|=std::uint32_t(u8())<<(8*i);
        return r;
    }
    std::uint64_t u64(){
        std::uint64_t r=0;
        for(int i=0;i<8;++i)r|=std::uint64_t(u8())<<(8*i);
        return r;
    }
    std::uint64_t count(std::uint64_t max) {
        const auto n=u64();
        if(n>max)throw std::runtime_error("oversized map collection");
        return n;
    }
    double f64(){
        std::uint64_t bits=u64();double out;
        std::memcpy(&out,&bits,sizeof(out));
        if(!std::isfinite(out))throw std::runtime_error("nonfinite map geometry");
        return out;
    }
    float f32(){
        std::uint32_t bits=u32();float out;
        std::memcpy(&out,&bits,sizeof(out));
        if(!std::isfinite(out))throw std::runtime_error("nonfinite descriptor");
        return out;
    }
    std::string str(){
        auto n=u32();
        if(!n||n>1024||n>end-at)throw std::runtime_error("invalid map identity string");
        std::string s(b.begin()+at,b.begin()+at+n);at+=n;return s;
    }
    PoseState pose(){
        PoseState p;
        const double x=f64(),y=f64(),z=f64(),w=f64();
        if(std::abs(x*x+y*y+z*z+w*w-1.0)>0.01)
            throw std::runtime_error("invalid stored pose quaternion");
        p.q=quaternion(w,x,y,z);
        for(int i=0;i<3;++i)p.p[i]=f64();
        return p;
    }
};
std::uint64_t checksum(const std::vector<std::uint8_t> &b,std::size_t n){
    std::uint64_t h=14695981039346656037ULL;
    for(std::size_t i=0;i<n;++i){h^=b[i];h*=1099511628211ULL;}
    return h;
}
void write_session(Writer &w,const RelocalizationReferenceSession &s){
    w.u64(s.session_id);w.u64(s.semantic_tracking_session_id);
    w.u8(s.global_alignment?1:0);
    if(s.global_alignment){
        const auto &a=*s.global_alignment;
        w.u64(a.frame_id);w.f64(a.timestamp);w.f64(a.yaw_deg);
        for(int i=0;i<3;++i)w.f64(a.translation[i]);
    }
    w.u8(s.map_alignment ? 1 : 0);
    if (s.map_alignment) {
        w.f64(s.map_alignment->yaw_rad);
        for (int i = 0; i < 3; ++i)
            w.f64(s.map_alignment->translation[i]);
    }
    w.u64(s.archive.size());
    for(std::size_t frame_id:s.archive.insertion_order()){
        const auto *kf=s.archive.get(frame_id);
        if(!kf)throw std::runtime_error("corrupt archive insertion order");
        w.u64(kf->frame_id);w.f64(kf->timestamp);
        w.pose(kf->body_pose);w.pose(kf->camera_pose);
        for(int row=0;row<3;++row)for(int col=0;col<3;++col)w.f64(kf->K(row,col));
        w.u64(kf->observations.size());
        for(const auto &obs:kf->observations){
            w.u64(obs.track_id);w.u64(obs.keypoint_index);
            for(int i=0;i<3;++i)w.f64(obs.bearing[i]);
            for(int i=0;i<2;++i)w.f64(obs.pixel[i]);
            for(int i=0;i<3;++i)w.f64(obs.landmark_world[i]);
        }
        w.u8(kf->local_descriptors_complete?1:0);
        const auto &d=kf->local_descriptors;
        if(!d.valid() && (d.size()!=0 || kf->local_descriptors_complete))
            throw std::runtime_error("invalid local descriptor association");
        w.u8(static_cast<std::uint8_t>(d.type));
        w.u64(d.dimension);w.u64(d.source_indices.size());
        for(auto index:d.source_indices){
            if(index>=kf->observations.size())
                throw std::runtime_error("invalid descriptor-to-observation index");
            w.u64(index);
        }
        if(d.type==LocalDescriptorType::BINARY_U8){
            w.u64(d.binary_values.size());
            for(auto byte:d.binary_values)w.u8(byte);
        }else{
            w.u64(d.float_values.size());
            for(float f:d.float_values)w.f32(f);
        }
    }
    w.u64(s.places.size());
    for(const auto &p:s.places.entries()){
        w.u64(p.key);w.u64(p.frame_id);w.f64(p.timestamp);
        w.u64(p.descriptor.values.size());
        for(float f:p.descriptor.values)w.f32(f);
    }
}
RelocalizationReferenceSession read_session(Reader &r,std::uint32_t version) {
    RelocalizationReferenceSession s;
    s.session_id=static_cast<size_t>(r.u64());
    s.semantic_tracking_session_id=r.u64();
    if(r.u8()){
        PlaceGraph4DoFCorrection a;
        a.frame_id=static_cast<size_t>(r.u64());
        a.timestamp=r.f64();a.yaw_deg=r.f64();
        for(int i=0;i<3;++i)a.translation[i]=r.f64();
        s.global_alignment=a;
    }
    if (version >= 2 && r.u8()) {
        PersistentMapAlignment alignment;
        alignment.yaw_rad=r.f64();
        if (std::abs(alignment.yaw_rad)>6.283185307179586)
            throw std::runtime_error("invalid saved map yaw");
        for (int i=0;i<3;++i) alignment.translation[i]=r.f64();
        s.map_alignment=alignment;
    }
    for(auto i=r.count(MAX_KEYFRAMES);i>0;--i){
        ArchivedKeyframe kf;
        kf.frame_id=static_cast<size_t>(r.u64());kf.timestamp=r.f64();
        kf.body_pose=r.pose();kf.camera_pose=r.pose();
        for(int row=0;row<3;++row)for(int col=0;col<3;++col)kf.K(row,col)=r.f64();
        for(auto j=r.count(MAX_OBSERVATIONS);j>0;--j){
            ArchivedLandmarkObservation obs;
            obs.track_id=static_cast<size_t>(r.u64());
            obs.keypoint_index=static_cast<size_t>(r.u64());
            for(int t=0;t<3;++t)obs.bearing[t]=r.f64();
            for(int t=0;t<2;++t)obs.pixel[t]=r.f64();
            for(int t=0;t<3;++t)obs.landmark_world[t]=r.f64();
            kf.observations.emplace_back(std::move(obs));
        }
        kf.local_descriptors_complete=r.u8()!=0;
        auto &d=kf.local_descriptors;
        const auto type=r.u8();
        if(type>1)throw std::runtime_error("unknown descriptor type");
        d.type=static_cast<LocalDescriptorType>(type);
        d.dimension=static_cast<size_t>(r.count(4096));
        for(auto j=r.count(MAX_DESCRIPTOR_SCALARS);j>0;--j){
            const auto index=static_cast<size_t>(r.u64());
            if(index>=kf.observations.size())
                throw std::runtime_error("invalid saved descriptor index");
            d.source_indices.push_back(index);
        }
        const auto n=r.count(MAX_DESCRIPTOR_SCALARS);
        if(type==0){
            for(auto j=n;j>0;--j)d.binary_values.push_back(r.u8());
        }else{
            for(auto j=n;j>0;--j)d.float_values.push_back(r.f32());
        }
        if((kf.local_descriptors_complete || !d.source_indices.empty()) && !d.valid())
            throw std::runtime_error("malformed local descriptors");
        if(!s.archive.upsert(std::move(kf)))
            throw std::runtime_error("duplicate archived keyframe ID");
    }
    for(auto i=r.count(MAX_KEYFRAMES);i>0;--i){
        PlaceKeyframe p;
        p.key=r.u64();p.frame_id=static_cast<size_t>(r.u64());
        p.timestamp=r.f64();
        for(auto j=r.count(4096);j>0;--j)p.descriptor.values.push_back(r.f32());
        if(!s.places.add(p))throw std::runtime_error("duplicate place key");
    }
    return s;
}
} // namespace

bool save_relocalization_map(const PersistentRelocalizationMap &map,
                             const std::string &path,
                             const std::string &identity,
                             std::string *error) {
    try{
        Writer w;
        for(auto c:MAGIC)w.u8(static_cast<std::uint8_t>(c));
        w.u32(VERSION);w.str(identity);
        w.u64(map.session_count());
        for(const auto &s:map.sessions())write_session(w,s);
        if(w.bytes.size()>MAX_FILE-8)throw std::runtime_error("map archive too large");
        const auto digest=checksum(w.bytes,w.bytes.size());
        w.u64(digest);
        const std::filesystem::path dest(path);
        const std::filesystem::path tmp=dest.string()+".tmp";
        {
            std::ofstream out(tmp,std::ios::binary|std::ios::trunc);
            if(!out)throw std::runtime_error("could not open map temporary file");
            out.write(reinterpret_cast<const char *>(w.bytes.data()),
                      static_cast<std::streamsize>(w.bytes.size()));
            out.flush();
            if(!out)throw std::runtime_error("map temporary write failed");
        }
        std::filesystem::rename(tmp,dest);
        return true;
    }catch(const std::exception &e){if(error)*error=e.what();return false;}
}
bool load_relocalization_map(PersistentRelocalizationMap &map,
                             const std::string &path,
                             const std::string &identity,
                             std::string *error) {
    try {
        std::ifstream in(path,std::ios::binary|std::ios::ate);
        if(!in)throw std::runtime_error("reference map file not readable");
        const auto length=in.tellg();
        if(length<24||length>static_cast<std::streamoff>(MAX_FILE))
            throw std::runtime_error("invalid map archive size");
        in.seekg(0);
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
        in.read(reinterpret_cast<char*>(bytes.data()),length);
        if(!in)throw std::runtime_error("short map read");
        Reader r(bytes,bytes.size()-8);
        for(char c:MAGIC)if(r.u8()!=static_cast<std::uint8_t>(c))
            throw std::runtime_error("unknown map archive magic");
        const auto version=r.u32();
        if (version!=1 && version!=VERSION)
            throw std::runtime_error("unsupported map archive version");
        if(r.str()!=identity)throw std::runtime_error("map identity mismatch");
        Reader trailer(bytes,bytes.size());trailer.at=bytes.size()-8;
        if(checksum(bytes,bytes.size()-8)!=trailer.u64())
            throw std::runtime_error("map checksum mismatch");
        PersistentRelocalizationMap restored;
        std::unordered_set<PlaceKey> keys;
        for(auto i=r.count(MAX_SESSIONS);i>0;--i){
            auto session=read_session(r,version);
            for(const auto &p:session.places.entries())
                if(!keys.emplace(p.key).second)
                    throw std::runtime_error("duplicate place keys across sessions");
            if(!restored.restore_session(std::move(session)))
                throw std::runtime_error("duplicate or unordered reference session");
        }
        if(r.at!=r.end)throw std::runtime_error("trailing data in map archive");
        map=std::move(restored);
        return true;
    }catch(const std::exception &e){if(error)*error=e.what();return false;}
}
} // namespace xrslam
