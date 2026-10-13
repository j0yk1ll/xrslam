#include <XRSLAM.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <utility>
#include <cmath>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
double seconds(const builtin_interfaces::msg::Time & stamp) {
  return rclcpp::Time(stamp).seconds();
}
bool finite(double x) { return std::isfinite(x); }
bool valid_quat(const geometry_msgs::msg::Quaternion & q) {
  const double n=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
  return finite(n) && std::abs(n-1.0)<1e-3;
}
YAML::Node vec3(const geometry_msgs::msg::Vector3 & p) {
  YAML::Node n(YAML::NodeType::Sequence);
  n.push_back(p.x);n.push_back(p.y);n.push_back(p.z);return n;
}
YAML::Node quat4(const geometry_msgs::msg::Quaternion & q) {
  YAML::Node n(YAML::NodeType::Sequence);
  n.push_back(q.x);n.push_back(q.y);n.push_back(q.z);n.push_back(q.w);return n;
}
geometry_msgs::msg::Quaternion inverse_quat(const geometry_msgs::msg::Quaternion & q) {
  geometry_msgs::msg::Quaternion o=q;o.x=-q.x;o.y=-q.y;o.z=-q.z;return o;
}
} // namespace

class XrslamRos2 final : public rclcpp::Node {
public:
  XrslamRos2()
  : Node("xrslam_ros2"),
    tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_),
    tf_broadcaster_(*this) {
    slam_yaml_=declare_parameter<std::string>("slam_yaml","");
    template_yaml_=declare_parameter<std::string>("device_template_yaml","");
    image_topic_=declare_parameter<std::string>("image_topic","");
    info_topic_=declare_parameter<std::string>("camera_info_topic","");
    imu_topic_=declare_parameter<std::string>("imu_topic","/imu/data_raw");
    camera_frame_=declare_parameter<std::string>("camera_optical_frame","");
    imu_frame_=declare_parameter<std::string>("imu_frame","imu_link");
    base_frame_=declare_parameter<std::string>("base_frame","base_link");
    odom_frame_=declare_parameter<std::string>("odom_frame","odom");
    pose_validated_=declare_parameter<bool>("pose_convention_validated",false);
    enable_map_tf_=declare_parameter<bool>("enable_map_tf",false);
    map_frame_=declare_parameter<std::string>("map_frame","map");
    map_id_=declare_parameter<std::string>("map_id","");
    if(enable_map_tf_ && (map_id_.empty() || map_frame_.empty() ||
       map_frame_==odom_frame_ || map_frame_==base_frame_))
      throw std::runtime_error("enable_map_tf needs map_id and a distinct map_frame");
    if(enable_map_tf_) {
      const char *env_map_id=std::getenv("XRSLAM_MAP_ID");
      if(!env_map_id || map_id_!=env_map_id)
        throw std::runtime_error("map_id must equal XRSLAM_MAP_ID for this process");
    }
    offset_=declare_parameter<double>("camera_imu_offset_sec",std::numeric_limits<double>::quiet_NaN());
    max_jump_=declare_parameter<double>("max_pose_step_m",1.0);
    max_angular_jump_=declare_parameter<double>("max_pose_step_rad",1.0);
    if(slam_yaml_.empty()||template_yaml_.empty()||image_topic_.empty()||
       info_topic_.empty()||camera_frame_.empty()||!finite(offset_)) {
      throw std::runtime_error("Configure SLAM YAML, calibrated device template, image/CameraInfo topics, camera frame and measured camera-IMU offset");
    }
    auto qos=rclcpp::SensorDataQoS();
    auto latched=rclcpp::QoS(1).reliable().transient_local();
    status_pub_=create_publisher<std_msgs::msg::String>("/xrslam/tracking_status",latched);
    valid_pub_=create_publisher<std_msgs::msg::Bool>("/xrslam/localization_valid",latched);
    map_valid_pub_=create_publisher<std_msgs::msg::Bool>(
      "/xrslam/map_localization_valid",latched);
    map_status_pub_=create_publisher<std_msgs::msg::String>(
      "/xrslam/map_localization_status",latched);
    odom_pub_=create_publisher<nav_msgs::msg::Odometry>("/odom",rclcpp::QoS(10));
    pose_pub_=create_publisher<geometry_msgs::msg::PoseStamped>("/xrslam/pose",rclcpp::QoS(10));
    imu_sub_=create_subscription<sensor_msgs::msg::Imu>(imu_topic_,qos,
      [this](sensor_msgs::msg::Imu::ConstSharedPtr p){ imu(*p); });
    info_sub_=create_subscription<sensor_msgs::msg::CameraInfo>(info_topic_,qos,
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr p){ info(*p); });
    image_sub_=create_subscription<sensor_msgs::msg::Image>(image_topic_,qos,
      [this](sensor_msgs::msg::Image::ConstSharedPtr p){ image(p); });
    timer_=create_wall_timer(std::chrono::milliseconds(20),[this]{ poll(); });
    announce("waiting_calibration",false);
    map_announce(enable_map_tf_ ? "waiting_calibration" : "disabled",false);
  }
  ~XrslamRos2() override {
    std::lock_guard<std::mutex> lock(mutex_);
    if(initialized_) XRSLAMDestroy();
    if(!generated_yaml_.empty()) {
      std::error_code ec; std::filesystem::remove(generated_yaml_,ec);
    }
  }
private:
  void announce(const std::string & status,bool valid) {
    if(status==last_status_ && valid==last_valid_) return;
    last_status_=status;last_valid_=valid;
    std_msgs::msg::String s;s.data=status;status_pub_->publish(s);
    std_msgs::msg::Bool b;b.data=valid;valid_pub_->publish(b);
  }
  void map_announce(const std::string &status,bool valid) {
    if(map_status_==status && map_valid_==valid)return;
    map_status_=status;map_valid_=valid;
    std_msgs::msg::String s;s.data=status;
    map_status_pub_->publish(s);
    std_msgs::msg::Bool b;b.data=valid;
    map_valid_pub_->publish(b);
  }

  bool read_verified_map_alignment(XRSLAMVerifiedMapAlignment &alignment) {
    alignment={};
    if(!enable_map_tf_)return false;
    if(!XRSLAMGetVerifiedMapAlignment(&alignment)) {
      map_announce("xrslam_map_api_unavailable",false);return false;
    }
    if(map_id_!=std::string(alignment.map_id)) {
      map_announce("map_identity_mismatch",false);return false;
    }
    if(!alignment.valid) {
      map_announce("waiting_verified_map_alignment",false);return false;
    }
    if(!finite(alignment.verified_timestamp) ||
       !finite(alignment.yaw_rad) || std::abs(alignment.yaw_rad)>6.283186) {
      map_announce("invalid_map_alignment",false);return false;
    }
    for(double v:alignment.translation)
      if(!finite(v)){map_announce("invalid_map_alignment",false);return false;}
    return true;
  }

  bool init_locked() {
    if(initialized_||!info_)return initialized_;
    geometry_msgs::msg::TransformStamped body_imu,body_cam;
    try {
      body_imu=tf_buffer_.lookupTransform(base_frame_,imu_frame_,tf2::TimePointZero);
      body_cam=tf_buffer_.lookupTransform(base_frame_,camera_frame_,tf2::TimePointZero);
    }catch(const tf2::TransformException & e) {
      RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),5000,"Waiting for URDF static TF: %s",e.what());
      return false;
    }
    if(!valid_quat(body_imu.transform.rotation)||!valid_quat(body_cam.transform.rotation))
      throw std::runtime_error("Invalid static TF quaternion");
    YAML::Node dev=YAML::LoadFile(template_yaml_);
    YAML::Node slam=YAML::LoadFile(slam_yaml_);
    if(!slam["sensor"] || !slam["sensor"]["mode"] ||
       slam["sensor"]["mode"].as<std::string>()!="monocular_imu")
      throw std::runtime_error("Milestone A supports sensor.mode=monocular_imu only");
    // XRSLAM cam0.q_bc/p_bc encode the camera's orientation/origin in body.
    // Confirm via a nonidentity test before enabling pose output.
    dev["cam0"]["extrinsic"]["q_bc"]=quat4(body_cam.transform.rotation);
    dev["cam0"]["extrinsic"]["p_bc"]=vec3(body_cam.transform.translation);
    dev["imu"]["extrinsic"]["q_bi"]=quat4(body_imu.transform.rotation);
    dev["imu"]["extrinsic"]["p_bi"]=vec3(body_imu.transform.translation);
    const auto & c=*info_;
    // Rectified monocular pinhole: P defines actual image geometry.
    if(c.roi.do_rectify || c.p[0]<=0 || c.p[5]<=0 ||
       std::abs(c.p[3])>1e-8 || std::abs(c.p[7])>1e-8)
      throw std::runtime_error("Expected rectified monocular CameraInfo P with no stereo baseline");
    YAML::Node k(YAML::NodeType::Sequence);
    for(double v:{c.p[0],c.p[5],c.p[2],c.p[6]})k.push_back(v);
    dev["cam0"]["intrinsics"]=k;
    YAML::Node resolution(YAML::NodeType::Sequence);
    resolution.push_back(c.width);resolution.push_back(c.height);
    dev["cam0"]["resolution"]=resolution;
    dev["cam0"]["camera_model"]="pinhole";
    dev["cam0"]["camera_distortion_flag"]=0;
    YAML::Node zeros(YAML::NodeType::Sequence);
    for(int i=0;i<4;++i)zeros.push_back(0.0);
    dev["cam0"]["distortion"]=zeros;
    dev["cam0"]["time_offset"]=offset_;
    auto tmp=std::filesystem::temp_directory_path()/
      ("xrslam_ros2_"+std::to_string(reinterpret_cast<std::uintptr_t>(this))+".yaml");
    generated_yaml_=tmp.string();
    {
      std::ofstream out(generated_yaml_);
      if(!out) throw std::runtime_error("Could not write XRSLAM runtime device configuration");
      out<<dev;
    }
    void * config=nullptr;
    if(!XRSLAMCreate(slam_yaml_.c_str(),generated_yaml_.c_str(),"","xrslam_ros2",&config))
      throw std::runtime_error("XRSLAMCreate failed");
    initialized_=true;
    if(!XRSLAMRequiresIMU(config)) throw std::runtime_error("XRSLAM configuration unexpectedly has no IMU");
    announce("initializing",false);
    return true;
  }
  void info(const sensor_msgs::msg::CameraInfo & c) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(initialized_)return;
    if(c.header.frame_id!=camera_frame_ || c.width==0 || c.height==0 ||
       !finite(c.p[0])||!finite(c.p[5]))return;
    info_=c;
  }
  void imu(const sensor_msgs::msg::Imu & m) {
    std::lock_guard<std::mutex> lock(mutex_);
    const double t=seconds(m.header.stamp);
    if(m.header.frame_id!=imu_frame_||t<=0||
       !finite(m.angular_velocity.x)||!finite(m.angular_velocity.y)||!finite(m.angular_velocity.z)||
       !finite(m.linear_acceleration.x)||!finite(m.linear_acceleration.y)||!finite(m.linear_acceleration.z))return;
    if(last_imu_time_>0 && t<=last_imu_time_) {
      if(t<last_imu_time_-0.1) fail("imu_clock_discontinuity");
      return;
    }
    if(last_imu_time_>0 && t-last_imu_time_>0.3 && initialized_) fail("imu_gap");
    last_imu_time_=t;
    if(!initialized_||failed_)return;
    XRSLAMGyroscope gyro{};
    gyro.timestamp=t;gyro.data[0]=m.angular_velocity.x;
    gyro.data[1]=m.angular_velocity.y;gyro.data[2]=m.angular_velocity.z;
    XRSLAMAcceleration accel{};
    accel.timestamp=t;accel.data[0]=m.linear_acceleration.x;
    accel.data[1]=m.linear_acceleration.y;accel.data[2]=m.linear_acceleration.z;
    // Both channels, exactly one ROS IMU stamp.
    XRSLAMPushSensorData(XRSLAM_SENSOR_GYROSCOPE,&gyro);
    XRSLAMPushSensorData(XRSLAM_SENSOR_ACCELERATION,&accel);
    drain();
  }
  void image(sensor_msgs::msg::Image::ConstSharedPtr m) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(failed_)return;
    const double t=seconds(m->header.stamp);
    if(m->header.frame_id!=camera_frame_ || t<=0)return;
    if(!pending_.empty() && t<=seconds(pending_.back()->header.stamp))return;
    if(t<=last_image_time_)return;
    if(pending_.size()>=8){ fail("camera_queue_overflow");return; }
    pending_.push_back(m);
    if(initialized_)drain();
  }
  void poll() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(failed_)return;
    if(!initialized_) {
      try{ if(!init_locked())return; }catch(const std::exception & e){
        RCLCPP_ERROR(get_logger(),"Calibration/initialization failure: %s",e.what());
        fail("configuration_error");return;
      }
    }
    drain();
    read_pose();
  }
  void drain() {
    if(!initialized_||failed_)return;
    while(!pending_.empty()) {
      const auto m=pending_.front();
      const double t=seconds(m->header.stamp);
      if(last_imu_time_ < t+offset_+0.001)break;
      if(t<=last_image_time_){pending_.pop_front();continue;}
      if(m->width!=info_->width||m->height!=info_->height) {fail("image_geometry_changed");return;}
      // Preserve RGB/BGR conventions expected by XRSLAMManager::PushImage.
      int channels=0;
      if(m->encoding=="mono8") channels=1;
      else if(m->encoding=="bgr8"||m->encoding=="rgb8")channels=3;
      else if(m->encoding=="bgra8"||m->encoding=="rgba8")channels=4;
      else {fail("unsupported_image_encoding");return;}
      if(m->step<m->width*static_cast<unsigned>(channels) ||
         m->data.size()<static_cast<size_t>(m->step)*m->height) {
        fail("malformed_image");return;
      }
      std::vector<unsigned char> converted;
      const unsigned char* data=m->data.data();
      if(m->encoding=="rgb8"||m->encoding=="rgba8") {
        converted=m->data;
        for(size_t row=0;row<m->height;++row)
          for(size_t col=0;col<m->width;++col) {
            auto * px=converted.data()+row*m->step+col*channels;
            std::swap(px[0],px[2]);
          }
        data=converted.data();
      }
      XRSLAMImage camera{};
      camera.data=const_cast<unsigned char*>(data);
      camera.timeStamp=t;camera.camera_id=0;
      camera.channel=channels;camera.stride=static_cast<int>(m->step);
      XRSLAMPushSensorData(XRSLAM_SENSOR_CAMERA,&camera);
      XRSLAMRunOneFrame();
      last_image_time_=t;
      pending_.pop_front();
    }
  }
  void read_pose() {
    if(!initialized_||failed_)return;
    XRSLAMState state=XRSLAM_STATE_INITIALIZING;
    XRSLAMGetResult(XRSLAM_RESULT_STATE,&state);
    if(state==XRSLAM_STATE_TRACKING_FAIL){fail("tracking_lost");return;}
    if(state!=XRSLAM_STATE_TRACKING_SUCCESS){
      if(last_pose_time_>0){fail("tracking_session_reset");return;}
      announce("initializing",false);
      map_announce(enable_map_tf_?"waiting_tracking":"disabled",false);
      return;
    }
    XRSLAMVerifiedMapAlignment map_alignment{};
    const bool map_ready=read_verified_map_alignment(map_alignment);
    if(last_pose_time_>0 && enable_map_tf_ &&
       prior_map_epoch_!=0 && map_alignment.tracking_session_epoch!=prior_map_epoch_) {
      fail("map_tracking_epoch_changed");return;
    }
    XRSLAMPose p{};
    XRSLAMGetResult(XRSLAM_RESULT_BODY_POSE,&p);
    const double qnorm=p.quaternion[0]*p.quaternion[0]+p.quaternion[1]*p.quaternion[1]+
      p.quaternion[2]*p.quaternion[2]+p.quaternion[3]*p.quaternion[3];
    if(p.timestamp<=0||!finite(p.timestamp)||qnorm<0.99||qnorm>1.01||
       !finite(p.translation[0])||!finite(p.translation[1])||!finite(p.translation[2]))return;
    if(p.timestamp<=last_pose_time_)return;
    if(!pose_validated_) {
      announce("pose_convention_unverified",false);
      map_announce("pose_convention_unverified",false);
      return;
    }
    const rclcpp::Time stamp(static_cast<int64_t>(std::llround(p.timestamp*1e9)),RCL_ROS_TIME);
    geometry_msgs::msg::Pose pose;
    pose.position.x=p.translation[0];pose.position.y=p.translation[1];pose.position.z=p.translation[2];
    pose.orientation.x=p.quaternion[0];pose.orientation.y=p.quaternion[1];
    pose.orientation.z=p.quaternion[2];pose.orientation.w=p.quaternion[3];
    tf2::Quaternion q(pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w);
    q.normalize();
    if(last_pose_time_>0) {
      const double dt=p.timestamp-last_pose_time_;
      const double dx=pose.position.x-last_pose_.position.x;
      const double dy=pose.position.y-last_pose_.position.y;
      const double dz=pose.position.z-last_pose_.position.z;
      tf2::Quaternion prev(last_pose_.orientation.x,last_pose_.orientation.y,
                            last_pose_.orientation.z,last_pose_.orientation.w);
      prev.normalize();
      if(dt>0.5||std::sqrt(dx*dx+dy*dy+dz*dz)>max_jump_||
         prev.angleShortestPath(q)>max_angular_jump_) {
        fail("pose_discontinuity");return;
      }
    }
    nav_msgs::msg::Odometry odom;
    odom.header.stamp=stamp;odom.header.frame_id=odom_frame_;odom.child_frame_id=base_frame_;
    odom.pose.pose=pose;
    // The C API exposes no covariance. Do not claim estimator certainty.
    for(int i=0;i<36;++i){odom.pose.covariance[i]=0;odom.twist.covariance[i]=0;}
    for(int i=0;i<6;++i){odom.pose.covariance[i*6+i]=100;odom.twist.covariance[i*6+i]=100;}
    if(last_pose_time_>0) {
      const double dt=p.timestamp-last_pose_time_;
      tf2::Vector3 displacement(pose.position.x-last_pose_.position.x,
                                pose.position.y-last_pose_.position.y,
                                pose.position.z-last_pose_.position.z);
      tf2::Quaternion prev(last_pose_.orientation.x,last_pose_.orientation.y,
                            last_pose_.orientation.z,last_pose_.orientation.w);
      prev.normalize();
      const tf2::Vector3 local=quatRotate(prev.inverse(),displacement)/dt;
      odom.twist.twist.linear.x=local.x();odom.twist.twist.linear.y=local.y();odom.twist.twist.linear.z=local.z();
      auto delta=prev.inverse()*q;delta.normalize();
      const double angle=delta.getAngleShortestPath();
      const auto axis=delta.getAxis();
      odom.twist.twist.angular.x=axis.x()*angle/dt;
      odom.twist.twist.angular.y=axis.y()*angle/dt;
      odom.twist.twist.angular.z=axis.z()*angle/dt;
    }
    // First velocity is unknown: no zero-as-measurement output.
    if(last_pose_time_<=0){last_pose_time_=p.timestamp;last_pose_=pose;announce("tracking_velocity_warmup",false);return;}
    geometry_msgs::msg::PoseStamped stamped;
    stamped.header=odom.header;stamped.pose=pose;
    pose_pub_->publish(stamped);
    odom_pub_->publish(odom);
    geometry_msgs::msg::TransformStamped tf;
    tf.header=odom.header;tf.child_frame_id=base_frame_;
    tf.transform.translation.x=pose.position.x;
    tf.transform.translation.y=pose.position.y;
    tf.transform.translation.z=pose.position.z;
    tf.transform.rotation=pose.orientation;
    tf_broadcaster_.sendTransform(tf);
    last_pose_time_=p.timestamp;last_pose_=pose;
    announce("tracking",true);
    if(map_ready) {
      // Compare the status for the same source session again immediately
      // before publishing the corresponding dynamic transform.
      XRSLAMVerifiedMapAlignment second{};
      const bool confirmed=read_verified_map_alignment(second);
      if(confirmed &&
         second.tracking_session_epoch==map_alignment.tracking_session_epoch &&
         second.reference_session_id==map_alignment.reference_session_id &&
         second.verified_timestamp<=p.timestamp+1e-6) {
        // T_map_odom = T_map_vio * T_vio_base * inverse(T_odom_base).
        // Here the adapter's T_odom_base is the same local body pose,
        // but retain the composition to document and test frame direction.
        tf2::Quaternion vio_q(pose.orientation.x,pose.orientation.y,
                              pose.orientation.z,pose.orientation.w);
        vio_q.normalize();
        const tf2::Vector3 vio_xyz(pose.position.x,pose.position.y,pose.position.z);
        const tf2::Transform T_vio_base(vio_q,vio_xyz);
        const tf2::Transform T_odom_base(vio_q,vio_xyz);
        tf2::Quaternion map_q(tf2::Vector3(0,0,1),second.yaw_rad);
        const tf2::Transform T_map_vio(
          map_q,tf2::Vector3(second.translation[0],second.translation[1],
                            second.translation[2]));
        const tf2::Transform T_map_odom =
          T_map_vio*T_vio_base*T_odom_base.inverse();
        geometry_msgs::msg::TransformStamped global_tf;
        global_tf.header.stamp=stamp;
        global_tf.header.frame_id=map_frame_;
        global_tf.child_frame_id=odom_frame_;
        global_tf.transform=tf2::toMsg(T_map_odom);
        tf_broadcaster_.sendTransform(global_tf);
        prior_map_epoch_=second.tracking_session_epoch;
        map_announce("verified:"+map_id_,true);
      }else{
        map_announce("map_alignment_changed_or_not_current",false);
      }
    }else if(enable_map_tf_){
      map_announce("waiting_verified_map_alignment",false);
    }
  }
  void fail(const std::string & reason) {
    if(failed_)return;
    failed_=true;pending_.clear();
    announce(reason,false);
    map_announce(reason,false);
    RCLCPP_ERROR(get_logger(),"Odometry inhibited: %s (restart requires supervised validation)",reason.c_str());
  }

  std::mutex mutex_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr valid_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr map_valid_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr map_status_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::optional<sensor_msgs::msg::CameraInfo> info_;
  std::deque<sensor_msgs::msg::Image::ConstSharedPtr> pending_;
  std::string slam_yaml_,template_yaml_,image_topic_,info_topic_,imu_topic_;
  std::string camera_frame_,imu_frame_,base_frame_,odom_frame_,generated_yaml_;
  std::string map_frame_,map_id_,map_status_;
  std::string last_status_;
  bool last_valid_=false,initialized_=false,failed_=false,pose_validated_=false;
  bool enable_map_tf_=false,map_valid_=false;
  std::uint64_t prior_map_epoch_=0;
  double offset_=0,max_jump_=1,max_angular_jump_=1;
  double last_imu_time_=-1,last_image_time_=-1,last_pose_time_=-1;
  geometry_msgs::msg::Pose last_pose_;
};

int main(int argc,char ** argv) {
  rclcpp::init(argc,argv);
  try {
    auto node=std::make_shared<XrslamRos2>();
    rclcpp::spin(node);
  }catch(const std::exception & e) {
    std::fprintf(stderr,"xrslam_ros2 fatal: %s\n",e.what());
    rclcpp::shutdown();return 1;
  }
  rclcpp::shutdown();return 0;
}
