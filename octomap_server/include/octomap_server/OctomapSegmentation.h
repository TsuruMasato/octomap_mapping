#pragma once
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/OccupancyGrid.h>
#include <std_msgs/ColorRGBA.h>

// #include <moveit_msgs/CollisionObject.h>
// #include <moveit_msgs/CollisionMap.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_srvs/Empty.h>
#include <dynamic_reconfigure/server.h>
#include <octomap_server/OctomapServerConfig.h>

#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/filters/random_sample.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations" // pcl::SAC_SAMPLE_SIZE is protected since PCL 1.8.0
#include <pcl/sample_consensus/model_types.h>
#pragma GCC diagnostic pop

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/pca.h>
#include <pcl/surface/convex_hull.h>>

#include <tf/transform_listener.h>
#include <tf/message_filter.h>
#include <message_filters/subscriber.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/GetOctomap.h>
#include <octomap_msgs/BoundingBoxQuery.h>
#include <octomap_msgs/conversions.h>

#include <octomap_ros/conversions.h>
#include <octomap/octomap.h>
#include <octomap/OcTreeKey.h>
#include <octomap_server/OctomapServer.h>

using namespace octomap;
using namespace octomap_server;

class OctomapSegmentation
{
  typedef pcl::PointXYZRGBNormal PCLPoint;
  typedef pcl::PointCloud<pcl::PointXYZRGBNormal> PCLPointCloud;

public:
  OctomapSegmentation();
  ~OctomapSegmentation(){};
  void set_frame_id(const std::string &world_frame_id) { frame_id_ = world_frame_id; };
  void set_camera_initial_height(const double &input_height) { camera_initial_height_ = input_height; };

  /* main function */
  pcl::PointCloud<pcl::PointXYZRGB> segmentation(octomap_server::OctomapServer::OcTreeT* &target_octomap, visualization_msgs::MarkerArray &marker_array);
  bool remove_floor_RANSAC(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &input, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &floor_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &obstacle_cloud);
  pcl::ModelCoefficients ransac_horizontal_plane(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &input_cloud, const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &floor_cloud, const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &obstacle_cloud, double plane_thickness = 0.1, const Eigen::Vector3f &axis = Eigen::Vector3f(0.0, 0.0, 1.0));
  bool clustering(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input_cloud, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &output_results);
  void change_colors_debug(std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &clusters);
  void add_color_and_accumulate(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &point_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result_cloud, uint8_t r = 255, uint8_t g = 255, uint8_t b = 255);
  void add_color_and_accumulate(std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &clusters, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result_cloud, uint8_t r = 255, uint8_t g = 255, uint8_t b = 255);
  static bool CustomCondition(const pcl::PointXYZRGBNormal &seedPoint, const pcl::PointXYZRGBNormal &candidatePoint, float squaredDistance);
  bool PCA_classify(std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &input_clusters, visualization_msgs::MarkerArray &marker_array, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &cubic_clusters, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &plane_clusters, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &sylinder_clusters, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &the_others);
  bool ransac_wall_detection(std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &input_clusters);
  void add_wall_marker(pcl::PCA<PCLPoint> &pca, int &marker_id, visualization_msgs::MarkerArray &marker_array, const std::string &frame_id);
  void add_floor_marker(pcl::PCA<PCLPoint> &pca, int &marker_id, visualization_msgs::MarkerArray &marker_array, const std::string &frame_id);
  void add_step_marker(pcl::PCA<PCLPoint> &pca, int &marker_id, visualization_msgs::MarkerArray &marker_array, const std::string &frame_id);
  void add_handrail_marker(pcl::PCA<PCLPoint> &pca, int &marker_id, visualization_msgs::MarkerArray &marker_array, const std::string &frame_id);
  void add_cylinder_marker(pcl::PCA<PCLPoint> &pca, int &marker_id, visualization_msgs::MarkerArray &marker_array, const std::string &frame_id);

  /* floor_removal() */
  bool isSpeckleNode(const OcTreeKey &nKey, octomap_server::OctomapServer::OcTreeT *&target_octomap);

private:
  std::string frame_id_ = "map";
  double camera_initial_height_ = 1.05;
  int id_;
  // visualization_msgs::MarkerArray marker_array_;
  bool convert_octomap_to_pcl_cloud(octomap_server::OctomapServer::OcTreeT* &input_octomap, const pcl::PointCloud<PCLPoint>::Ptr &output_cloud);
  geometry_msgs::Point convert_eigen_to_geomsg(const Eigen::Vector3f input_vector);
  void computeOBB(const pcl::PointCloud<PCLPoint>::Ptr &input_cloud, pcl::PCA<PCLPoint> &input_pca, Eigen::Vector3f &min_point, Eigen::Vector3f &max_point, Eigen::Vector3f &OBB_center, Eigen::Matrix3f &obb_rotational_matrix);
  void add_OBB_marker(const Eigen::Vector3f &min_obb, const Eigen::Vector3f &max_obb, const Eigen::Vector3f &center_obb, const Eigen::Matrix3f &rot_obb, int &marker_id, visualization_msgs::MarkerArray &marker_array, const std::string &frame_id);
  void add_line_marker(const pcl::PointCloud<PCLPoint>::Ptr &input_vertices, const std::vector<pcl::Vertices> &input_surface, const std::vector<uint8_t> &rgb, int &marker_id, visualization_msgs::MarkerArray &marker_array, std::string frame_id);
  // ros::NodeHandle nh_;
  // ros::Publisher pub_segmented_pc_;
  // ros::Publisher pub_normal_vector_markers_;
};