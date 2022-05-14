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

  /* main function */
  pcl::PointCloud<pcl::PointXYZRGB> segmentation(octomap_server::OctomapServer::OcTreeT *&target_octomap);
  bool remove_floor_RANSAC(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &input, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &floor_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &obstacle_cloud);
  bool plane_ransac(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &input_cloud, const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &output_cloud, double plane_thickness = 0.1, const Eigen::Vector3f &axis = Eigen::Vector3f(0.0, 0.0, 1.0));
  bool clustering(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input_cloud);
  void change_colors_debug(std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &clusters);
  static bool CustomCondition(const pcl::PointXYZRGBNormal &seedPoint, const pcl::PointXYZRGBNormal &candidatePoint, float squaredDistance);

  /* floor_removal() */
  bool isSpeckleNode(const OcTreeKey &nKey, octomap_server::OctomapServer::OcTreeT *&target_octomap);

private:
  bool tsuru_ = true;
  ros::NodeHandle nh_;
  ros::Publisher pub_segmented_pc_;
};