/*
 * Copyright (c) 2010-2013, A. Hornung, University of Freiburg
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University of Freiburg nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef OCTOMAP_SERVER_OCTOMAPSERVER_H
#define OCTOMAP_SERVER_OCTOMAPSERVER_H

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/OccupancyGrid.h>
#include <std_msgs/ColorRGBA.h>
#include <deque>
#include <unordered_map>

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
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"  // pcl::SAC_SAMPLE_SIZE is protected since PCL 1.8.0
#include <pcl/sample_consensus/model_types.h>
#pragma GCC diagnostic pop

#include <pcl/segmentation/sac_segmentation.h>
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

//#define COLOR_OCTOMAP_SERVER // switch color here - easier maintenance, only maintain OctomapServer. Two targets are defined in the cmake, octomap_server_color and octomap_server. One has this defined, and the other doesn't
#define EXTEND_OCTOMAP_SERVER

#ifdef COLOR_OCTOMAP_SERVER
#include <octomap/ColorOcTree.h>
#endif

using namespace octomap;

namespace octomap_server {


/* definitions for ExOctomap class */

// forward declaraton for "friend"
class ExOcTree;
// node definition
class ExOcTreeNode : public OcTreeNode
{
public:
  friend class ExOcTree; // needs access to node children (inherited)

  class Color
  {
  public:
    Color() : r(255), g(255), b(255) {}
    Color(uint8_t _r, uint8_t _g, uint8_t _b)
        : r(_r), g(_g), b(_b) {}
    inline bool operator==(const Color &other) const
    {
      return (r == other.r && g == other.g && b == other.b);
    }
    inline bool operator!=(const Color &other) const
    {
      return (r != other.r || g != other.g || b != other.b);
    }
    uint8_t r, g, b;
  };

  enum ShapePrimitive
  {
    WALL,
    FLOOR,
    CEILING,
    HANDRAIL,
    OTHER,
    FREE
  };

  enum ContactType
  {
    SURFACE,
    GRASP,
    LANDING,
    NOT_AVAILABLE
  };

public:
  ExOcTreeNode() : OcTreeNode() {}

  ExOcTreeNode(const ExOcTreeNode &rhs) : OcTreeNode(rhs), color(rhs.color) {}

  bool operator==(const ExOcTreeNode &rhs) const
  {
    return (rhs.value == value && rhs.color == color);
  }

  void copyData(const ExOcTreeNode &from)
  {
    OcTreeNode::copyData(from);
    this->color = from.getColor();
  }

  inline Color getColor() const { return color; }
  inline void setColor(Color c) { this->color = c; }
  inline void setColor(uint8_t r, uint8_t g, uint8_t b)
  {
    this->color = Color(r, g, b);
  }

  Color &getColor() { return color; }

  // has any color been integrated? (pure white is very unlikely...)
  inline bool isColorSet() const
  {
    return ((color.r != 255) || (color.g != 255) || (color.b != 255));
  }

  void updateColorChildren();

  ExOcTreeNode::Color getAverageChildColor() const;

  /* shape primitive*/
  inline ShapePrimitive getPrimitive() const { return shape_primitive; }
  inline void setPrimitive(ShapePrimitive p) { shape_primitive = p; }
  inline void averagePrimitive(ShapePrimitive p)
  {
    latest_10_shape_primitives.push_back(p);
    while(latest_10_shape_primitives.size() > 10)
    {
      latest_10_shape_primitives.pop_front();
    }
    ShapePrimitive mode_p = extract_mode_primitive(latest_10_shape_primitives);
    setPrimitive(mode_p);
  }

  inline ShapePrimitive extract_mode_primitive(std::deque<ShapePrimitive> &input_primitive_deque)
  {
    std::unordered_map<ShapePrimitive, size_t> count_table;
    for (auto itr = input_primitive_deque.begin(); itr != input_primitive_deque.end(); itr++)
    {
      if(count_table.find(*itr) != count_table.end())
      {
        count_table.at(*itr)++;
      }
      else
      {
        count_table[*itr] = 1;
      }
    }
    auto max_itr = std::max_element(count_table.begin(), count_table.end(), [](const auto &a, const auto &b) -> bool 
    {
      return (a.second < b.second);
    }
    );
    ExOcTreeNode::ShapePrimitive p_mode = max_itr->first;
    return p_mode;
  }

  inline Eigen::Vector3d getNormalVector() const { return normal_vector; }

  inline void setNormalVector(Eigen::Vector3d n_vector_input)
  {
    normal_vector = n_vector_input;
  }

  inline void setNormalVector(double input_x, double input_y, double input_z)
  {
    normal_vector << input_x, input_y, input_z;
  }

  inline bool isNormalVectorSet() const
  {
    return ((normal_vector.x() != 0) || (normal_vector.y() != 0) || (normal_vector.z() != 0));
  }

  inline bool isAffordanceReady() const { return affordance_ready; }

  inline bool setAffordanceReady(bool input)
  {
    affordance_ready = input;
    return affordance_ready;
  }

  Eigen::Vector3d getApproachDirection() { return approach_direction; }

  void setApproachDirection(Eigen::Vector3d input_vec)
  {
    approach_direction = input_vec;
  }

  ContactType getContactType() const { return contact_type; }

  void setContactType(ContactType input_contact_type)
  {
    contact_type = input_contact_type;
  }

  // file I/O
  std::istream &readData(std::istream &s);
  std::ostream &writeData(std::ostream &s) const;

protected:
  Color color;
  ShapePrimitive shape_primitive;
  std::deque<ShapePrimitive> latest_10_shape_primitives;
  Eigen::Vector3d normal_vector{0.0, 0.0, 0.0};
  bool affordance_ready = false;
  Eigen::Vector3d approach_direction;
  ContactType contact_type = ContactType::NOT_AVAILABLE;
};

// tree definition
class ExOcTree : public OccupancyOcTreeBase<ExOcTreeNode>
{

public:
  /// Default constructor, sets resolution of leafs
  ExOcTree(double resolution);

  /// virtual constructor: creates a new object of same type
  /// (Covariant return type requires an up-to-date compiler)
  ExOcTree *create() const { return new ExOcTree(resolution); }

  std::string getTreeType() const { return "ColorOcTree"; }

  /**
   * Prunes a node when it is collapsible. This overloaded
   * version only considers the node occupancy for pruning,
   * different colors of child nodes are ignored.
   * @return true if pruning was successful
   */
  virtual bool pruneNode(ExOcTreeNode *node);

  virtual bool isNodeCollapsible(const ExOcTreeNode *node) const;

  // set node color at given key or coordinate. Replaces previous color.
  ExOcTreeNode *setNodeColor(const OcTreeKey &key, uint8_t r,
                             uint8_t g, uint8_t b);

  ExOcTreeNode *setNodeColor(float x, float y,
                             float z, uint8_t r,
                             uint8_t g, uint8_t b)
  {
    OcTreeKey key;
    if (!this->coordToKeyChecked(point3d(x, y, z), key))
      return NULL;
    return setNodeColor(key, r, g, b);
  }

  // integrate color measurement at given key or coordinate. Average with previous color
  ExOcTreeNode *averageNodeColor(const OcTreeKey &key, uint8_t r,
                                 uint8_t g, uint8_t b);

  ExOcTreeNode *averageNodeColor(float x, float y,
                                 float z, uint8_t r,
                                 uint8_t g, uint8_t b)
  {
    OcTreeKey key;
    if (!this->coordToKeyChecked(point3d(x, y, z), key))
      return NULL;
    return averageNodeColor(key, r, g, b);
  }

  // integrate color measurement at given key or coordinate. Average with previous color
  ExOcTreeNode *integrateNodeColor(const OcTreeKey &key, uint8_t r,
                                   uint8_t g, uint8_t b);

  ExOcTreeNode *integrateNodeColor(float x, float y,
                                   float z, uint8_t r,
                                   uint8_t g, uint8_t b)
  {
    OcTreeKey key;
    if (!this->coordToKeyChecked(point3d(x, y, z), key))
      return NULL;
    return integrateNodeColor(key, r, g, b);
  }

  ExOcTreeNode *averageNodeNormalVector(const OcTreeKey &key, float n_vec_x, float n_vec_y, float n_vec_z);

  ExOcTreeNode *averageNodeNormalVector(float x, float y, float z, float n_vec_x, float n_vec_y, float n_vec_z)
  {
    OcTreeKey key;
    if (!this->coordToKeyChecked(point3d(x, y, z), key))
      return NULL;
    return averageNodeNormalVector(key, n_vec_x, n_vec_y, n_vec_z);
  }

  ExOcTreeNode *setNodeNormalVector(const OcTreeKey &key,
                                         const Eigen::Vector3d n_vec);

  ExOcTreeNode *averageNodePrimitive(const OcTreeKey &key, ExOcTreeNode::ShapePrimitive p);

  ExOcTreeNode *averageNodePrimitive(float x, float y, float z, ExOcTreeNode::ShapePrimitive p) // overlay for (x,y,z) argument
  {
    OcTreeKey key;
    if (!this->coordToKeyChecked(point3d(x, y, z), key))
      return NULL;
    return averageNodePrimitive(key, p);
  }
  ExOcTreeNode *setNodePrimitive(const OcTreeKey &key,
                                 ExOcTreeNode::ShapePrimitive p);

  // update inner nodes, sets color to average child color
  void updateInnerOccupancy();

  // uses gnuplot to plot a RGB histogram in EPS format
  void writeColorHistogram(std::string filename);

protected:
  void updateInnerOccupancyRecurs(ExOcTreeNode *node, unsigned int depth);

  /**
   * Static member object which ensures that this OcTree's prototype
   * ends up in the classIDMapping only once. You need this as a 
   * static member in any derived octree class in order to read .ot
   * files through the AbstractOcTree factory. You should also call
   * ensureLinking() once from the constructor.
   */
  class StaticMemberInitializer
  {
  public:
    StaticMemberInitializer()
    {
      ExOcTree *tree = new ExOcTree(0.1);
      tree->clearKeyRays();
      AbstractOcTree::registerTreeType(tree);
    }

    /**
       * Dummy function to ensure that MSVC does not drop the
       * StaticMemberInitializer, causing this tree failing to register.
       * Needs to be called from the constructor of this octree.
       */
    void ensureLinking(){};
  };
  /// static member to ensure static initialization (only once)
  static StaticMemberInitializer ExOcTreeMemberInit;
};

//! user friendly output in format (r g b)
std::ostream &operator<<(std::ostream &out, ExOcTreeNode::Color const &c);



/* main process implementation as the server system */

class OctomapServer {

public:
#ifdef COLOR_OCTOMAP_SERVER
  typedef pcl::PointXYZRGB PCLPoint;
  typedef pcl::PointCloud<pcl::PointXYZRGB> PCLPointCloud;
  // typedef octomap::ColorOcTree OcTreeT;
#elif defined(EXTEND_OCTOMAP_SERVER)
  typedef pcl::PointXYZRGBNormal PCLPoint;
  typedef pcl::PointCloud<pcl::PointXYZRGBNormal> PCLPointCloud;
  typedef ExOcTree OcTreeT;
#else
  typedef pcl::PointXYZ PCLPoint;
  typedef pcl::PointCloud<pcl::PointXYZ> PCLPointCloud;
  typedef octomap::OcTree OcTreeT;
#endif
  typedef octomap_msgs::GetOctomap OctomapSrv;
  typedef octomap_msgs::BoundingBoxQuery BBXSrv;

  OctomapServer(const ros::NodeHandle private_nh_ = ros::NodeHandle("~"), const ros::NodeHandle &nh_ = ros::NodeHandle());
  virtual ~OctomapServer();
  virtual bool octomapBinarySrv(OctomapSrv::Request  &req, OctomapSrv::GetOctomap::Response &res);
  virtual bool octomapFullSrv(OctomapSrv::Request  &req, OctomapSrv::GetOctomap::Response &res);
  bool clearBBXSrv(BBXSrv::Request& req, BBXSrv::Response& resp);
  bool resetSrv(std_srvs::Empty::Request& req, std_srvs::Empty::Response& resp);

  virtual void insertCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud);
  virtual bool openFile(const std::string& filename);

  /// Test if key is within update area of map (2D, ignores height)
  inline bool isInUpdateBBX(const OcTreeT::iterator &it) const
  {
    // 2^(tree_depth-depth) voxels wide:
    unsigned voxelWidth = (1 << (m_maxTreeDepth - it.getDepth()));
    octomap::OcTreeKey key = it.getIndexKey(); // lower corner of voxel
    return (key[0] + voxelWidth >= m_updateBBXMin[0] && key[1] + voxelWidth >= m_updateBBXMin[1] && key[0] <= m_updateBBXMax[0] && key[1] <= m_updateBBXMax[1]);
  }

protected:
  inline static void updateMinKey(const octomap::OcTreeKey& in, octomap::OcTreeKey& min) {
    for (unsigned i = 0; i < 3; ++i)
      min[i] = std::min(in[i], min[i]);
  };

  inline static void updateMaxKey(const octomap::OcTreeKey& in, octomap::OcTreeKey& max) {
    for (unsigned i = 0; i < 3; ++i)
      max[i] = std::max(in[i], max[i]);
  };


  void reconfigureCallback(octomap_server::OctomapServerConfig& config, uint32_t level);
  void publishBinaryOctoMap(const ros::Time& rostime = ros::Time::now()) const;
  void publishFullOctoMap(const ros::Time& rostime = ros::Time::now()) const;
  virtual void publishAll(const ros::Time& rostime = ros::Time::now());

  /**
  * @brief update occupancy map with a scan labeled as ground and nonground.
  * The scans should be in the global map frame.
  *
  * @param sensorOrigin origin of the measurements for raycasting
  * @param ground scan endpoints on the ground plane (only clear space)
  * @param nonground all other endpoints (clear up to occupied endpoint)
  */
  virtual void insertScan(const tf::Point& sensorOrigin, const PCLPointCloud& ground, const PCLPointCloud& nonground);
  virtual void insertScanWithPrimitives(const tf::Point &sensorOriginTf, const std::vector<PCLPointCloud> &pc_array, const std::vector<ExOcTreeNode::ShapePrimitive> &primitive_array);

  /// label the input cloud "pc" into ground and nonground. Should be in the robot's fixed frame (not world!)
  void filterGroundPlane(const PCLPointCloud &pc, PCLPointCloud &ground, PCLPointCloud &nonground) const;

  /**
  * @brief Find speckle nodes (single occupied voxels with no neighbors). Only works on lowest resolution!
  * @param key
  * @return
  */
  bool isSpeckleNode(const octomap::OcTreeKey& key) const;

  /// hook that is called before traversing all nodes
  virtual void handlePreNodeTraversal(const ros::Time& rostime);

  /// hook that is called when traversing all nodes of the updated Octree (does nothing here)
  virtual void handleNode(const OcTreeT::iterator& it) {};

  /// hook that is called when traversing all nodes of the updated Octree in the updated area (does nothing here)
  virtual void handleNodeInBBX(const OcTreeT::iterator& it) {};

  /// hook that is called when traversing occupied nodes of the updated Octree
  virtual void handleOccupiedNode(const OcTreeT::iterator& it);

  /// hook that is called when traversing occupied nodes in the updated area (updates 2D map projection here)
  virtual void handleOccupiedNodeInBBX(const OcTreeT::iterator& it);

  /// hook that is called when traversing free nodes of the updated Octree
  virtual void handleFreeNode(const OcTreeT::iterator& it);

  /// hook that is called when traversing free nodes in the updated area (updates 2D map projection here)
  virtual void handleFreeNodeInBBX(const OcTreeT::iterator& it);

  /// hook that is called after traversing all nodes
  virtual void handlePostNodeTraversal(const ros::Time& rostime);

  /// updates the downprojected 2D map as either occupied or free
  virtual void update2DMap(const OcTreeT::iterator& it, bool occupied);

  inline unsigned mapIdx(int i, int j) const {
    return m_gridmap.info.width * j + i;
  }

  inline unsigned mapIdx(const octomap::OcTreeKey& key) const {
    return mapIdx((key[0] - m_paddedMinKey[0]) / m_multires2DScale,
                  (key[1] - m_paddedMinKey[1]) / m_multires2DScale);

  }

  /**
   * Adjust data of map due to a change in its info properties (origin or size,
   * resolution needs to stay fixed). map already contains the new map info,
   * but the data is stored according to oldMapInfo.
   */

  void adjustMapData(nav_msgs::OccupancyGrid& map, const nav_msgs::MapMetaData& oldMapInfo) const;

  inline bool mapChanged(const nav_msgs::MapMetaData& oldMapInfo, const nav_msgs::MapMetaData& newMapInfo) {
    return (    oldMapInfo.height != newMapInfo.height
                || oldMapInfo.width != newMapInfo.width
                || oldMapInfo.origin.position.x != newMapInfo.origin.position.x
                || oldMapInfo.origin.position.y != newMapInfo.origin.position.y);
  }

  static std_msgs::ColorRGBA heightMapColor(double h);
  ros::NodeHandle m_nh;
  ros::NodeHandle m_nh_private;
  ros::Publisher  m_markerPub, m_binaryMapPub, m_fullMapPub, m_pointCloudPub, m_collisionObjectPub, m_mapPub, m_cmapPub, m_fmapPub, m_fmarkerPub, m_normalVectorPub;
  message_filters::Subscriber<sensor_msgs::PointCloud2>* m_pointCloudSub;
  tf::MessageFilter<sensor_msgs::PointCloud2>* m_tfPointCloudSub;
  ros::ServiceServer m_octomapBinaryService, m_octomapFullService, m_clearBBXService, m_resetService;
  tf::TransformListener m_tfListener;
  boost::recursive_mutex m_config_mutex;
  dynamic_reconfigure::Server<OctomapServerConfig> m_reconfigureServer;

  OcTreeT* m_octree;
  octomap::KeyRay m_keyRay;  // temp storage for ray casting
  octomap::OcTreeKey m_updateBBXMin;
  octomap::OcTreeKey m_updateBBXMax;

  double m_maxRange;
  std::string m_worldFrameId; // the map frame
  std::string m_baseFrameId; // base of the robot for ground plane filtering
  bool m_useHeightMap;
  std_msgs::ColorRGBA m_color;
  std_msgs::ColorRGBA m_colorFree;
  double m_colorFactor;

  bool m_latchedTopics;
  bool m_publishFreeSpace;

  double m_res;
  unsigned m_treeDepth;
  unsigned m_maxTreeDepth;

  double m_pointcloudMinX;
  double m_pointcloudMaxX;
  double m_pointcloudMinY;
  double m_pointcloudMaxY;
  double m_pointcloudMinZ;
  double m_pointcloudMaxZ;
  double m_occupancyMinZ;
  double m_occupancyMaxZ;
  double m_minSizeX;
  double m_minSizeY;
  bool m_filterSpeckles;

  bool m_filterGroundPlane;
  double m_groundFilterDistance;
  double m_groundFilterAngle;
  double m_groundFilterPlaneDistance;

  double m_camera_initial_height;

  bool m_compressMap;

  bool m_initConfig;

  // downprojected 2D map:
  bool m_incrementalUpdate;
  nav_msgs::OccupancyGrid m_gridmap;
  bool m_publish2DMap;
  bool m_mapOriginChanged;
  octomap::OcTreeKey m_paddedMinKey;
  unsigned m_multires2DScale;
  bool m_projectCompleteMap;
  bool m_useColoredMap;

  // Tsuru add:
  bool use_virtual_wall_;
  PCLPointCloud virtual_wall_cloud_;
  bool dynamic_local_mode_;
  float dynamic_area_x_max_, dynamic_area_x_min_, dynamic_area_y_max_, dynamic_area_y_min_;
  double worst_insertion_time_, worst_publication_time_;

  pcl::PointCloud<pcl::PointXYZRGB> segmented_pc_;
  visualization_msgs::MarkerArray marker_array_;

  bool subtract_point_cloud(PCLPointCloud::Ptr point_cloud);
  };
}

#endif
