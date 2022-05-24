#include <octomap_server/OctomapSegmentation.h>

OctomapSegmentation::OctomapSegmentation()
{
  // pub_segmented_pc_ = nh_.advertise<sensor_msgs::PointCloud2>("segmented_pc", 1);
  // pub_normal_vector_markers_ = nh_.advertise<visualization_msgs::MarkerArray>("normal_vectors", 1);
}

pcl::PointCloud<pcl::PointXYZRGB> OctomapSegmentation::segmentation(OctomapServer::OcTreeT* &target_octomap, visualization_msgs::MarkerArray &arrow_markers)
{
  // ROS_ERROR("OctomapSegmentation::segmentation() start");
  // init pointcloud:
  pcl::PointCloud<OctomapServer::PCLPoint>::Ptr pcl_cloud(new pcl::PointCloud<OctomapServer::PCLPoint>);

  // call pre-traversal hook:
  ros::Time rostime = ros::Time::now();

  /* *********************************************************** */
  /* convert Octomap to PCL point cloud for segmentation process */
  /* *********************************************************** */

  // now, traverse all leafs in the tree:
  for (OctomapServer::OcTreeT::iterator it = target_octomap->begin(target_octomap->getTreeDepth()), end = target_octomap->end(); it != end; ++it)
  {
    // bool inUpdateBBX = OctomapServer::isInUpdateBBX(it);
    if (target_octomap->isNodeOccupied(*it))
    {  
      // Ignore speckles in the map:
      if ((it.getDepth() >= target_octomap->getTreeDepth()) && isSpeckleNode(it.getKey(), target_octomap))
      {
        target_octomap->deleteNode(it.getKey(), target_octomap->getTreeDepth());
        // ROS_ERROR("Ignoring single speckle at (%f,%f,%f)", x, y, z);
        continue;
      } // else: current octree node is no speckle, send it out
      

      OctomapServer::PCLPoint point;
      point.x = it.getX();
      point.y = it.getY();
      point.z = it.getZ();
      point.r = it->getColor().r;
      point.g = it->getColor().g;
      point.b = it->getColor().b;
      point.normal_x = it->getNormalVector().x();
      point.normal_y = it->getNormalVector().y();
      point.normal_z = it->getNormalVector().z();
      // double half_size = it.getSize() / 2.0;

      pcl_cloud->push_back(point);
    }
  }

  /* ******************************** */
  /* segmentation in PointCloud style */
  /* ******************************** */

  // Remove Floor plane
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr floor_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr obstacle_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  auto floor_coefficients = ransac_horizontal_plane(/*input*/ pcl_cloud, /*output 1*/ floor_cloud, /*output 2*/ obstacle_cloud);
  if (floor_coefficients.header.frame_id == "FAIL")
  {
    ROS_ERROR("Failed Floor in Octomap by RANSAC!!");
  }
  else
  {
    visualization_msgs::Marker floor_marker;
    floor_marker.header.frame_id = "/map";
    // floor_marker.lifetime = ros::Duration(1.0);
    floor_marker.ns = "cluster_eigen";
    floor_marker.type = visualization_msgs::Marker::CUBE;
    floor_marker.id = 0;
    floor_marker.pose.position.x = 0.0f;
    floor_marker.pose.position.y = 0.0f;
    floor_marker.pose.position.z = -floor_coefficients.values.at(3);
    floor_marker.pose.orientation.x = 0.0f;
    floor_marker.pose.orientation.y = 0.0f;
    floor_marker.pose.orientation.z = 0.0f;
    floor_marker.pose.orientation.w = 1.0f;
    floor_marker.scale.x = 10.0f;
    floor_marker.scale.y = 10.0f;
    floor_marker.scale.z = 0.001f;
    floor_marker.color.r = 0.8;
    floor_marker.color.g = 1.0;
    floor_marker.color.b = 0.9;
    floor_marker.color.a = 1.0;
    arrow_markers.markers.push_back(floor_marker);
  }

  // clustering
  std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> clusters;
  bool clustering_success = clustering(obstacle_cloud, clusters);
  
  // primitive clustering
  // arrow_markers.markers.clear();
  // TODO: multi-thread process.
  std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> cubic_clusters, plane_clusters, sylinder_clusters, the_others;
  PCA_classify(clusters, arrow_markers, cubic_clusters, plane_clusters, sylinder_clusters, the_others);

  // RANSAC wall detection
  // ransac_wall_detection(plane_clusters);

  /* update Octomap according to the PCL segmentation results */

  /* merge the all cluster types with changing colors*/
  pcl::copyPointCloud(*obstacle_cloud, *result_cloud);
  add_color_and_accumulate(floor_cloud, result_cloud, /*rgb*/ 191, 255, 128);
  // add_color_and_accumulate(cubic_clusters, result_cloud, /*rgb*/ 255, 128, 191);
  // add_color_and_accumulate(plane_clusters, result_cloud, /*rgb*/ 255, 191, 128);
  // add_color_and_accumulate(sylinder_clusters, result_cloud, /*rgb*/ 150, 245, 252);
  // add_color_and_accumulate(the_others, result_cloud, /*rgb*/ 100, 100, 100);
  

  pcl::PointCloud<pcl::PointXYZRGB> simplified_pc;
  pcl::copyPointCloud(*result_cloud, simplified_pc);
  return simplified_pc;
};

pcl::ModelCoefficients OctomapSegmentation::ransac_horizontal_plane(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &input_cloud, const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &floor_cloud, const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &obstacle_cloud, double plane_thickness, const Eigen::Vector3f &axis)
{
  // ROS_INFO("ransac_horizontal_plane");
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZRGBNormal> seg;

  // Optional
  seg.setOptimizeCoefficients(true);
  // Mandatory
  seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(200);
  seg.setAxis(axis);
  seg.setEpsAngle(0.05);
  seg.setDistanceThreshold(plane_thickness); // 0.025 0.018
  seg.setInputCloud(input_cloud);
  seg.segment(*inliers, *coefficients);
  // ROS_INFO("plane size : %d", inliers->indices.size());

  if (inliers->indices.size() < 20)
  {
    ROS_WARN("[OctomapSegmentation::ransac_horizontal_plane] no horizontal plane in input_cloud");
    pcl::copyPointCloud(*input_cloud, *obstacle_cloud);
    coefficients->header.frame_id = "FAIL";
    return *coefficients;
  }

  if (-coefficients->values.at(3) < -0.5) // when the height of the plane is enough low:
  {
    // ROS_WARN("floor height : %f", -coefficients->values.at(3));
    pcl::ExtractIndices<pcl::PointXYZRGBNormal> extract;
    extract.setInputCloud(input_cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*floor_cloud);
    extract.setNegative(true);
    extract.filter(*obstacle_cloud);
    return *coefficients;
  }

  else
  {
    // ROS_WARN("ceiling height : %f", -coefficients->values.at(3));
    // the detected plane must be a ceiling. remove it and try RANSAC again.
    pcl::PointCloud<OctomapServer::PCLPoint>::Ptr ceiling_cloud(new pcl::PointCloud<OctomapServer::PCLPoint>);
    pcl::ExtractIndices<pcl::PointXYZRGBNormal> extract;
    extract.setInputCloud(input_cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*ceiling_cloud);
    extract.setNegative(true);
    extract.filter(*input_cloud); // keep the whole cloud in "input_cloud" except the ceiling.

    /* 2nd RANSAC with same params. */
    inliers->indices.clear();
    coefficients->values.clear();
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() < 20)
    {
      ROS_WARN("2nd plane size is not enough large to remove.");
      coefficients->header.frame_id = "FAIL";
      return *coefficients;
    }
    if (-coefficients->values.at(3) < -0.5) // the floor hight must be low.
    {
      // extract.setInputCloud(input_cloud);
      extract.setIndices(inliers);
      extract.setNegative(false);
      extract.filter(*floor_cloud);
      extract.setNegative(true);
      extract.filter(*obstacle_cloud);
      return *coefficients;
    }
    else
    {
      ROS_WARN("[OctomapSegmentation::ransac_horizontal_plane] found undesired horizontal plane (maybe tables)");
      pcl::copyPointCloud(*input_cloud, *obstacle_cloud);
      return *coefficients;
    }
  }
}

bool OctomapSegmentation::clustering(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input_cloud, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &output_results)
{
  // std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> clusters;
  /*クラスタ後のインデックスが格納されるベクトル*/
  std::vector<pcl::PointIndices> cluster_indices;
  /*今回の主役（trueで初期化しないといけないらしい）*/
  pcl::ConditionalEuclideanClustering<pcl::PointXYZRGBNormal> cec(true);
  /*クラスリング対象の点群をinput*/
  cec.setInputCloud(input_cloud);
  /*カスタム条件の関数を指定*/
  cec.setConditionFunction(&OctomapSegmentation::CustomCondition);
  /*距離の閾値を設定*/
  float cluster_tolerance = 0.12;
  cec.setClusterTolerance(cluster_tolerance);
  /*各クラスタのメンバの最小数を設定*/
  int min_cluster_size = 30;
  cec.setMinClusterSize(min_cluster_size);
  /*各クラスタのメンバの最大数を設定*/
  cec.setMaxClusterSize(input_cloud->points.size());
  /*クラスリング実行*/
  cec.segment(cluster_indices);
  /*メンバ数が最小値以下のクラスタと最大値以上のクラスタを取得できる*/
  // cec.getRemovedClusters (small_clusters, large_clusters);

  /*dividing（クラスタごとに点群を分割）*/
  pcl::ExtractIndices<pcl::PointXYZRGBNormal> ei;
  ei.setInputCloud(input_cloud);
  ei.setNegative(false);
  for (size_t i = 0; i < cluster_indices.size(); i++)
  {
    /*extract*/
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr tmp_clustered_points(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointIndices::Ptr tmp_clustered_indices(new pcl::PointIndices);
    *tmp_clustered_indices = cluster_indices[i];
    ei.setIndices(tmp_clustered_indices);
    ei.filter(*tmp_clustered_points);
    /*input*/
    output_results.push_back(tmp_clustered_points);
  }
  ROS_INFO("Clustering finish. It was devided into %d groups", output_results.size());

  change_colors_debug(output_results);
  input_cloud->clear();
  for (size_t i = 0; i < output_results.size(); i++)
  {
    *input_cloud += *output_results.at(i);
  }
  return true;
}

bool OctomapSegmentation::CustomCondition(const pcl::PointXYZRGBNormal &seedPoint, const pcl::PointXYZRGBNormal &candidatePoint, float squaredDistance)
{
  Eigen::Vector3d N1(
      seedPoint.normal_x,
      seedPoint.normal_y,
      seedPoint.normal_z);
  Eigen::Vector3d N2(
      candidatePoint.normal_x,
      candidatePoint.normal_y,
      candidatePoint.normal_z);
  double angle = acos(N1.dot(N2) / N1.norm() / N2.norm()); //法線ベクトル間の角度[rad]

  const double threshold_angle = 3.0; //閾値[deg]
  if (angle / M_PI * 180.0 < threshold_angle)
    return true;
  else
    return false;
}


void OctomapSegmentation::change_colors_debug(std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &clusters)
{
  for (size_t i = 0; i < clusters.size(); i++)
  {
    auto target_cluster_ptr = clusters.at(i);
    uint8_t random_r = rand() % 255;
    uint8_t random_g = rand() % 255;
    uint8_t random_b = rand() % 255;

    // about i-th cluster,
    for (auto itr = target_cluster_ptr->begin(); itr != target_cluster_ptr->end(); itr++)
    {
      itr->r = random_r;
      itr->g = random_g;
      itr->b = random_b;
    }
  }
  return;
}

void OctomapSegmentation::add_color_and_accumulate(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &point_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result_cloud, uint8_t r, uint8_t g, uint8_t b)
{
  for (auto itr = point_cloud->begin(); itr != point_cloud->end(); itr++)
  {
    itr->r = r;
    itr->g = g;
    itr->b = b;
  }
  *result_cloud += *point_cloud;
}

void OctomapSegmentation::add_color_and_accumulate(std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &clusters, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result_cloud, uint8_t r, uint8_t g, uint8_t b)
{
  for (size_t i = 0; i < clusters.size(); i++)
  {
    auto target_cloud_ptr = clusters.at(i);
    for (auto itr = target_cloud_ptr->begin(); itr != target_cloud_ptr->end(); itr++)
    {
      itr->r = r;
      itr->g = g;
      itr->b = b;
    }
    *result_cloud += *target_cloud_ptr;
  }
}

bool OctomapSegmentation::PCA_classify(std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &input_clusters, visualization_msgs::MarkerArray &arrow_array, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &cubic_clusters, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &plane_clusters, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &sylinder_clusters, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &the_others)
{
  int marker_id = 1; // id:0 is floor
  for (size_t i = 0; i < input_clusters.size(); i++)
  {
    auto target_ptr = input_clusters.at(i);
    pcl::PCA<PCLPoint> pca;
    pca.setInputCloud(target_ptr);
    float norm_x = pca.getEigenValues().x();
    float norm_y = pca.getEigenValues().y();
    float norm_z = pca.getEigenValues().z();
    ROS_ERROR("norm_x : %.2f, norm_y : %.2f, norm_z : %.2f", norm_x, norm_y, norm_z);

    bool is_meaningful_x = norm_x > 3.0f;
    bool is_meaningful_y = norm_y > 3.0f;
    bool is_meaningful_z = norm_z > 3.0f;
    int8_t num_meaningful_axis = is_meaningful_x + is_meaningful_y + is_meaningful_z;

    /* if there are two meaningful norms, this cluster is a plane */
    switch (num_meaningful_axis)
    {
    case 3:
      cubic_clusters.push_back(target_ptr);
      break;

    case 2:
      plane_clusters.push_back(target_ptr);
      pushEigenMarker(pca, marker_id, arrow_array, 1.0, "map");
      break;

    case 1:
      sylinder_clusters.push_back(target_ptr);
      break;

    case 0:
      the_others.push_back(target_ptr);
      break;

    default:
      ROS_ERROR("[OctomapSegmentation] failed in switch. num_meaningful_axis : %d", num_meaningful_axis);
      break;
    }    
  }
  return true;
}

void OctomapSegmentation::pushEigenMarker(pcl::PCA<PCLPoint> &pca,
                                          int &marker_id,
                                          visualization_msgs::MarkerArray &arrow_array,
                                          double scale,
                                          const std::string &frame_id)
{
  visualization_msgs::Marker marker;
  // marker.lifetime = ros::Duration(1.0);
  marker.header.frame_id = frame_id;
  marker.ns = "cluster_eigen";
  marker.type = visualization_msgs::Marker::ARROW;
  marker.scale.x = 0.4;   // length:40cm
  marker.scale.y = 0.04;  // width of the allow
  marker.scale.z = 0.04;  // width of the allow

  Eigen::Vector3f center_position;
  center_position << pca.getMean().coeff(0), pca.getMean().coeff(1), pca.getMean().coeff(2);
  marker.pose.position.x = center_position.x();
  marker.pose.position.y = center_position.y();
  marker.pose.position.z = center_position.z();

  Eigen::Quaternionf qx, qy, qz;
  Eigen::Matrix3f eigen_vec = pca.getEigenVectors();
  Eigen::Vector3f eigen_values = pca.getEigenValues();
  Eigen::Vector3f axis_x(eigen_vec.coeff(0, 0), eigen_vec.coeff(1, 0), eigen_vec.coeff(2, 0));
  Eigen::Vector3f axis_y(eigen_vec.coeff(0, 1), eigen_vec.coeff(1, 1), eigen_vec.coeff(2, 1));
  Eigen::Vector3f axis_z(eigen_vec.coeff(0, 2), eigen_vec.coeff(1, 2), eigen_vec.coeff(2, 2));
  qx.setFromTwoVectors(Eigen::Vector3f(1, 0, 0), axis_x);
  qy.setFromTwoVectors(Eigen::Vector3f(1, 0, 0), axis_y);
  qz.setFromTwoVectors(Eigen::Vector3f(1, 0, 0), axis_z);

  /* visualize arrow only with the 3rd Eigen Vector, because it is the plane normal vector.*/
  marker.id = marker_id++;
  marker.pose.orientation.x = qz.x();
  marker.pose.orientation.y = qz.y();
  marker.pose.orientation.z = qz.z();
  marker.pose.orientation.w = qz.w();
  marker.color.b = 0.1;
  marker.color.g = 0.1;
  marker.color.r = 1.0;
  marker.color.a = 1.0;
  arrow_array.markers.push_back(marker);

  // text "plane"
  marker.id = marker_id++;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.pose.position = convert_eigen_to_geomsg(center_position + axis_z * 0.3f + axis_x * 0.2f);
  marker.scale.z = 0.2;
  marker.color.b = 0.9;
  marker.color.g = 0.9;
  marker.color.r = 0.9;
  marker.color.a = 1.0;
  marker.text = "Normal";
  arrow_array.markers.push_back(marker);

  // visualize plane vertices
  /*ゴミコード
  marker.id = marker_id++;
  marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
  marker.pose.orientation.x = qz.x();
  marker.pose.orientation.y = qz.y();
  marker.pose.orientation.z = qz.z();
  marker.pose.orientation.w = qz.w();
  marker.scale.x = 1.0;
  marker.scale.y = 1.0;
  marker.scale.z = 1.0;
  marker.color.r = 1.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;
  marker.color.a = 1.0;

  // calc 4 vertices
  Eigen::Vector3f point_ur, point_ul, point_br, point_bl; // upper right, upper left, bottom right, bottom left
  Eigen::Vector3f point_um, point_bm, point_rm, point_lm;

  point_um = center_position + axis_x * eigen_values.x() * 0.001;
  point_bm = center_position - axis_x * eigen_values.x() * 0.001;
  point_rm = center_position + axis_y * eigen_values.y() * 0.001;
  point_lm = center_position - axis_y * eigen_values.y() * 0.001;

  marker.points.push_back(convert_eigen_to_geomsg(point_um));
  marker.points.push_back(convert_eigen_to_geomsg(point_bm));
  marker.points.push_back(convert_eigen_to_geomsg(point_rm));
  // marker.points.push_back(convert_eigen_to_geomsg(point_lm));

  arrow_array.markers.push_back(marker);
  */
}

bool OctomapSegmentation::ransac_wall_detection(std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &input_clusters)
{
  for (size_t i = 0; i < input_clusters.size(); i++)
  {

  }
  return true;
}

bool OctomapSegmentation::isSpeckleNode(const OcTreeKey &nKey, OctomapServer::OcTreeT* &target_octomap)
{
  OcTreeKey key;
  bool neighborFound = false;
  for (key[2] = nKey[2] - 1; !neighborFound && key[2] <= nKey[2] + 1; ++key[2])
  {
    for (key[1] = nKey[1] - 1; !neighborFound && key[1] <= nKey[1] + 1; ++key[1])
    {
      for (key[0] = nKey[0] - 1; !neighborFound && key[0] <= nKey[0] + 1; ++key[0])
      {
        if (key != nKey)
        {
          OcTreeNode *node = target_octomap->search(key);
          if (node && target_octomap->isNodeOccupied(node))
          {
            // we have a neighbor => break!
            neighborFound = true;
          }
        }
      }
    }
  }

  return !neighborFound;
}

geometry_msgs::Point OctomapSegmentation::convert_eigen_to_geomsg(const Eigen::Vector3f input_vector)
{
  geometry_msgs::Point result;
  result.x = input_vector.x();
  result.y = input_vector.y();
  result.z = input_vector.z();
  return result;
}
