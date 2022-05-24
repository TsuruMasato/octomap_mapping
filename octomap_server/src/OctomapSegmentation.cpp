#include <octomap_server/OctomapSegmentation.h>

OctomapSegmentation::OctomapSegmentation()
{
  pub_segmented_pc_ = nh_.advertise<sensor_msgs::PointCloud2>("segmented_pc", 1);
  pub_normal_vector_markers_ = nh_.advertise<visualization_msgs::MarkerArray>("normal_vectors", 1);
}

pcl::PointCloud<pcl::PointXYZRGB> OctomapSegmentation::segmentation(OctomapServer::OcTreeT *&target_octomap)
{
  ROS_ERROR("OctomapSegmentation::segmentation() start");
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

  /* ******************************* */
  /* segmentation in PointCloud style*/
  /* ******************************* */

  // Remove Floor plane
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr floor_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr obstacle_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  bool found_floor = plane_ransac(/*input*/ pcl_cloud, /*output 1*/ floor_cloud, /*output 2*/ obstacle_cloud);
  if(!found_floor)
  {
    ROS_ERROR("Failed Floor in Octomap by RANSAC!!");
  }
  
  // clustering
  ROS_ERROR("clustering");
  std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> clusters;
  bool clustering_success = clustering(obstacle_cloud, clusters);

  // RANSAC wall detection
  ransac_wall_detection(clusters);

  /* update Octomap according to the PCL segmentation results */

  // とりあえず除去した床も着色してマージ
  uint8_t floor_r = 191;
  uint8_t floor_g = 255;
  uint8_t floor_b = 128;
  for (auto itr = floor_cloud->begin(); itr != floor_cloud->end(); itr++)
  {
    itr->r = floor_r;
    itr->g = floor_g;
    itr->b = floor_b;
  }
  *obstacle_cloud += *floor_cloud;

  pcl::PointCloud<pcl::PointXYZRGB> simplified_pc;
  pcl::copyPointCloud(*obstacle_cloud, simplified_pc);
  return simplified_pc;
};

bool OctomapSegmentation::plane_ransac(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &input_cloud, const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &floor_cloud, const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &obstacle_cloud, double plane_thickness, const Eigen::Vector3f &axis)
{
  // ROS_INFO("plane_ransac");
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
    ROS_WARN("[OctomapSegmentation::plane_ransac] no horizontal plane in input_cloud");
    pcl::copyPointCloud(*input_cloud, *obstacle_cloud);
    return false;
  }

  if (-coefficients->values.at(3) < -0.5) // when the height of the plane is enough low:
  {
    ROS_WARN("floor height : %f", -coefficients->values.at(3));
    pcl::ExtractIndices<pcl::PointXYZRGBNormal> extract;
    extract.setInputCloud(input_cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*floor_cloud);
    extract.setNegative(true);
    extract.filter(*obstacle_cloud);
    return true;
  }

  else
  {
    ROS_WARN("ceiling height : %f", -coefficients->values.at(3));
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
      ROS_WARN("plane size is not enough large to remove.");
      return false;
    }
    ROS_WARN("[2nd] floor height : %f", -coefficients->values.at(3));
    if (-coefficients->values.at(3) < -0.5) // the floor hight must be low.
    {
      // extract.setInputCloud(input_cloud);
      extract.setIndices(inliers);
      extract.setNegative(false);
      extract.filter(*floor_cloud);
      extract.setNegative(true);
      extract.filter(*obstacle_cloud);
      return true;
    }
    else
    {
      ROS_WARN("[OctomapSegmentation::plane_ransac] found undesired horizontal plane (maybe tables)");
      pcl::copyPointCloud(*input_cloud, *obstacle_cloud);
      return true;
    }
  }
}

/*
bool OctomapSegmentation::remove_floor_RANSAC(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &input, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &floor_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &obstacle_cloud)
{
  floor_cloud->header = input->header;
  obstacle_cloud->header = input->header;

  if (input->size() < 20)
  {
    ROS_WARN("[OctSegmentation::remove_floor_RANSAC] Input pointcloud is too small. No plane");
    *obstacle_cloud = *input;
    return false;
  }
  else
  {
    // plane detection for ground plane removal:
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    // Create the segmentation object and set up:
    pcl::SACSegmentation<PCLPointCloud> seg;
    seg.setOptimizeCoefficients(true);
    // TODO: maybe a filtering based on the surface normals might be more robust / accurate?
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.04);
    seg.setAxis(Eigen::Vector3f(0, 0, 1));
    seg.setEpsAngle(0.15);

    const PCLPointCloud::Ptr cloud_filtered(input);
    // Create the filtering object
    pcl::ExtractIndices<PCLPoint> extract;
    bool groundPlaneFound = false;

    while (cloud_filtered->size() > 30 && !groundPlaneFound)
    {
      seg.setInputCloud(input);
      seg.segment(*inliers, *coefficients);
      if (inliers->indices.size() == 0)
      {
        ROS_INFO("PCL segmentation did not find any plane.");

        break;
      }

      extract.setInputCloud(cloud_filtered.makeShared());
      extract.setIndices(inliers);

      if (-coefficients->values.at(3) < -0.8)
      { // check d of the plane [m]
        ROS_WARN("Ground plane found: %zu/%zu inliers. Coeff: %f %f %f %f, m_groundFilterPlaneDistance: %f", inliers->indices.size(), cloud_filtered.size(),
                 coefficients->values.at(0), coefficients->values.at(1), coefficients->values.at(2), coefficients->values.at(3), m_groundFilterPlaneDistance);
        extract.setNegative(false);
        extract.filter(ground);

        // remove ground points from full pointcloud:
        // workaround for PCL bug:
        if (inliers->indices.size() != cloud_filtered.size() || true)
        {
          extract.setNegative(true);
          PCLPointCloud cloud_out;
          extract.filter(cloud_out);
          nonground += cloud_out;
          cloud_filtered = cloud_out;
        }

        groundPlaneFound = true;
      }
      else
      {
        ROS_WARN("Horizontal plane (not ground) found: %zu/%zu inliers. Coeff: %f %f %f %f, m_groundFilterPlaneDistance: %f", inliers->indices.size(), cloud_filtered.size(),
                 coefficients->values.at(0), coefficients->values.at(1), coefficients->values.at(2), coefficients->values.at(3), m_groundFilterPlaneDistance);
        pcl::PointCloud<PCLPoint> cloud_out;
        extract.setNegative(false);
        extract.filter(cloud_out);
        nonground += cloud_out;
        // debug
        //            pcl::PCDWriter writer;
        //            writer.write<PCLPoint>("nonground_plane.pcd",cloud_out, false);

        // remove current plane from scan for next iteration:
        // workaround for PCL bug:
        if (inliers->indices.size() != cloud_filtered.size())
        {
          extract.setNegative(true);
          cloud_out.points.clear();
          extract.filter(cloud_out);
          cloud_filtered = cloud_out;
        }
        else
        {
          cloud_filtered.points.clear();
        }
      }
    }
    // TODO: also do this if overall starting pointcloud too small?
    if (!groundPlaneFound)
    { // no plane found or remaining points too small
      ROS_WARN("No ground plane found in scan");

      // do a rough fitlering on height to prevent spurious obstacles
      /* [Tsuru] 結構ヤバイ処理。点群の上下一定範囲を強制的に床平面と見做して分割してる */
      /*
      pcl::PassThrough<PCLPoint> second_pass;
      second_pass.setFilterFieldName("z");
      second_pass.setFilterLimits(-m_groundFilterPlaneDistance, m_groundFilterPlaneDistance);
      second_pass.setInputCloud(pc.makeShared());
      second_pass.filter(ground);

      second_pass.setFilterLimitsNegative (true);
      second_pass.filter(nonground);
      
      nonground = pc; // [Tsuru] regard all points as obstacles.
    }

    // debug:
    //        pcl::PCDWriter writer;
    //        if (pc_ground.size() > 0)
    //          writer.write<PCLPoint>("ground.pcd",pc_ground, false);
    //        if (pc_nonground.size() > 0)
    //          writer.write<PCLPoint>("nonground.pcd",pc_nonground, false);
  }
  return true;
}
*/

bool OctomapSegmentation::clustering(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input_cloud, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &output_results)
{
  ROS_ERROR("start Clustering");
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
  ROS_WARN("start clustering function");
  cec.segment(cluster_indices);
  ROS_WARN("finish clustering function");
  /*メンバ数が最小値以下のクラスタと最大値以上のクラスタを取得できる*/
  // cec.getRemovedClusters (small_clusters, large_clusters);

  std::cout << "cluster_indices.size() = " << cluster_indices.size() << std::endl;

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
  ROS_ERROR("Clustering finish. It was devided into %d groups", output_results.size());

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

bool OctomapSegmentation::ransac_wall_detection(std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &input_clusters)
{
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
