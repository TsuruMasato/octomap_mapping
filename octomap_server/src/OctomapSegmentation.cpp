#include <octomap_server/OctomapSegmentation.h>

OctomapSegmentation::OctomapSegmentation()
{
  // pub_segmented_pc_ = nh_.advertise<sensor_msgs::PointCloud2>("segmented_pc", 1);
  // pub_normal_vector_markers_ = nh_.advertise<visualization_msgs::MarkerArray>("normal_vectors", 1);
}

pcl::PointCloud<pcl::PointXYZRGB> OctomapSegmentation::segmentation(OctomapServer::OcTreeT *&target_octomap, visualization_msgs::MarkerArray &marker_array)
{
  // ROS_ERROR("OctomapSegmentation::segmentation() start");
  // init pointcloud:
  pcl::PointCloud<OctomapServer::PCLPoint>::Ptr pcl_cloud(new pcl::PointCloud<OctomapServer::PCLPoint>);
  set_private_octree(target_octomap);

  // call pre-traversal hook:
  ros::Time rostime = ros::Time::now();

  /* *********************************************************** */
  /* convert Octomap to PCL point cloud for segmentation process */
  /* *********************************************************** */
  bool is_success = convert_octomap_to_pcl_cloud(target_octomap, pcl_cloud);
  if(!is_success)
  {
    ROS_ERROR("[OctomapSegmentation] failed to convert Octomap to PCL PointCloud");
    pcl::PointCloud<pcl::PointXYZRGB> empty;
    return empty;
  }

  /* for sure. it works on May.27th. The number of voxels and PCL point cloud are exacly same. */
  // pcl::PointCloud<pcl::PointXYZRGB> debug_pointcloud;
  // pcl::copyPointCloud(*pcl_cloud, debug_pointcloud);
  // return debug_pointcloud; // for debug

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
    floor_marker.header.frame_id = frame_id_;
    // floor_marker.lifetime = ros::Duration(1.0);
    floor_marker.ns = "floor_plane";
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
    floor_marker.color.r = 0.5;
    floor_marker.color.g = 0.9;
    floor_marker.color.b = 0.9;
    floor_marker.color.a = 0.5;
    marker_array.markers.push_back(floor_marker);

    // text "floor"
    visualization_msgs::Marker floor_txt;
    floor_txt.header.frame_id = frame_id_;
    floor_txt.ns = "floor_plane";
    floor_txt.id = 1;
    floor_txt.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    floor_txt.pose.position.x = 0;
    floor_txt.pose.position.y = 0;
    floor_txt.pose.position.z = -floor_coefficients.values.at(3) + 0.2f;
    floor_txt.scale.z = 0.15;
    floor_txt.color.b = 0.9;
    floor_txt.color.g = 0.9;
    floor_txt.color.r = 0.9;
    floor_txt.color.a = 1.0;
    char floor_equation[30];
    sprintf(floor_equation, "%.2fx+%.2fy+%.2fz-%.2f=0", floor_coefficients.values.at(0), floor_coefficients.values.at(1), floor_coefficients.values.at(2), floor_coefficients.values.at(3));
    floor_txt.text = "Floor\n" + std::string(floor_equation);
    marker_array.markers.push_back(floor_txt);
  }

  // clustering
  std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> clusters;
  bool clustering_success = clustering(obstacle_cloud, clusters); // obstacle cloud becomes smaller...

  // primitive clustering
  // marker_array.markers.clear();
  // TODO: multi-thread process.
  std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> cubic_clusters, plane_clusters, sylinder_clusters, the_others;
  PCA_classify(clusters, marker_array, cubic_clusters, plane_clusters, sylinder_clusters, the_others);


  /* update Octomap according to the PCL segmentation results */


  /* merge the all cluster types with changing colors*/
  pcl::copyPointCloud(*obstacle_cloud, *result_cloud);
  // add_color_and_accumulate(floor_cloud, result_cloud, /*rgb*/ 191, 255, 128);
  // add_color_and_accumulate(cubic_clusters, result_cloud, /*rgb*/ 255, 128, 191);
  // add_color_and_accumulate(plane_clusters, result_cloud, /*rgb*/ 255, 191, 128);
  // add_color_and_accumulate(sylinder_clusters, result_cloud, /*rgb*/ 150, 245, 252);
  // add_color_and_accumulate(the_others, result_cloud, /*rgb*/ 100, 100, 100);
  

  pcl::PointCloud<pcl::PointXYZRGB> simplified_pc;
  pcl::copyPointCloud(*result_cloud, simplified_pc);
  return simplified_pc;
};

bool OctomapSegmentation::convert_octomap_to_pcl_cloud(OctomapServer::OcTreeT* &input_octomap, const pcl::PointCloud<OctomapServer::PCLPoint>::Ptr &output_cloud)
{
  // now, traverse all leafs in the tree:
  for (OctomapServer::OcTreeT::iterator it = input_octomap->begin(input_octomap->getTreeDepth()), end = input_octomap->end(); it != end; ++it)
  {
    // bool inUpdateBBX = OctomapServer::isInUpdateBBX(it);
    if (input_octomap->isNodeOccupied(*it))
    {
      // Ignore speckles in the map:
      bool isSpeckleFilterEnable = false;
      if (isSpeckleFilterEnable && (it.getDepth() >= input_octomap->getTreeDepth()) && isSpeckleNode(it.getKey(), input_octomap))
      {
        input_octomap->deleteNode(it.getKey(), input_octomap->getTreeDepth());
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

      output_cloud->push_back(point);
    }
  }
  if(output_cloud->size() != 0)
    return true;
  else
    return false;
}

pcl::ModelCoefficients OctomapSegmentation::ransac_horizontal_plane(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &input_cloud, const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &floor_cloud, const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &obstacle_cloud, double plane_thickness, const Eigen::Vector3f &axis)
{
  // ROS_INFO("ransac_horizontal_plane");
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZRGBNormal> seg;

  float floor_height_threshold = -camera_initial_height_ + 0.5;

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

  if (-coefficients->values.at(3) < floor_height_threshold) // when the height of the plane is enough low:
  {
    // ROS_WARN("floor height : %f", -coefficients->values.at(3));
    pcl::ExtractIndices<pcl::PointXYZRGBNormal> extract;
    extract.setInputCloud(input_cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*floor_cloud);
    extract.setNegative(true);
    extract.filter(*obstacle_cloud);

    for (size_t i = 0; i < floor_cloud->size(); i++)
    {
      private_octomap_->averageNodePrimitive(floor_cloud->at(i).x, floor_cloud->at(i).y, floor_cloud->at(i).z, ExOcTreeNode::ShapePrimitive::FLOOR);
    }

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

    for (size_t i = 0; i < ceiling_cloud->size(); i++)
    {
      private_octomap_->averageNodePrimitive(ceiling_cloud->at(i).x, ceiling_cloud->at(i).y, ceiling_cloud->at(i).z, ExOcTreeNode::ShapePrimitive::CEILING);
    }

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
    if (-coefficients->values.at(3) < floor_height_threshold) // the floor hight must be low.
    {
      // extract.setInputCloud(input_cloud);
      extract.setIndices(inliers);
      extract.setNegative(false);
      extract.filter(*floor_cloud);
      extract.setNegative(true);
      extract.filter(*obstacle_cloud);

      for (size_t i = 0; i < floor_cloud->size(); i++)
      {
        private_octomap_->averageNodePrimitive(floor_cloud->at(i).x, floor_cloud->at(i).y, floor_cloud->at(i).z, ExOcTreeNode::ShapePrimitive::FLOOR);
      }

      /* give back ceiling points */
      *obstacle_cloud += *ceiling_cloud;
      return *coefficients;
    }
    else
    {
      ROS_WARN("[OctomapSegmentation::ransac_horizontal_plane] found undesired horizontal plane (maybe tables)");
      pcl::copyPointCloud(*input_cloud, *obstacle_cloud);
      /* give back ceiling points */
      *obstacle_cloud += *ceiling_cloud;
      coefficients->header.frame_id = "FAIL";
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
  int min_cluster_size = 15;  // it was 30 for long time, and changed to 20. And now for Choreonoid simulation, became smaller
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

  /* the others (noise or worthless part) */
  pcl::PointIndices::Ptr the_rest_indices (new pcl::PointIndices);
  for (size_t i = 0; i < cluster_indices.size(); i++)
  {
    the_rest_indices->indices.insert(the_rest_indices->indices.end(), cluster_indices[i].indices.begin(), cluster_indices[i].indices.end());
  }
  ei.setNegative(true);
  
  /*extract*/
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr tmp_clustered_points(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  ei.setIndices(the_rest_indices);
  ei.filter(*tmp_clustered_points);
  /* add the rest noise cluster at the end of clusters vector. */
  output_results.push_back(tmp_clustered_points);
  

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

  const double threshold_angle = 10.0; //閾値[deg]
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

bool OctomapSegmentation::PCA_classify(std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &input_clusters, visualization_msgs::MarkerArray &marker_array, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &cubic_clusters, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &plane_clusters, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &sylinder_clusters, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &the_others)
{
  int marker_id = 2; // id:0 and 1 is used for floor
  for (size_t i = 0; i < input_clusters.size() - 1; i++)
  {
    auto target_cloud = input_clusters.at(i);
    pcl::PCA<PCLPoint> pca;
    pca.setInputCloud(target_cloud);
    float norm_x = pca.getEigenValues().x();  // the most effective eigen value
    float norm_y = pca.getEigenValues().y();  // 2nd eigen value
    float norm_z = pca.getEigenValues().z();  // 3rd eigen value
    Eigen::Vector3f first_eigen_vector = pca.getEigenVectors().row(0);
    Eigen::Vector3f second_eigen_vector = pca.getEigenVectors().row(1);
    Eigen::Vector3f third_eigen_vector = pca.getEigenVectors().row(2);
    ROS_ERROR("norm_x : %.2f, norm_y : %.2f, norm_z : %.2f", norm_x, norm_y, norm_z);
    ROS_ERROR("1st EigenVector : (%.2f, %.2f, %.2f)", first_eigen_vector.x(), first_eigen_vector.y(), first_eigen_vector.z());
    ROS_ERROR("2nd EigenVector : (%.2f, %.2f, %.2f)", second_eigen_vector.x(), second_eigen_vector.y(), second_eigen_vector.z());
    ROS_ERROR("3rd EigenVector : (%.2f, %.2f, %.2f)", third_eigen_vector.x(), third_eigen_vector.y(), third_eigen_vector.z());

    float first_principal_component, second_principal_component, third_principal_component;
    std::vector<float> norms;
    norms.push_back(norm_x);
    norms.push_back(norm_y);
    norms.push_back(norm_z);
    std::sort(norms.begin(), norms.end(), std::greater<float>());
    first_principal_component = norms.at(0);
    second_principal_component = norms.at(1);
    third_principal_component = norms.at(2);

    // bool is_meaningful_x, is_meaningful_y, is_meaningful_z;
    pcl::PointCloud<PCLPoint>::Ptr cloud_hull(new pcl::PointCloud<PCLPoint>);
    std::vector<pcl::Vertices> cloud_vertices;
    pcl::ConvexHull<PCLPoint> chull;
    chull.setInputCloud(target_cloud);
    chull.setDimension(3);
    chull.reconstruct(*cloud_hull, cloud_vertices);

    if(second_principal_component > 2.0 && second_principal_component > third_principal_component * 10.0)
    {
      // this cluster has a certain x-y area, and it's very thin : Plane
      if(abs(third_eigen_vector.z()) < 0.5f)
      {
        ROS_INFO("wall");
        for (size_t i = 0; i < target_cloud->size(); i++)
        {
          private_octomap_->averageNodePrimitive(target_cloud->at(i).x, target_cloud->at(i).y, target_cloud->at(i).z, ExOcTreeNode::ShapePrimitive::WALL);
        }
        add_wall_marker(pca, marker_id, marker_array, frame_id_);
        std::vector<uint8_t> color{0, 255, 0};
        add_line_marker(cloud_hull, cloud_vertices, color, marker_id, marker_array, frame_id_);
      }
      else
      {
        ROS_INFO("big floor or stair");
        // TODO: should I detect the height, to destingish ceiling?
        if (target_cloud->begin()->z > 0.5f && target_cloud->end()->z > 0.5f)
        { // if it's at high position, ceiling
          for (size_t i = 0; i < target_cloud->size(); i++)
          {
            private_octomap_->averageNodePrimitive(target_cloud->at(i).x, target_cloud->at(i).y, target_cloud->at(i).z, ExOcTreeNode::ShapePrimitive::CEILING);
          }
        }
        else
        { // if not so high, a steppable horizontal plane.
          for (size_t i = 0; i < target_cloud->size(); i++)
          {
            private_octomap_->averageNodePrimitive(target_cloud->at(i).x, target_cloud->at(i).y, target_cloud->at(i).z, ExOcTreeNode::ShapePrimitive::STEP);
          }
        }
        add_floor_marker(pca, marker_id, marker_array, frame_id_);
        std::vector<uint8_t> color{255, 20, 20};
        add_line_marker(cloud_hull, cloud_vertices, color, marker_id, marker_array, frame_id_);
      }
    }
    else if(first_principal_component > second_principal_component * 5.0)
    {
      // when the 1st principal is very big, it must be small step, cylinder, or stick-shape.
      if(abs(first_eigen_vector.z()) < 0.1 && abs(third_eigen_vector.z()) > 0.8)
      {
        ROS_WARN("small step. you can land here");
        for (size_t i = 0; i < target_cloud->size(); i++)
        {
          private_octomap_->averageNodePrimitive(target_cloud->at(i).x, target_cloud->at(i).y, target_cloud->at(i).z, ExOcTreeNode::ShapePrimitive::STEP);
        }
        add_step_marker(pca, marker_id, marker_array, frame_id_);
        std::vector<uint8_t> color{255, 220, 200};
        add_line_marker(cloud_hull, cloud_vertices, color, marker_id, marker_array, frame_id_);
      }
      else if(abs(first_eigen_vector.z()) > 0.3)
      {
        ROS_WARN("hand-rail or some vertical stick");        
        for (size_t i = 0; i < target_cloud->size(); i++)
        {
          private_octomap_->averageNodePrimitive(target_cloud->at(i).x, target_cloud->at(i).y, target_cloud->at(i).z, ExOcTreeNode::ShapePrimitive::HANDRAIL);
        }
        add_handrail_marker(pca, marker_id, marker_array, frame_id_);
        /* // not good accuracy. 
        pcl::ModelCoefficients ransac_result_coeff;
        if(ransac_cylinder_alignment(target_cloud, ransac_result_coeff))
          add_handrail_marker(ransac_result_coeff, marker_id, marker_array, frame_id_);
        */
        std::vector<uint8_t> color{180, 50, 255};
        add_line_marker(cloud_hull, cloud_vertices, color, marker_id, marker_array, frame_id_);
      }
      else
      {
        ROS_WARN("ladder step (poll shape. Be careful if you want to land here)");
        for (size_t i = 0; i < target_cloud->size(); i++)
        {
          private_octomap_->averageNodePrimitive(target_cloud->at(i).x, target_cloud->at(i).y, target_cloud->at(i).z, ExOcTreeNode::ShapePrimitive::HANDRAIL);
        }
      }
      sylinder_clusters.push_back(target_cloud);
    }
  }

  auto the_rest_noise_cluster = input_clusters.at(input_clusters.size()-1);
  for (size_t i = 0; i < the_rest_noise_cluster->size(); i++)
  {
    private_octomap_->averageNodePrimitive(the_rest_noise_cluster->at(i).x, the_rest_noise_cluster->at(i).y, the_rest_noise_cluster->at(i).z, ExOcTreeNode::ShapePrimitive::OTHER);
  }

  return true;
}

void OctomapSegmentation::add_wall_marker(pcl::PCA<PCLPoint> &pca,
                                          int &marker_id,
                                          visualization_msgs::MarkerArray &marker_array,
                                          const std::string &frame_id)
{
  visualization_msgs::Marker marker;
  marker.lifetime = ros::Duration(3.0);
  marker.header.frame_id = frame_id;
  marker.ns = "normal_vectors";
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
  Eigen::Vector3f axis_1st(eigen_vec.coeff(0, 0), eigen_vec.coeff(1, 0), eigen_vec.coeff(2, 0));
  Eigen::Vector3f axis_2nd(eigen_vec.coeff(0, 1), eigen_vec.coeff(1, 1), eigen_vec.coeff(2, 1));
  Eigen::Vector3f axis_3rd(eigen_vec.coeff(0, 2), eigen_vec.coeff(1, 2), eigen_vec.coeff(2, 2));
  // qx.setFromTwoVectors(Eigen::Vector3f(1, 0, 0), axis_1st);
  // qy.setFromTwoVectors(Eigen::Vector3f(1, 0, 0), axis_2nd);
  qz.setFromTwoVectors(Eigen::Vector3f(1, 0, 0), axis_3rd);

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
  marker_array.markers.push_back(marker);

  // text
  marker.id = marker_id++;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.pose.position = convert_eigen_to_geomsg(center_position + axis_3rd * 0.3f + axis_1st * 0.2f);
  marker.scale.z = 0.13;
  marker.color.b = 0.9;
  marker.color.g = 0.9;
  marker.color.r = 0.9;
  marker.color.a = 1.0;
  char plane_equation[30];
  sprintf(plane_equation, "(%.2f,%.2f,%.2f)", axis_3rd.x(), axis_3rd.y(), axis_3rd.z());
  marker.text = "Wall\n" + std::string(plane_equation);
  
  marker_array.markers.push_back(marker);
  
  visualization_msgs::Marker wall_plane_marker;
  wall_plane_marker.header.frame_id = frame_id_;
  wall_plane_marker.ns = "wall_plane";
  wall_plane_marker.type = visualization_msgs::Marker::CUBE;
  wall_plane_marker.id = marker_id++;
  wall_plane_marker.pose.position.x = center_position.x();
  wall_plane_marker.pose.position.y = center_position.y();
  wall_plane_marker.pose.position.z = center_position.z();
  wall_plane_marker.pose.orientation.x = qz.x();
  wall_plane_marker.pose.orientation.y = qz.y();
  wall_plane_marker.pose.orientation.z = qz.z();
  wall_plane_marker.pose.orientation.w = qz.w();
  wall_plane_marker.scale.x = 0.001f;
  wall_plane_marker.scale.y = eigen_values.y() * 0.01; // plane width
  wall_plane_marker.scale.z = eigen_values.x() * 0.01; // plane height, the largest component in 3 eigen values
  wall_plane_marker.color.r = 1.0;
  wall_plane_marker.color.g = 0.7;
  wall_plane_marker.color.b = 0.7;
  wall_plane_marker.color.a = 1.0;
  marker_array.markers.push_back(wall_plane_marker);
  
}

void OctomapSegmentation::add_floor_marker(pcl::PCA<PCLPoint> &pca,
                                          int &marker_id,
                                          visualization_msgs::MarkerArray &marker_array,
                                          const std::string &frame_id)
{
  visualization_msgs::Marker marker;
  marker.lifetime = ros::Duration(3.0);
  marker.header.frame_id = frame_id;
  marker.ns = "normal_vectors";
  marker.type = visualization_msgs::Marker::ARROW;
  marker.scale.x = 0.4;  // length:40cm
  marker.scale.y = 0.04; // width of the allow
  marker.scale.z = 0.04; // width of the allow

  Eigen::Vector3f center_position;
  center_position << pca.getMean().coeff(0), pca.getMean().coeff(1), pca.getMean().coeff(2);
  marker.pose.position.x = center_position.x();
  marker.pose.position.y = center_position.y();
  marker.pose.position.z = center_position.z();

  Eigen::Quaternionf qx, qy, qz;
  Eigen::Matrix3f eigen_vec = pca.getEigenVectors();
  Eigen::Vector3f eigen_values = pca.getEigenValues();
  Eigen::Vector3f axis_1st(eigen_vec.coeff(0, 0), eigen_vec.coeff(1, 0), eigen_vec.coeff(2, 0));
  Eigen::Vector3f axis_2nd(eigen_vec.coeff(0, 1), eigen_vec.coeff(1, 1), eigen_vec.coeff(2, 1));
  Eigen::Vector3f axis_3rd(eigen_vec.coeff(0, 2), eigen_vec.coeff(1, 2), eigen_vec.coeff(2, 2));
  // qx.setFromTwoVectors(Eigen::Vector3f(1, 0, 0), axis_1st);
  // qy.setFromTwoVectors(Eigen::Vector3f(1, 0, 0), axis_2nd);
  if (axis_3rd.z() < 0) // if the z axis was detected in negative direction :
  {
    axis_3rd = -axis_3rd; // flip
  }
  qz.setFromTwoVectors(Eigen::Vector3f(1, 0, 0), axis_3rd);

  /* visualize arrow only with the 3rd Eigen Vector, because it is the plane normal vector.*/
  marker.id = marker_id++;
  marker.pose.orientation.x = qz.x();
  marker.pose.orientation.y = qz.y();
  marker.pose.orientation.z = qz.z();
  marker.pose.orientation.w = qz.w();
  marker.color.r = 0.4;
  marker.color.g = 0.8;
  marker.color.b = 1.0;
  marker.color.a = 1.0;
  marker_array.markers.push_back(marker);

  // text
  marker.id = marker_id++;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.pose.position = convert_eigen_to_geomsg(center_position + axis_3rd * 0.3f + axis_1st * 0.2f);
  marker.scale.z = 0.13;
  marker.color.b = 0.9;
  marker.color.g = 0.9;
  marker.color.r = 0.9;
  marker.color.a = 1.0;
  char plane_equation[30];
  sprintf(plane_equation, "(%.2f,%.2f,%.2f)", axis_3rd.x(), axis_3rd.y(), axis_3rd.z());
  marker.text = "Horizontal\n" + std::string(plane_equation);

  marker_array.markers.push_back(marker);

  visualization_msgs::Marker wall_plane_marker;
  wall_plane_marker.header.frame_id = frame_id_;
  wall_plane_marker.ns = "wall_plane";
  wall_plane_marker.type = visualization_msgs::Marker::CUBE;
  wall_plane_marker.id = marker_id++;
  wall_plane_marker.pose.position.x = center_position.x();
  wall_plane_marker.pose.position.y = center_position.y();
  wall_plane_marker.pose.position.z = center_position.z();
  wall_plane_marker.pose.orientation.x = qz.x();
  wall_plane_marker.pose.orientation.y = qz.y();
  wall_plane_marker.pose.orientation.z = qz.z();
  wall_plane_marker.pose.orientation.w = qz.w();
  wall_plane_marker.scale.x = 0.001f;
  wall_plane_marker.scale.y = eigen_values.y() * 0.01; // plane width
  wall_plane_marker.scale.z = eigen_values.x() * 0.01; // plane height, the largest component in 3 eigen values
  wall_plane_marker.color.r = 1.0;
  wall_plane_marker.color.g = 0.7;
  wall_plane_marker.color.b = 0.7;
  wall_plane_marker.color.a = 1.0;
  marker_array.markers.push_back(wall_plane_marker);
}

void OctomapSegmentation::add_step_marker(pcl::PCA<PCLPoint> &pca,
                                           int &marker_id,
                                           visualization_msgs::MarkerArray &marker_array,
                                           const std::string &frame_id)
{
  visualization_msgs::Marker marker;
  marker.lifetime = ros::Duration(3.0);
  marker.header.frame_id = frame_id;
  marker.ns = "normal_vectors";
  marker.type = visualization_msgs::Marker::ARROW;
  marker.scale.x = 0.4;  // length:40cm
  marker.scale.y = 0.04; // width of the allow
  marker.scale.z = 0.04; // width of the allow

  Eigen::Vector3f center_position;
  center_position << pca.getMean().coeff(0), pca.getMean().coeff(1), pca.getMean().coeff(2);
  marker.pose.position.x = center_position.x();
  marker.pose.position.y = center_position.y();
  marker.pose.position.z = center_position.z();

  Eigen::Quaternionf qx, qy, qz;
  Eigen::Matrix3f eigen_vec = pca.getEigenVectors();
  Eigen::Vector3f eigen_values = pca.getEigenValues();
  Eigen::Vector3f axis_1st(eigen_vec.coeff(0, 0), eigen_vec.coeff(1, 0), eigen_vec.coeff(2, 0));
  Eigen::Vector3f axis_2nd(eigen_vec.coeff(0, 1), eigen_vec.coeff(1, 1), eigen_vec.coeff(2, 1));
  Eigen::Vector3f axis_3rd(eigen_vec.coeff(0, 2), eigen_vec.coeff(1, 2), eigen_vec.coeff(2, 2));
  if(axis_3rd.z() < 0)  // if the z axis was detected in negative direction :
  {
    axis_3rd = -axis_3rd; // flip
  }
  // qx.setFromTwoVectors(Eigen::Vector3f(1, 0, 0), axis_1st);
  // qy.setFromTwoVectors(Eigen::Vector3f(1, 0, 0), axis_2nd);
  qz.setFromTwoVectors(Eigen::Vector3f(1, 0, 0), axis_3rd);

  /* visualize arrow only with the 3rd Eigen Vector, because it is the plane normal vector.*/
  marker.id = marker_id++;
  marker.pose.orientation.x = qz.x();
  marker.pose.orientation.y = qz.y();
  marker.pose.orientation.z = qz.z();
  marker.pose.orientation.w = qz.w();
  marker.color.r = 1.0;
  marker.color.g = 0.95;
  marker.color.b = 0.0;
  marker.color.a = 1.0;
  marker_array.markers.push_back(marker);

  // text
  marker.id = marker_id++;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.pose.position = convert_eigen_to_geomsg(center_position + axis_3rd * 0.3f + axis_1st * 0.2f);
  marker.scale.z = 0.13;
  marker.color.b = 0.9;
  marker.color.g = 0.9;
  marker.color.r = 0.9;
  marker.color.a = 1.0;
  char plane_equation[30];
  sprintf(plane_equation, "(%.2f,%.2f,%.2f)", axis_3rd.x(), axis_3rd.y(), axis_3rd.z());
  marker.text = "small step\n" + std::string(plane_equation);

  marker_array.markers.push_back(marker);

  visualization_msgs::Marker wall_plane_marker;
  wall_plane_marker.header.frame_id = frame_id_;
  wall_plane_marker.ns = "wall_plane";
  wall_plane_marker.type = visualization_msgs::Marker::CUBE;
  wall_plane_marker.id = marker_id++;
  wall_plane_marker.pose.position.x = center_position.x();
  wall_plane_marker.pose.position.y = center_position.y();
  wall_plane_marker.pose.position.z = center_position.z();
  wall_plane_marker.pose.orientation.x = qz.x();
  wall_plane_marker.pose.orientation.y = qz.y();
  wall_plane_marker.pose.orientation.z = qz.z();
  wall_plane_marker.pose.orientation.w = qz.w();
  wall_plane_marker.scale.x = 0.001f;
  wall_plane_marker.scale.y = eigen_values.y() * 0.01; // plane width
  wall_plane_marker.scale.z = eigen_values.x() * 0.01; // plane height, the largest component in 3 eigen values
  wall_plane_marker.color.r = 1.0;
  wall_plane_marker.color.g = 0.7;
  wall_plane_marker.color.b = 0.7;
  wall_plane_marker.color.a = 1.0;
  marker_array.markers.push_back(wall_plane_marker);
}

void OctomapSegmentation::add_handrail_marker(pcl::PCA<PCLPoint> &pca,
                                          int &marker_id,
                                          visualization_msgs::MarkerArray &marker_array,
                                          const std::string &frame_id)
{
  visualization_msgs::Marker marker;
  marker.lifetime = ros::Duration(3.0);
  marker.header.frame_id = frame_id;
  marker.ns = "normal_vectors";
  marker.type = visualization_msgs::Marker::ARROW;
  marker.scale.x = 0.4;  // length:40cm
  marker.scale.y = 0.04; // width of the allow
  marker.scale.z = 0.04; // width of the allow

  Eigen::Vector3f center_position;
  center_position << pca.getMean().coeff(0), pca.getMean().coeff(1), pca.getMean().coeff(2);
  marker.pose.position.x = center_position.x();
  marker.pose.position.y = center_position.y();
  marker.pose.position.z = center_position.z();

  Eigen::Quaternionf qx, qy, qz;
  Eigen::Matrix3f eigen_vec = pca.getEigenVectors();
  Eigen::Vector3f eigen_values = pca.getEigenValues();
  Eigen::Vector3f axis_1st(eigen_vec.coeff(0, 0), eigen_vec.coeff(1, 0), eigen_vec.coeff(2, 0));
  Eigen::Vector3f axis_2nd(eigen_vec.coeff(0, 1), eigen_vec.coeff(1, 1), eigen_vec.coeff(2, 1));
  Eigen::Vector3f axis_3rd(eigen_vec.coeff(0, 2), eigen_vec.coeff(1, 2), eigen_vec.coeff(2, 2));
  if (axis_3rd.z() < 0) // if the z axis was detected in negative direction :
  {
    axis_3rd = -axis_3rd; // flip
  }
  // qx.setFromTwoVectors(Eigen::Vector3f(1, 0, 0), axis_1st);
  // qy.setFromTwoVectors(Eigen::Vector3f(1, 0, 0), axis_2nd);
  qz.setFromTwoVectors(Eigen::Vector3f(1, 0, 0), axis_3rd);

  /* visualize arrow only with the 3rd Eigen Vector, because it is the plane normal vector.*/
  marker.id = marker_id++;
  marker.pose.orientation.x = qz.x();
  marker.pose.orientation.y = qz.y();
  marker.pose.orientation.z = qz.z();
  marker.pose.orientation.w = qz.w();
  marker.color.r = 1.0;
  marker.color.g = 0.0;
  marker.color.b = 1.0;
  marker.color.a = 1.0;
  marker_array.markers.push_back(marker);

  // text
  marker.id = marker_id++;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.pose.position = convert_eigen_to_geomsg(center_position + axis_3rd * 0.3f + axis_1st * 0.2f);
  marker.scale.z = 0.13;
  marker.color.b = 0.9;
  marker.color.g = 0.9;
  marker.color.r = 0.9;
  marker.color.a = 1.0;
  char plane_equation[30];
  sprintf(plane_equation, "(%.2f,%.2f,%.2f)", axis_3rd.x(), axis_3rd.y(), axis_3rd.z());
  marker.text = "handrail\n" + std::string(plane_equation);

  marker_array.markers.push_back(marker);

  visualization_msgs::Marker wall_plane_marker;
  wall_plane_marker.header.frame_id = frame_id_;
  wall_plane_marker.ns = "wall_plane";
  wall_plane_marker.type = visualization_msgs::Marker::CUBE;
  wall_plane_marker.id = marker_id++;
  wall_plane_marker.pose.position.x = center_position.x();
  wall_plane_marker.pose.position.y = center_position.y();
  wall_plane_marker.pose.position.z = center_position.z();
  wall_plane_marker.pose.orientation.x = qz.x();
  wall_plane_marker.pose.orientation.y = qz.y();
  wall_plane_marker.pose.orientation.z = qz.z();
  wall_plane_marker.pose.orientation.w = qz.w();
  wall_plane_marker.scale.x = 0.001f;
  wall_plane_marker.scale.y = eigen_values.y() * 0.01; // plane width
  wall_plane_marker.scale.z = eigen_values.x() * 0.01; // plane height, the largest component in 3 eigen values
  wall_plane_marker.color.r = 1.0;
  wall_plane_marker.color.g = 0.7;
  wall_plane_marker.color.b = 0.7;
  wall_plane_marker.color.a = 1.0;
  marker_array.markers.push_back(wall_plane_marker);
}

void OctomapSegmentation::add_handrail_marker(pcl::ModelCoefficients cylinder_coefficient,
                                              int &marker_id,
                                              visualization_msgs::MarkerArray &marker_array,
                                              const std::string &frame_id)
{
  visualization_msgs::Marker marker;
  marker.lifetime = ros::Duration(1.5);
  marker.header.frame_id = frame_id;
  marker.ns = "normal_vectors";
  marker.type = visualization_msgs::Marker::CYLINDER;
  marker.scale.x = cylinder_coefficient.values.at(6); // length:40cm
  marker.scale.y = cylinder_coefficient.values.at(6); // width of the allow
  marker.scale.z = 0.6; // width of the allow

  Eigen::Vector3f center_position;
  center_position << cylinder_coefficient.values.at(0), cylinder_coefficient.values.at(1), cylinder_coefficient.values.at(2);
  marker.pose.position.x = center_position.x();
  marker.pose.position.y = center_position.y();
  marker.pose.position.z = center_position.z();

  auto q_cylinder = tf::createQuaternionFromRPY(cylinder_coefficient.values.at(3), cylinder_coefficient.values.at(4), cylinder_coefficient.values.at(5));
  
  /* visualize arrow only with the 3rd Eigen Vector, because it is the plane normal vector.*/
  marker.id = marker_id++;
  marker.pose.orientation.x = q_cylinder.x();
  marker.pose.orientation.y = q_cylinder.y();
  marker.pose.orientation.z = q_cylinder.z();
  marker.pose.orientation.w = q_cylinder.w();
  marker.color.r = 1.0;
  marker.color.g = 0.3;
  marker.color.b = 0.8;
  marker.color.a = 1.0;
  marker_array.markers.push_back(marker);

  // text
  marker.id = marker_id++;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.pose.position = convert_eigen_to_geomsg(center_position);
  marker.scale.z = 0.13;
  marker.color.b = 0.9;
  marker.color.g = 0.9;
  marker.color.r = 0.9;
  marker.color.a = 1.0;
  char plane_equation[30];
  sprintf(plane_equation, "(%.2f,%.2f,%.2f)", center_position.x(), center_position.y(), center_position.z());
  marker.text = "handrail_cylinder\n" + std::string(plane_equation);

  marker_array.markers.push_back(marker);
}

void OctomapSegmentation::add_cylinder_marker(pcl::PCA<PCLPoint> &pca, int &marker_id, visualization_msgs::MarkerArray &marker_array, const std::string &frame_id)
{
  visualization_msgs::Marker marker;
  marker.lifetime = ros::Duration(3.0);
  marker.header.frame_id = frame_id;
  marker.ns = "normal_vectors";
  marker.type = visualization_msgs::Marker::CYLINDER;
  marker.scale.x = 0.4;  // length:40cm
  marker.scale.y = 0.04; // width of the allow
  marker.scale.z = 0.04; // width of the allow

  Eigen::Vector3f center_position;
  center_position << pca.getMean().coeff(0), pca.getMean().coeff(1), pca.getMean().coeff(2);
  marker.pose.position.x = center_position.x();
  marker.pose.position.y = center_position.y();
  marker.pose.position.z = center_position.z();

  Eigen::Quaternionf qx, qy, qz;
  Eigen::Matrix3f eigen_vec = pca.getEigenVectors();
  Eigen::Vector3f eigen_values = pca.getEigenValues();
  Eigen::Vector3f axis_1st(eigen_vec.coeff(0, 0), eigen_vec.coeff(1, 0), eigen_vec.coeff(2, 0));
  Eigen::Vector3f axis_2nd(eigen_vec.coeff(0, 1), eigen_vec.coeff(1, 1), eigen_vec.coeff(2, 1));
  Eigen::Vector3f axis_3rd(eigen_vec.coeff(0, 2), eigen_vec.coeff(1, 2), eigen_vec.coeff(2, 2));
  // qx.setFromTwoVectors(Eigen::Vector3f(1, 0, 0), axis_1st);
  // qy.setFromTwoVectors(Eigen::Vector3f(1, 0, 0), axis_2nd);
  qz.setFromTwoVectors(Eigen::Vector3f(1, 0, 0), axis_3rd);

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
  marker_array.markers.push_back(marker);

  // text
  marker.id = marker_id++;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.pose.position = convert_eigen_to_geomsg(center_position + axis_3rd * 0.3f + axis_1st * 0.2f);
  marker.scale.z = 0.13;
  marker.color.b = 0.9;
  marker.color.g = 0.9;
  marker.color.r = 0.9;
  marker.color.a = 1.0;
  char plane_equation[30];
  sprintf(plane_equation, "(%.2f,%.2f,%.2f)", axis_3rd.x(), axis_3rd.y(), axis_3rd.z());
  marker.text = "Cylinder\n" + std::string(plane_equation);

  marker_array.markers.push_back(marker);  
}

bool OctomapSegmentation::ransac_wall_detection(std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> &input_clusters)
{
  for (size_t i = 0; i < input_clusters.size(); i++)
  {

  }
  return true;
}

bool OctomapSegmentation::ransac_cylinder_alignment(pcl::PointCloud<OctomapServer::PCLPoint>::Ptr &input_cloud, pcl::ModelCoefficients &output_coefficients)
{
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentationFromNormals<pcl::PointXYZRGBNormal, pcl::Normal> seg;
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  normals->header = input_cloud->header;
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    // pcl::Normal temp(input_cloud->at(i).normal);
    pcl::Normal temp(input_cloud->at(i).normal_x, input_cloud->at(i).normal_y, input_cloud->at(i).normal_z);
    normals->points.push_back(temp);
  }

  // Optional
  seg.setOptimizeCoefficients(true);
  // Mandatory
  seg.setModelType(pcl::SACMODEL_CYLINDER);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setNormalDistanceWeight(0.1);
  seg.setMaxIterations(10000);
  seg.setDistanceThreshold(0.05);
  seg.setRadiusLimits(0, 0.2);
  // seg.setAxis(axis);
  // seg.setEpsAngle(0.05);
  seg.setInputCloud(input_cloud);
  seg.setInputNormals(normals);
  seg.segment(*inliers, *coefficients);
  // ROS_INFO("plane size : %d", inliers->indices.size());

  if (inliers->indices.size() < 10)
  {
    ROS_ERROR("[OctomapSegmentation::ransac_cylinder] failed to align a cylinder model");
    output_coefficients.header.frame_id = "FAIL";
    return false;
  }

  // pcl::ExtractIndices<pcl::PointXYZRGBNormal> extract;
  // extract.setInputCloud(input_cloud);
  // extract.setIndices(inliers);
  // extract.setNegative(false);
  // extract.filter(*output_cloud);
  ROS_INFO("[OctomapSegmentation::ransac_cylinder] succeed in aligning a cylinder model");
  output_coefficients = *coefficients;
  std::cerr << "Cylinder coefficients: " << output_coefficients << std::endl;
  // output_coefficients.values.
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

void OctomapSegmentation::computeOBB(const pcl::PointCloud<PCLPoint>::Ptr &input_cloud, pcl::PCA<PCLPoint> &input_pca, Eigen::Vector3f &min_point, Eigen::Vector3f &max_point, Eigen::Vector3f &OBB_center, Eigen::Matrix3f &obb_rotational_matrix)
{
  min_point.x() = std::numeric_limits<float>::max();
  min_point.y() = std::numeric_limits<float>::max();
  min_point.z() = std::numeric_limits<float>::max();

  max_point.x() = std::numeric_limits<float>::min();
  max_point.y() = std::numeric_limits<float>::min();
  max_point.z() = std::numeric_limits<float>::min();

  Eigen::Vector3f center_position;
  center_position << input_pca.getMean().coeff(0), input_pca.getMean().coeff(1), input_pca.getMean().coeff(2);

  Eigen::Matrix3f eigen_vec = input_pca.getEigenVectors();
  Eigen::Vector3f axis_1st(eigen_vec.coeff(0, 0), eigen_vec.coeff(1, 0), eigen_vec.coeff(2, 0));
  Eigen::Vector3f axis_2nd(eigen_vec.coeff(0, 1), eigen_vec.coeff(1, 1), eigen_vec.coeff(2, 1));
  Eigen::Vector3f axis_3rd(eigen_vec.coeff(0, 2), eigen_vec.coeff(1, 2), eigen_vec.coeff(2, 2));

  unsigned int number_of_points = static_cast<unsigned int>(input_cloud->size());
  for (unsigned int i = 0; i < number_of_points; i++)
  {
    float x = (input_cloud->at(i).x - center_position.x()) * axis_1st.x() + (input_cloud->at(i).y - center_position.y()) * axis_1st.y() + (input_cloud->at(i).z - center_position.z()) * axis_1st.z();
    float y = (input_cloud->at(i).x - center_position.x()) * axis_2nd.x() + (input_cloud->at(i).y - center_position.y()) * axis_2nd.y() + (input_cloud->at(i).z - center_position.z()) * axis_2nd.z();
    float z = (input_cloud->at(i).x - center_position.x()) * axis_3rd.x() + (input_cloud->at(i).y - center_position.y()) * axis_3rd.y() + (input_cloud->at(i).z - center_position.z()) * axis_3rd.z();

    // float x = ((*input_)[(*indices_)[i]].x - mean_value_(0)) * major_axis_(0) +
    //           ((*input_)[(*indices_)[i]].y - mean_value_(1)) * major_axis_(1) +
    //           ((*input_)[(*indices_)[i]].z - mean_value_(2)) * major_axis_(2);
    // float y = ((*input_)[(*indices_)[i]].x - mean_value_(0)) * middle_axis_(0) +
    //           ((*input_)[(*indices_)[i]].y - mean_value_(1)) * middle_axis_(1) +
    //           ((*input_)[(*indices_)[i]].z - mean_value_(2)) * middle_axis_(2);
    // float z = ((*input_)[(*indices_)[i]].x - mean_value_(0)) * minor_axis_(0) +
    //           ((*input_)[(*indices_)[i]].y - mean_value_(1)) * minor_axis_(1) +
    //           ((*input_)[(*indices_)[i]].z - mean_value_(2)) * minor_axis_(2);

    if (x <= min_point.x())
      min_point.x() = x;
    if (y <= min_point.y())
      min_point.y() = y;
    if (z <= min_point.z())
      min_point.z() = z;

    if (x >= max_point.x())
      max_point.x() = x;
    if (y >= max_point.y())
      max_point.y() = y;
    if (z >= max_point.z())
      max_point.z() = z;
  }

  obb_rotational_matrix << axis_1st(0), axis_2nd(0), axis_3rd(0),
      axis_1st(1), axis_2nd(1), axis_3rd(1),
      axis_1st(2), axis_2nd(2), axis_3rd(2);

  Eigen::Vector3f shift(
      (max_point.x() + min_point.x()) / 2.0f,
      (max_point.y() + min_point.y()) / 2.0f,
      (max_point.z() + min_point.z()) / 2.0f);

  min_point.x() -= shift(0);
  min_point.y() -= shift(1);
  min_point.z() -= shift(2);

  max_point.x() -= shift(0);
  max_point.y() -= shift(1);
  max_point.z() -= shift(2);

  OBB_center = center_position + obb_rotational_matrix * shift;
}

void OctomapSegmentation::add_OBB_marker(const Eigen::Vector3f &min_obb, const Eigen::Vector3f &max_obb, const Eigen::Vector3f &center_obb, const Eigen::Matrix3f &rot_obb, int &marker_id, visualization_msgs::MarkerArray &marker_array, const std::string &frame_id)
{
  visualization_msgs::Marker wall_plane_marker;
  wall_plane_marker.header.frame_id = frame_id;
  wall_plane_marker.ns = "plane_bounding_box";
  wall_plane_marker.type = visualization_msgs::Marker::CUBE;
  wall_plane_marker.id = marker_id++;
  wall_plane_marker.pose.position.x = center_obb.x();
  wall_plane_marker.pose.position.y = center_obb.y();
  wall_plane_marker.pose.position.z = center_obb.z();
  Eigen::Quaternionf quat(rot_obb);
  quat.normalize();
  wall_plane_marker.pose.orientation.x = quat.x();
  wall_plane_marker.pose.orientation.y = quat.y();
  wall_plane_marker.pose.orientation.z = quat.z();
  wall_plane_marker.pose.orientation.w = quat.w();
  // according to the normal vector, there are 8 cases:
  Eigen::Vector3f normal_vec;
  normal_vec << rot_obb(0,2), rot_obb(1,2), rot_obb(2,2);
  /*
  if ((normal_vec.x() > 0.0 && normal_vec.y() > 0.0 && normal_vec.z() > 0.0) || (normal_vec.x() < 0.0 && normal_vec.y() < 0.0 && normal_vec.z() < 0.0))
  {
    float size_x = abs(max_obb.x() - min_obb.x());
    float size_y = abs(max_obb.y() - min_obb.y());
    wall_plane_marker.scale.x = size_x;
    wall_plane_marker.scale.y = size_y;
  }
  else if ((normal_vec.x() > 0.0 && normal_vec.y() > 0.0 && normal_vec.z() < 0.0) || (normal_vec.x() < 0.0 && normal_vec.y() < 0.0 && normal_vec.z() > 0.0))
  {
    float size_x = abs(max_obb.x() - min_obb.x());
    float size_y = abs(max_obb.y() - min_obb.y());
    wall_plane_marker.scale.x = size_x;
    wall_plane_marker.scale.y = size_y;
  }
  */
  float size_x = abs(max_obb.x() - min_obb.x());
  float size_y = abs(max_obb.y() - min_obb.y());
  wall_plane_marker.scale.x = size_x;
  wall_plane_marker.scale.y = 0.3f;
  wall_plane_marker.scale.z = 0.001f;
  wall_plane_marker.color.r = 1.0;
  wall_plane_marker.color.g = 0.7;
  wall_plane_marker.color.b = 0.7;
  wall_plane_marker.color.a = 1.0;
  marker_array.markers.push_back(wall_plane_marker);
}

void OctomapSegmentation::add_line_marker(const pcl::PointCloud<PCLPoint>::Ptr &input_vertices, const std::vector<pcl::Vertices> &input_surface, const std::vector<uint8_t> &rgb, int &marker_id, visualization_msgs::MarkerArray &marker_array, std::string frame_id)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = frame_id;
  marker.id = marker_id++;
  marker.ns = "plane_bound";
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.scale.x = 0.005;
  marker.color.r = rgb[0] / 255.0f;
  marker.color.g = rgb[1] / 255.0f;
  marker.color.b = rgb[2] / 255.0f;
  marker.color.a = 0.8f;
  marker.pose.orientation.w = 1.0;

  geometry_msgs::Point vertix_0, vertix_1, vertix_2;
  for (size_t i = 0; i < input_surface.size(); i++)
  {
    auto index_0 = input_surface.at(i).vertices.at(0);
    auto index_1 = input_surface.at(i).vertices.at(1);
    auto index_2 = input_surface.at(i).vertices.at(2);

    vertix_0.x = input_vertices->at(index_0).x;
    vertix_0.y = input_vertices->at(index_0).y;
    vertix_0.z = input_vertices->at(index_0).z;

    vertix_1.x = input_vertices->at(index_1).x;
    vertix_1.y = input_vertices->at(index_1).y;
    vertix_1.z = input_vertices->at(index_1).z;

    vertix_2.x = input_vertices->at(index_2).x;
    vertix_2.y = input_vertices->at(index_2).y;
    vertix_2.z = input_vertices->at(index_2).z;

    marker.points.push_back(vertix_0);
    marker.points.push_back(vertix_1);
    marker.points.push_back(vertix_2);
  }

  marker_array.markers.push_back(marker);

}