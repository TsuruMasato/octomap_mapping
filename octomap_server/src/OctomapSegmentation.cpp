#include <octomap_server/OctomapSegmentation.h>

OctomapSegmentation::OctomapSegmentation()
{
  pub_segmented_pc_ = nh_.advertise<sensor_msgs::PointCloud2>("segmented_pc", 10);
}

pcl::PointCloud<pcl::PointXYZRGB> OctomapSegmentation::segmentation(OctomapServer::OcTreeT *&target_octomap)
{
  ROS_ERROR("OctomapSegmentation::segmentation() start");
  // init pointcloud:
  pcl::PointCloud<OctomapServer::PCLPoint>::Ptr pcl_cloud(new pcl::PointCloud<OctomapServer::PCLPoint>);

  // call pre-traversal hook:
  ros::Time rostime = ros::Time::now();

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

  /* segmentation in PointCloud style*/


  /* update Octomap according to the PCL segmentation results */


  pcl::PointCloud<pcl::PointXYZRGB> simplified_pc;
  pcl::copyPointCloud(*pcl_cloud, simplified_pc);
  return simplified_pc;
};

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
