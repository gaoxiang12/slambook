/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"

namespace myslam
{

VisualOdometry::VisualOdometry() :
    state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 )
{
    num_of_features_    = Config::get<int> ( "number_of_features" );
    scale_factor_       = Config::get<double> ( "scale_factor" );
    level_pyramid_      = Config::get<int> ( "level_pyramid" );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<float> ( "max_num_lost" );
    min_inliers_        = Config::get<int> ( "min_inliers" );
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
    map_point_erase_ratio_ = Config::get<double> ("map_point_erase_ratio");
    orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
}

VisualOdometry::~VisualOdometry()
{

}

bool VisualOdometry::addFrame ( Frame::Ptr frame )
{
    switch ( state_ )
    {
    case INITIALIZING:
    {
        state_ = OK;
        curr_ = ref_ = frame;
        map_->insertKeyFrame ( frame );
        // extract features from first frame and add them into map
        extractKeyPoints();
        computeDescriptors();
        // add the keypoints into map
        addAllKeypointsIntoMap();
        // set the first frame as key-frame
        ref_->is_key_frame_ = true;
        break;
    }
    case OK:
    {
        frame->setPose ( pose_curr_estimated_ );
        curr_ = frame;
        extractKeyPoints();
        computeDescriptors();
        boost::timer timer;
        featureMatching();
        cout<<"feature matching costs time: "<<timer.elapsed()<<endl;
        poseEstimationRGBD();
        cout<<"chekcing estimated pose"<<endl;
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_->setPose ( pose_curr_estimated_ );
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
                addKeyFrame();
            }
        }
        else // bad estimation due to various reasons
        {
            pose_curr_estimated_ = curr_->T_c_w_;
            num_lost_++;
            if ( num_lost_ > max_num_lost_ )
            {
                state_ = LOST;
            }
            return false;
        }
        break;
    }
    case LOST:
    {
        cout<<"vo has lost."<<endl;
        break;
    }
    }

    return true;
}

void VisualOdometry::extractKeyPoints()
{
    orb_->detect ( curr_->color_, keypoints_curr_ );
    cout<<"detect total "<<keypoints_curr_.size() <<" features"<<endl;
    /*
    Mat img_show = curr_->color_.clone();
    cv::drawKeypoints ( curr_->color_, keypoints_curr_, img_show );
    cv::imshow ( "features", img_show );
    cv::waitKey ( 0 );
    */
}

void VisualOdometry::computeDescriptors()
{
    orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
}

void VisualOdometry::addAllKeypointsIntoMap()
{
    // curr_->keypoints_ = keypoints_curr_;
    for ( int i=0; i<keypoints_curr_.size(); i++ )
    {
        double d = curr_->findDepth ( keypoints_curr_[i] );
        if ( d<0 )  continue;
        // valid depth, create a map point and insert it into map
        MapPoint::Ptr map_point = MapPoint::createMapPoint();
        map_point->pos_ = curr_->camera_->pixel2world (
                              Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ),
                              curr_->T_c_w_, d
                          );
        map_point->norm_ = ( map_point->pos_ - curr_->getCamCenter() );
        map_point->norm_.normalize();
        map_point->descriptor_ = descriptors_curr_.row ( i ).clone();
        // map_point->observed_frames_.push_back ( curr_.get() );
        map_->insertMapPoint ( map_point );
    }
    map_->insertKeyFrame ( curr_ );
    cout<<"total points in map: "<<map_->map_points_.size() <<endl;
}

void VisualOdometry::featureMatching()
{
    cout<<"ref id = "<<ref_->id_<<endl;
    cout<<"current id = "<<curr_->id_<<endl;
    // select the candidate of matching
    vector<MapPoint::Ptr>    pts_candidate;
    Mat descriptor_map;
    for ( auto pt:map_->map_points_ )
    {
        if ( pts_candidate.size() > 2000 )
            break;
        
        MapPoint::Ptr point = pt.second;
        // check if the point is in current frame point->pos_
        if ( !curr_->isInFrame ( point->pos_ ) )
        {
            // cout<<"point outside"<<endl;
            continue;
        }
        Vector3d view_angle = point->pos_ - curr_->getCamCenter();
        view_angle.normalize();
        if ( ( view_angle.transpose() *point->norm_ ) < 0.6 ) // v^T*n>cos(60)
            continue;
        pts_candidate.push_back ( point );
        descriptor_map.push_back ( point->descriptor_ );
    }
    cout<<"map points candidate = "<<pts_candidate.size() <<endl;
    
    vector<cv::DMatch> matches;
    cv::BFMatcher matcher ( cv::NORM_HAMMING );
    matcher.match ( descriptor_map, descriptors_curr_, matches );
    cout<<"caught total "<<matches.size() <<" matches."<<endl;
    
    // select the best matches
    float min_dis = std::min_element (
                        matches.begin(), matches.end(),
                        [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
    {
        return m1.distance < m2.distance;
    } )->distance;

    match_map_.clear();
    match_curr_.clear();

    for ( cv::DMatch& m : matches )
    {
        if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
        {
            match_map_.push_back ( pts_candidate[m.queryIdx]->id_ );
            pts_candidate[m.queryIdx]->observed_times_ ++;
            match_curr_.push_back ( m.trainIdx );
        }
    }
    cout<<"good matches: "<<match_map_.size() <<endl;
    
}

void VisualOdometry::poseEstimationRGBD()
{
    // construct the 3d 3d observations
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3 ( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
                            curr_->T_c_w_.rotation_matrix(), curr_->T_c_w_.translation()
                        ) );
    optimizer.addVertex ( pose );

    // edges
    int index = 0;
    vector<EdgeProjectXYZ2UVPoseOnly*> edges_uv;
    vector<EdgeProjectXYZRGBDPoseOnly*> edges_xyz;
    vector<bool> inlier ( match_curr_.size(), true );

    for ( int i=0; i<match_curr_.size(); i++ )
    {
        cv::KeyPoint kp = keypoints_curr_[ match_curr_[i] ];
        double d = curr_->findDepth ( kp );
        /*
        if ( d<0 )
        {
            // 3D -> 2D projection
            EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
            edge->setId(index++);
            edge->setVertex(0, pose);
            edge->camera_ = curr_->camera_;
            edge->point_ = map_->map_points_[ match_map_[i] ]->pos_;
            edge->setMeasurement( Vector2d(kp.pt.x, kp.pt.y) );
            edge->setInformation( Eigen::Matrix2d::Identity() );
            edge->setRobustKernel( new g2o::RobustKernelHuber );
            edges_uv.push_back( edge );
            optimizer.addEdge( edge );
        }
        else
            */
        {
            // 3D-3D observation
            EdgeProjectXYZRGBDPoseOnly* edge = new EdgeProjectXYZRGBDPoseOnly();
            edge->setId ( index++ );
            edge->setVertex ( 0, pose );
            edge->point_ = map_->map_points_[ match_map_[i] ]->pos_;
            edge->setMeasurement (
                curr_->camera_->pixel2camera (
                    Vector2d ( kp.pt.x, kp.pt.y ), d
                )
            );
            edge->setInformation ( Eigen::Matrix3d::Identity() *1e5 );
            edge->setRobustKernel ( new g2o::RobustKernelHuber );
            edges_xyz.push_back ( edge );
            optimizer.addEdge ( edge );
        }
    }

    cout<<"total edges: 3d-2d"<<edges_uv.size() <<", 3d-3d: "<<edges_xyz.size() <<endl;
    // start optimization
    optimizer.setVerbose ( false );
    optimizer.initializeOptimization();
    optimizer.optimize ( 10 );

    int num_outliers = 0;
    // remove the outlier
    for ( EdgeProjectXYZRGBDPoseOnly* edge: edges_xyz )
    {
        edge->computeError();
        double chi2 = edge->chi2();
        // cout<<"chi2 = "<<chi2<<endl;
        if ( chi2 > 20 )
        {
            // regard this as an outlier
            edge->setLevel ( 1 );
            inlier[edge->id()] = false;
            num_outliers++;
        }
        else
        {
            edge->setRobustKernel ( nullptr );
        }
    }

    cout<<"inliers: "<<num_inliers_<<endl;
    pose->setEstimate ( g2o::SE3Quat (
                            curr_->T_c_w_.rotation_matrix(), curr_->T_c_w_.translation()
                        ) );

    optimizer.initializeOptimization();
    optimizer.optimize ( 10 );

    cout<<"pose before: "<<curr_->T_c_w_.matrix() <<endl;
    cout<<"pose after: "<<pose->estimate() <<endl;

    /*
    curr_->setPose(
        SE3( pose->estimate().rotation(), pose->estimate().translation() )
    );
    */
    pose_curr_estimated_ = SE3 ( pose->estimate().rotation(), pose->estimate().translation() );

    num_inliers_ = 0;
    for ( int i=0; i<match_map_.size(); i++ )
    {
        if ( inlier[i] == true )
        {
            map_->map_points_[ match_map_[i] ]->correct_times_++;
            num_inliers_++;
        }
    }
    
    Mat img_show = curr_->color_.clone();
    for ( unsigned long pt_map_index : match_map_ )
    {
        Vector3d p_world = map_->map_points_[pt_map_index]->pos_;
        Vector2d pixel = curr_->camera_->world2pixel( p_world, curr_->T_c_w_ );
        cv::circle(img_show, cv::Point2d(pixel(0,0),pixel(1,0)), 5, cv::Scalar(0,250,0),2 );
    }
    cv::imshow("matched features", img_show);
    cv::waitKey(1);
    cout<<"pose estimation returns"<<endl;
}

bool VisualOdometry::checkEstimatedPose()
{
    // check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
        return false;
    }
    // if the motion is too large, it is probably wrong
    SE3 delta = ref_->T_c_w_.inverse() * pose_curr_estimated_;
    Sophus::Vector6d d = delta.log();
    if ( d.norm() > 5.0 )
    {
        cout<<"reject because motion is too large: "<<d.norm()<<endl;
        return false;
    }
    return true;
}

bool VisualOdometry::checkKeyFrame()
{
    SE3 delta = ref_->T_c_w_.inverse() * pose_curr_estimated_;
    Sophus::Vector6d d = delta.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}

void VisualOdometry::addKeyFrame()
{
    cout<<"adding a key-frame"<<endl;
    // add the curr_ into key-frames
    map_->insertKeyFrame ( curr_ );

    // remove the far away or hardly seen map points 
    for ( auto iter = map_->map_points_.begin(); iter!=map_->map_points_.end(); )
    {
        MapPoint::Ptr p = iter->second;
        if ( p->correct_times_==0 || 
            float(p->correct_times_)/p->observed_times_< map_point_erase_ratio_ )
        {
            // erase this point 
            iter = map_->map_points_.erase(iter);
            continue;
        }
        Vector3d p_cam = curr_->camera_->world2camera(p->pos_, curr_->T_c_w_);
        if ( p_cam(2,0)<0 || p_cam(2,0)>10 )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        /*
        Vector2d pixel = curr_->camera_->camera2pixel(p_cam);
        if ( pixel(0,0)<-20 || pixel(1,0)<-20 || pixel(0,0)>curr_->color_.cols+20 || pixel(1,0)>curr_->color_.rows+20 )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        */
        iter++;
    }
    map_->map_points_.clear();
    
    for ( int i=0; i<keypoints_curr_.size(); i++ )
    {
        if ( std::find ( match_curr_.begin(), match_curr_.end(), i ) != match_curr_.end() ) // this key-point is matched
        {
            continue;
        }
        // add this key point into map
        
        double d = curr_->findDepth ( keypoints_curr_[i] );
        if ( d<0 )  continue;
        MapPoint::Ptr map_point = MapPoint::createMapPoint();
        map_point->pos_ = curr_->camera_->pixel2world (
                              Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ),
                              curr_->T_c_w_, d
                          );
        map_point->norm_ = ( map_point->pos_ - curr_->getCamCenter() );
        map_point->norm_.normalize();
        map_point->descriptor_ = descriptors_curr_.row ( i ).clone();
        // map_point->observed_frames_.push_back ( curr_.get() );
        map_->insertMapPoint ( map_point );
    }
    
    ref_ = curr_;
    
    cout<<"total points in map: "<<map_->map_points_.size() <<endl;
}

}
