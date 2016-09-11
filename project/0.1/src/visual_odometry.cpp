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

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include <Eigen/src/Eigenvalues/ComplexSchur.h>

namespace myslam
{

VisualOdometry::VisualOdometry() :
    state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr )
{
    num_of_features_    = Config::get<int>("number_of_features");
    scale_factor_       = Config::get<double>("scale_factor");
    level_pyramid_      = Config::get<int>("level_pyramid");
    orb_ = cv::ORB::create( num_of_features_, scale_factor_, level_pyramid_);
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
        map_->insertKeyFrame( frame );
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
        curr_ = frame; 
        extractKeyPoints();
        computeDescriptors();
        featureMatching();
        break;
    }
    case LOST:
    {
        break;
    }
    }
    
    return true;
}

void VisualOdometry::extractKeyPoints()
{
    orb_->detect( curr_->color_, keypoints_curr_ );
    cout<<"detect total "<<keypoints_curr_.size()<<" features"<<endl;
}

void VisualOdometry::computeDescriptors()
{
    orb_->compute( curr_->color_, keypoints_curr_, descriptors_curr_ );
}

void VisualOdometry::addAllKeypointsIntoMap()
{
    curr_->keypoints_ = keypoints_curr_;
    for ( int i=0; i<keypoints_curr_.size(); i++ )
    {
        double d = curr_->findDepth( keypoints_curr_[i] );
        if ( d<0 )  continue;
        // valid depth, create a map point and insert it into map
        MapPoint::Ptr map_point = MapPoint::createMapPoint();
        map_point->pos_ = curr_->camera_->pixel2world( 
            Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), d 
        );
        map_point->norm_ = (map_point->pos_ - curr_->getCamCenter());
        map_point->norm_.normalize();
        map_point->descriptor_ = descriptors_curr_.row(i).clone();
        map_point->observed_frames_.push_back( curr_.get() );
        map_->insertMapPoint(map_point);
    }
    
    map_->insertKeyFrame( curr_ );
}

void VisualOdometry::featureMatching()
{
    // select the candidate of matching 
    cv::BFMatcher matcher (cv::NORM_HAMMING);
    vector<MapPoint*>    pts_candidate;
    Mat descriptor_map; 
    for ( auto pt:map_->map_points_ )
    {
        MapPoint* point = pt.second;
        // check if the point is in current frame point->pos_
        if ( curr_->isInFrame(point->pos_) )
            continue;
        pts_candidate.push_back( point );
        descriptor_map.push_back( point->descriptor_ );
    }
    vector<cv::DMatch> matches; 
    matcher.match( descriptor_map, match_map_, matches );
    cout<<"caught total "<<matches.size()<<" matches."<<endl;
    // select the best matches 
}



}
