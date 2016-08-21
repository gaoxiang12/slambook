#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <ctime>
#include <climits>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "g2o_types.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace std;

// 一次测量的值，包括一个世界坐标系下三维点与一个灰度值
struct Measurement
{
    Measurement( Eigen::Vector3d p, float g ): pos_world(p), grayscale(g) {}
    Eigen::Vector3d pos_world;
    float grayscale;
};

inline Eigen::Vector3d project2Dto3D ( int x, int y, int d, float fx, float fy, float cx, float cy, float scale )
{
    float zz = float ( d ) /scale;
    float xx = zz* ( x-cx ) /fx;
    float yy = zz* ( y-cy ) /fy;
    return Eigen::Vector3d ( xx, yy, zz );
}

inline Eigen::Vector2d project3Dto2D( float x, float y, float z, float fx, float fy, float cx, float cy )
{
    float u = fx*x/z+cx;
    float v = fy*y/z+cy;
    return Eigen::Vector2d(u,v);
}

// 直接法估计位姿
// 输入：测量值（空间点的灰度），新的灰度图，相机内参； 输出：相机位姿
// 返回：true为成功，false失败
bool poseEstimationDirect( const vector<Measurement>& measurements, cv::Mat* gray, Eigen::Matrix3f& intrinsics, Eigen::Isometry3d& Tcw );

int main ( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"usage: useLK path_to_dataset"<<endl;
        return 1;
    }
    srand( (unsigned int) time(0) );
    string path_to_dataset = argv[1];
    string associate_file = path_to_dataset + "/associate.txt";

    ifstream fin ( associate_file );

    string rgb_file, depth_file, time_rgb, time_depth;
    cv::Mat color, depth, gray;
    vector<Measurement> measurements;
    // 相机内参
    float cx = 325.5;
    float cy = 253.5;
    float fx = 518.0;
    float fy = 519.0;
    float depth_scale = 1000.0;
    Eigen::Matrix3f K;
    K<<fx,0.f,cx,0.f,fy,cy,0.f,0.f,1.0f;

    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity(); 

    cv::Mat prev_color;
    //  我们演示两个图像间的直接法计算
    for ( int index=0; index<5; index++ )
    {
        cout<<"*********** loop "<<index<<" ************"<<endl;
        fin>>time_rgb>>rgb_file>>time_depth>>depth_file;
        color = cv::imread ( path_to_dataset+"/"+rgb_file );
        depth = cv::imread ( path_to_dataset+"/"+depth_file, -1 );
        cv::cvtColor ( color, gray, cv::COLOR_BGR2GRAY );
        if ( index ==0 )
        {
            // 对第一帧提取FAST特征点
            vector<cv::KeyPoint> keypoints;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect ( color, keypoints );
            for ( auto kp:keypoints )
            {
                // 去掉邻近边缘处的点
                if ( kp.pt.x < 20 || kp.pt.y < 20 || (kp.pt.x+20)>color.cols || (kp.pt.y+20)>color.rows )
                    continue;
                ushort d = depth.ptr<ushort> ( int( kp.pt.y ) ) [ int(kp.pt.x) ];
                if ( d==0 )
                    continue;
                Eigen::Vector3d p3d = project2Dto3D ( kp.pt.x, kp.pt.y, d, fx, fy, cx, cy, depth_scale );
                float grayscale = float ( gray.ptr<uchar> ( int(kp.pt.y) ) [ int(kp.pt.x) ] );
                measurements.push_back ( Measurement( p3d, grayscale ) );
            }
            prev_color = color.clone();
            continue;
        }
        // 使用直接法计算相机运动及投影点
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        poseEstimationDirect( measurements, &gray, K, Tcw );
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
        cout<<"direct method costs time: "<<time_used.count()<<" seconds."<<endl;
        cout<<"Tcw="<<Tcw.matrix()<<endl;
        
        // plot the feature points 
        cv::Mat img_show( color.rows*2, color.cols, CV_8UC3 );
        prev_color.copyTo( img_show( cv::Rect(0,0,color.cols, color.rows) ) );
        color.copyTo( img_show(cv::Rect(0,color.rows,color.cols, color.rows)) );
        for ( Measurement m:measurements )
        {
            if (rand() > RAND_MAX/5)
                continue;
            Eigen::Vector3d p = m.pos_world;
            Eigen::Vector2d pixel_prev = project3Dto2D( p(0,0), p(1,0), p(2,0), fx, fy, cx, cy );
            Eigen::Vector3d p2 = Tcw*m.pos_world;
            Eigen::Vector2d pixel_now = project3Dto2D( p2(0,0), p2(1,0), p2(2,0), fx, fy, cx, cy );
            
            float b = 255*float(rand())/RAND_MAX;
            float g = 255*float(rand())/RAND_MAX;
            float r = 255*float(rand())/RAND_MAX;
            cv::circle(img_show, cv::Point2d(pixel_prev(0,0), pixel_prev(1,0)), 8, cv::Scalar(b,g,r), 2 );
            cv::circle(img_show, cv::Point2d(pixel_now(0,0), pixel_now(1,0)+color.rows), 8, cv::Scalar(b,g,r), 2 );
            cv::line( img_show, cv::Point2d(pixel_prev(0,0), pixel_prev(1,0)), cv::Point2d(pixel_now(0,0), pixel_now(1,0)+color.rows), cv::Scalar(b,g,r), 1 );
        }
        cv::imshow( "result", img_show );
        cv::waitKey(0);
        
    }
    return 0;
}

bool poseEstimationDirect ( const vector< Measurement >& measurements, cv::Mat* gray, Eigen::Matrix3f& K, Eigen::Isometry3d& Tcw )
{
    // 初始化g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;  // 求解的向量是6＊1的
    DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense< DirectBlock::PoseMatrixType > ();
    DirectBlock* solver_ptr = new DirectBlock( linearSolver );
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr ); // G-N
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );  // L-M
    g2o::SparseOptimizer optimizer; 
    optimizer.setAlgorithm( solver );
    optimizer.setVerbose( true );       // 打开调试输出
    
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); 
    pose->setEstimate( g2o::SE3Quat(Tcw.rotation(), Tcw.translation()) );
    pose->setId(0);
    optimizer.addVertex( pose );
    
    // 添加边
    int id=1;
    for( Measurement m: measurements )
    {
        g2o::EdgeSE3ProjectDirect* edge = new g2o::EdgeSE3ProjectDirect(
            m.pos_world,
            K(0,0), K(1,1), K(0,2), K(1,2), gray
        );
        edge->setVertex( 0, pose );
        edge->setMeasurement( m.grayscale );
        edge->setInformation( Eigen::Matrix<double,1,1>::Identity() );
        edge->setId( id++ );
        optimizer.addEdge(edge);
    }
    cout<<"edges in graph: "<<optimizer.edges().size()<<endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);
    Tcw = pose->estimate();
}

