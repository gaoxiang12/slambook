#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>

#include <sophus/se3.h>
#include <sophus/so3.h>

#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

using namespace std;
using Sophus::SE3;
using Sophus::SO3;

/************************************************
 * 本程序演示如何用g2o solver进行位姿图优化
 * sphere.g2o是人工生成的一个Pose graph，我们来优化它。
 * 尽管可以直接通过load函数读取整个图，但我们还是自己来实现读取代码，以期获得更深刻的理解
 * 本节使用李代数表达位姿图，节点和边的方式为自定义
 * **********************************************/

typedef Eigen::Matrix<double,6,6> Matrix6d;

// 给定误差求J_R^{-1}的近似
Matrix6d JRInv( SE3 e )
{
    Matrix6d J;
    J.block(0,0,3,3) = SO3::hat(e.so3().log());
    J.block(0,3,3,3) = SO3::hat(e.translation());
    J.block(3,0,3,3) = Eigen::Matrix3d::Zero(3,3);
    J.block(3,3,3,3) = SO3::hat(e.so3().log());
    J = J*0.5 + Matrix6d::Identity();
    return J;
}
// 李代数顶点
typedef Eigen::Matrix<double, 6, 1> Vector6d;
class VertexSE3LieAlgebra: public g2o::BaseVertex<6, SE3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool read ( istream& is )
    {
        double data[7];
        for ( int i=0; i<7; i++ )
            is>>data[i];
        setEstimate ( SE3 (
                Eigen::Quaterniond ( data[6],data[3], data[4], data[5] ),
                Eigen::Vector3d ( data[0], data[1], data[2] )
        ));
    }

    bool write ( ostream& os ) const
    {
        os<<id()<<" ";
        Eigen::Quaterniond q = _estimate.unit_quaternion();
        os<<_estimate.translation().transpose()<<" ";
        os<<q.coeffs()[0]<<" "<<q.coeffs()[1]<<" "<<q.coeffs()[2]<<" "<<q.coeffs()[3]<<endl;
        return true;
    }
    
    virtual void setToOriginImpl()
    {
        _estimate = Sophus::SE3();
    }
    // 左乘更新
    virtual void oplusImpl ( const double* update )
    {
        Sophus::SE3 up (
            Sophus::SO3 ( update[3], update[4], update[5] ),
            Eigen::Vector3d ( update[0], update[1], update[2] )
        );
        _estimate = up*_estimate;
    }
};

// 两个李代数节点之边
class EdgeSE3LieAlgebra: public g2o::BaseBinaryEdge<6, SE3, VertexSE3LieAlgebra, VertexSE3LieAlgebra>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool read ( istream& is )
    {
        double data[7];
        for ( int i=0; i<7; i++ )
            is>>data[i];
        Eigen::Quaterniond q ( data[6], data[3], data[4], data[5] );
        q.normalize();
        setMeasurement (
            Sophus::SE3 ( q, Eigen::Vector3d ( data[0], data[1], data[2] ) ) 
        );
        for ( int i=0; i<information().rows() && is.good(); i++ )
            for ( int j=i; j<information().cols() && is.good(); j++ )
            {
                is >> information() ( i,j );
                if ( i!=j )
                    information() ( j,i ) =information() ( i,j );
            }
        return true;
    }
    bool write ( ostream& os ) const
    {
        VertexSE3LieAlgebra* v1 = static_cast<VertexSE3LieAlgebra*> (_vertices[0]);
        VertexSE3LieAlgebra* v2 = static_cast<VertexSE3LieAlgebra*> (_vertices[1]);
        os<<v1->id()<<" "<<v2->id()<<" ";
        SE3 m = _measurement;
        Eigen::Quaterniond q = m.unit_quaternion();
        os<<m.translation().transpose()<<" ";
        os<<q.coeffs()[0]<<" "<<q.coeffs()[1]<<" "<<q.coeffs()[2]<<" "<<q.coeffs()[3]<<" ";
        // information matrix 
        for ( int i=0; i<information().rows(); i++ )
            for ( int j=i; j<information().cols(); j++ )
            {
                os << information() ( i,j ) << " ";
            }
        os<<endl;
        return true;
    }

    // 误差计算与书中推导一致
    virtual void computeError()
    {
        Sophus::SE3 v1 = (static_cast<VertexSE3LieAlgebra*> (_vertices[0]))->estimate();
        Sophus::SE3 v2 = (static_cast<VertexSE3LieAlgebra*> (_vertices[1]))->estimate();
        _error = (_measurement.inverse()*v1.inverse()*v2).log();
    }
    
    // 雅可比计算
    virtual void linearizeOplus()
    {
        Sophus::SE3 v1 = (static_cast<VertexSE3LieAlgebra*> (_vertices[0]))->estimate();
        Sophus::SE3 v2 = (static_cast<VertexSE3LieAlgebra*> (_vertices[1]))->estimate();
        Matrix6d J = JRInv(SE3::exp(_error));
        // 尝试把J近似为I？
        _jacobianOplusXi = - J* v2.inverse().Adj();
        _jacobianOplusXj = J*v2.inverse().Adj();
    }
};

int main ( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"Usage: pose_graph_gtsam sphere.g2o"<<endl;
        return 1;
    }
    ifstream fin ( argv[1] );
    if ( !fin )
    {
        cout<<"file "<<argv[1]<<" does not exist."<<endl;
        return 1;
    }
    
    gtsam::NonlinearFactorGraph::shared_ptr graph( new gtsam::NonlinearFactorGraph);
    gtsam::Values::shared_ptr initial( new gtsam::Values ); // init values
    // read from g2o file 
    while ( !fin.eof() )
    {
        string tag; 
        fin>>tag; 
        if ( tag == "VERTEX_SE3:QUAT" )
        {
            gtsam::Key id;
            fin>>id;
            double data[7];
            for ( int i=0; i<7; i++ ) fin>>data[i];
            gtsam::Rot3 R = gtsam::Rot3::Quaternion(data[6], data[3], data[4], data[5]);
            gtsam::Point3 t(data[0], data[1], data[2]);
            initial->insert(id, gtsam::Pose3(R,t));
        }
        else if ( tag == "EDGE_SE3:QUAT")
        {
            // factor 
            gtsam::Matrix m = gtsam::I_6x6;
            gtsam::Key id1, id2; 
            fin>>id1>>id2;
            double data[7];
            for ( int i=0; i<7; i++ ) fin>>data[i];
            gtsam::Rot3 R = gtsam::Rot3::Quaternion(data[6], data[3], data[4], data[5]);
            gtsam::Point3 t(data[0], data[1], data[2]);
            
        }
        if (!fin.good())
            break;
    }

}
