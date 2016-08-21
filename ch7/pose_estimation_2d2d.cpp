

void pose_estimation_2d2d(  std::vector<KeyPoint> keypoints_1,
			    std::vector<KeyPoint> keypoints_2,
			    std::vector< DMatch > matches){

  // 相机内参,TUM Freiburg2		
  Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

  
  //-- 把匹配点转换为vector<Point2f>的形式
  int point_count = max(keypoints_1.size(),keypoints_2.size());
  
  vector<Point2f> points1(point_count);
  vector<Point2f> points2(point_count);
  
  for( int i = 0; i < (int)matches.size(); i++ ){
    points1[i] =  keypoints_1[matches[i].queryIdx].pt;
    points2[i] =  keypoints_2[matches[i].trainIdx].pt;
  }

 
  //-- 计算基础矩阵
  Mat fundamental_matrix;
  fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT, 3, 0.99);
  cout<<"fundamental_matrix is "<< fundamental_matrix<<endl;
  
  
  //-- 计算本质矩阵 
  Point2d principal_point(325.1, 249.7);						//相机主点, TUM dataset标定值
  int focal_length = 521;								//相机焦距, TUM dataset标定值
  Mat essential_matrix, R, t;
  essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point, RANSAC);	
  cout<<"essential_matrix is "<< essential_matrix<<endl;

  
  //-- 计算单应矩阵
  Mat homography_matrix;
  homography_matrix = findHomography(points1, points2, RANSAC, 3, noArray(), 2000, 0.99);
  cout<<"homography_matrix is "<< homography_matrix<<endl;
  
  
  //-- 从本质矩阵中恢复旋转和平移信息.
  recoverPose(essential_matrix, points2, points1, R, t, focal_length, principal_point);				
  cout<<"R is "<<R<<endl;
  cout<<"t is "<<t<<endl;
}