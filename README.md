# NOTE
[Slambook 2](https://github.com/gaoxiang12/slambook2) has be released since 2019.8 which has better support on Ubuntu 18.04 and has a lot of new features. Slambook 1 will still be available on github but I suggest new readers switch to the second version. 

# slambook
This is the code written for my new book about visual SLAM called "14 lectures on visual SLAM" which was released in April 2017. It is highy recommended to download the code and run it in you own machine so that you can learn more efficiently and also modify it. The code is stored by chapters like "ch2" and "ch4". Note that chapter 9 is a project so I stored it in the "project" directory.

If you have any questions about the code, please add an issue so I can see it. Contact me for more information: gao dot xiang dot thu at gmail dot com.

These codes are under MIT license. You don't need permission to use it or change it. 
Please cite this book if you are doing academic work:
Xiang Gao, Tao Zhang, Yi Liu, Qinrui Yan, 14 Lectures on Visual SLAM: From Theory to Practice, Publishing House of Electronics Industry, 2017.

In LaTeX:
`` @Book{Gao2017SLAM, 
title={14 Lectures on Visual SLAM: From Theory to Practice}, 
publisher = {Publishing House of Electronics Industry},
year = {2017},
author = {Xiang Gao and Tao Zhang and Yi Liu and Qinrui Yan},
} ``

For English readers, we are currently translating this book into an online version, see [this page](https://gaoxiang12.github.io/slambook-en/) for details.

# Contents
- ch1 Preface
- ch2 Overview of SLAM & linux, cmake
- ch3 Rigid body motion & Eigen
- ch4 Lie group and Lie Algebra & Sophus
- ch5 Cameras and Images & OpenCV
- ch6 Non-linear optimization & Ceres, g2o
- ch7 Feature based Visual Odometry
- ch8 Direct (Intensity based) Visual Odometry
- ch9 Project
- ch10 Back end optimization & Ceres, g2o
- ch11 Pose graph and Factor graph & g2o, gtsam
- ch12 Loop closure & DBoW3
- ch13 Dense reconstruction & REMODE, Octomap

# slambook (中文说明)
我最近写了一本有关视觉SLAM的书籍，这是它对应的代码。书籍将会在明年春天由电子工业出版社出版。

我强烈建议你下载这个代码。书中虽然给出了一部分，但你最好在自己的机器上编译运行它们，然后对它们进行修改以获得更好的理解。这本书的代码是按章节划分的，比如第二章内容在”ch2“文件夹下。注意第九章是工程，所以我们没有”ch9“这个文件夹，而是在”project“中存储它。

如果你在运行代码中发现问题，请在这里提交一个issue，我就能看到它。如果你有更多的问题，请给我发邮件：gaoxiang12 dot mails dot tsinghua dot edu dot cn.

本书代码使用MIT许可。使用或修改、发布都不必经过我的同意。不过，如果你是在学术工作中使用它，建议你引用本书作为参考文献。

引用格式：
高翔, 张涛, 颜沁睿, 刘毅, 视觉SLAM十四讲：从理论到实践, 电子工业出版社, 2017

LaTeX格式:
`` @Book{Gao2017SLAM, 
title={视觉SLAM十四讲：从理论到实践}, 
publisher = {电子工业出版社},
year = {2017},
author = {高翔 and 张涛 and 刘毅 and 颜沁睿},
lang = {zh}
} ``

# 目录
- ch2 概述，cmake基础
- ch3 Eigen，三维几何
- ch4 Sophus，李群与李代数
- ch5 OpenCV，图像与相机模型
- ch6 Ceres and g2o，非线性优化
- ch7 特征点法视觉里程计
- ch8 直接法视觉里程计
- ch9 project
- ch10 Ceres and g2o，后端优化1
- ch11 g2o and gtsam，位姿图优化
- ch12 DBoW3，词袋方法
- ch13 稠密地图构建

关于勘误，请参照本代码根目录下的errata.xlsx文件。此文件包含本书从第一次印刷至现在的勘误信息。勘误将随着书籍的印刷版本更新。

# 备注
百度云备份：[https://pan.baidu.com/s/1slDE7cL]
Videos: [https://space.bilibili.com/38737757]
