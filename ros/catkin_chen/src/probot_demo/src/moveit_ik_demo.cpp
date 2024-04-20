/***********************************************************************
Copyright 2019 Wuhan PS-Micro Technology Co., Itd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
***********************************************************************/

#include <string>
#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <ros/package.h>
using namespace std;
#include <iostream>
#include <fstream>
int main(int argc, char **argv)
{
    std::string FileName = ros::package::getPath("demo_chen");
//    ifstream myfile_tra(FileName + "/data/rollout_tra.txt");
    ifstream myfile_tra(FileName + "/data/demo_tra.txt");
    if (!myfile_tra.is_open())
    {
        cout << "未成功打开文件 data" << endl;
        return 0;
    }

    ros::init(argc, argv, "moveit_fk_demo");
    ros::AsyncSpinner spinner(1);
    spinner.start();

    moveit::planning_interface::MoveGroupInterface arm("manipulator");

    //获取终端link的名称
    std::string end_effector_link = arm.getEndEffectorLink();

    //设置目标位置所使用的参考坐标系
    std::string reference_frame = "base_link";
    arm.setPoseReferenceFrame(reference_frame);

    //当运动规划失败后，允许重新规划
    arm.allowReplanning(true);

    //设置位置(单位：米)和姿态（单位：弧度）的允许误差
    arm.setGoalPositionTolerance(0.001);
    arm.setGoalOrientationTolerance(0.01);

    //设置允许的最大速度和加速度
    arm.setMaxAccelerationScalingFactor(0.2);
    arm.setMaxVelocityScalingFactor(0.2);

    geometry_msgs::Pose poses = arm.getCurrentPose(end_effector_link).pose;
    std::vector<geometry_msgs::Pose> waypoints;
    int sum=0;
    myfile_tra>>sum;
    for(int i=0;i<sum;i++)
    {
        myfile_tra>>poses.position.x;
        myfile_tra>>poses.position.y;
        myfile_tra>>poses.position.z;

        myfile_tra>>poses.orientation.x;
        myfile_tra>>poses.orientation.y;
        myfile_tra>>poses.orientation.z;
        myfile_tra>>poses.orientation.w;
        waypoints.push_back(poses);
    }

    // 笛卡尔空间下的路径规划
    moveit_msgs::RobotTrajectory trajectory;
    const double jump_threshold = 0.0;
    const double eef_step = 0.1;
    double fraction = 0.0;
    int maxtries = 100;   //最大尝试规划次数
    int attempts = 0;     //已经尝试规划次数

    while(fraction < 1.0 && attempts < maxtries)
    {
        fraction = arm.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
        attempts++;

        if(attempts % 10 == 0)
            ROS_INFO("Still trying after %d attempts...", attempts);
    }

    if(fraction == 1)
    {
        ROS_INFO("Path computed successfully. Moving the arm.");

        // 生成机械臂的运动规划数据
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        plan.trajectory_ = trajectory;

        // 执行运动
        arm.execute(plan);
    }
    else
    {
        ROS_INFO("Path planning failed with only %0.6f success after %d attempts.", fraction, maxtries);
    }

    ros::shutdown(); 

    return 0;
}
