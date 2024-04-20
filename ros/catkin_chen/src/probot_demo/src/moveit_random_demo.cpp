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

#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <ros/package.h>
using namespace std;
#include <iostream>
#include <fstream>
int main(int argc, char **argv)
{
    ros::init(argc, argv, "moveit_random_demo");
    ros::AsyncSpinner spinner(1);
    spinner.start();

    moveit::planning_interface::MoveGroupInterface arm("manipulator");

    arm.setGoalJointTolerance(0.001);

    arm.setMaxAccelerationScalingFactor(0.2);
    arm.setMaxVelocityScalingFactor(0.2);
    //获取终端link的名称
    std::string end_effector_link = arm.getEndEffectorLink();

    geometry_msgs::Pose start_pose = arm.getCurrentPose(end_effector_link).pose;
    ros::Rate loop_rate(10);

    std::string FileName = ros::package::getPath("demo_chen");
//**************************
    ofstream myfile_data_r(FileName + "/data/data_r.txt");
    if (!myfile_data_r.is_open())
    {
        cout << "未成功打开文件 data" << endl;
        return 0;
    }
    myfile_data_r.clear();
//**************************
//**************************
    ofstream myfile_data_t(FileName + "/data/data_t.txt");
    if (!myfile_data_t.is_open())
    {
        cout << "未成功打开文件 data" << endl;
        return 0;
    }
    myfile_data_t.clear();
//**************************
    while(ros::ok())
    {
        start_pose = arm.getCurrentPose(end_effector_link).pose;
        start_force =

        myfile_data_t<<start_pose.position.x<<" "<<start_pose.position.y<<" "<<start_pose.position.z<<endl;
        myfile_data_r<<start_pose.orientation.x<<" "<<start_pose.orientation.y<<" "<<start_pose.orientation.z<<" "<<start_pose.orientation.w<<endl;
        loop_rate.sleep();
    }

    ros::shutdown();

    return 0;
}
