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
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <ros/package.h>
int main(int argc, char **argv)
{
	ros::init(argc, argv, "moveit_cartesian_demo");
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

    // 控制机械臂先回到初始化位置

// 获取当前位姿数据最为机械臂运动的起始位姿
    geometry_msgs::Pose start_pose = arm.getCurrentPose(end_effector_link).pose;
    start_pose.orientation.x = 0.764433;
    start_pose.orientation.y = -0.644402;
    start_pose.orientation.z = 0.0133613;
    start_pose.orientation.w = 0.0144779;

    start_pose.position.x = 0.535312;
    start_pose.position.y = 0.190389;
    start_pose.position.z = 0.381599;
//    ros::Rate loop_rate(100);
//    while(ros::ok())
//    {
//        start_pose = arm.getCurrentPose(end_effector_link).pose;
//        std::cout<<start_pose.position;
//        std::cout<<start_pose.orientation<<std::endl;
//
//        loop_rate.sleep();
//    }



	std::vector<geometry_msgs::Pose> waypoints;
    geometry_msgs::Pose target_pose= start_pose;
	int sum=100;
	double r=0.1;

    tf::Quaternion q;
    tf::Quaternion q_start;
    tf::Quaternion q1;
    tf::quaternionMsgToTF(start_pose.orientation,q_start);

	for(int i=0;i<sum;i++)
    {
	    q.setEuler(M_PI/6.0*sin(i*M_PI/sum/2.0),0,0);
	    q1 = q_start*q;
        tf::quaternionTFToMsg(q1,target_pose.orientation);
        target_pose.position.x = start_pose.position.x + r*cos(i*M_PI/sum);
        target_pose.position.y = start_pose.position.y + r*sin(i*M_PI/sum);
        waypoints.push_back(target_pose);
    };

	// 笛卡尔空间下的路径规划
	moveit_msgs::RobotTrajectory trajectory;
	const double jump_threshold = 0.0;
	const double eef_step = 0.01;
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

    std::cout<<"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"<<std::endl;

	ros::shutdown(); 
	return 0;
}
