#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle, Yin Wu
Email   : heethesh@cmu.edu
Version : 1.2.1
Date    : Jan 20, 2019

Description:
Script to find the transformation between the Camera and the LiDAR

Example Usage:
1. To perform calibration using the GUI to pick correspondences:

    $ rosrun lidar_camera_calibration calibrate_camera_lidar.py --calibrate

    The point correspondences will be save as following:
    - PKG_PATH/calibration_data/lidar_camera_calibration/img_corners.npy
    - PKG_PATH/calibration_data/lidar_camera_calibration/pcl_corners.npy

    The calibrate extrinsic are saved as following:
    - PKG_PATH/calibration_data/lidar_camera_calibration/extrinsics.npz
    --> 'euler' : euler angles (3, )
    --> 'R'     : rotation matrix (3, 3)
    --> 'T'     : translation offsets (3, )

2. To display the LiDAR points projected on to the camera plane:

    $ roslaunch lidar_camera_calibration display_camera_lidar_calibration.launch

Notes:
Make sure this file has executable permissions:
$ chmod +x calibrate_camera_lidar.py

References: 
http://wiki.ros.org/message_filters
http://wiki.ros.org/cv_bridge/Tutorials/
http://docs.ros.org/api/image_geometry/html/python/
http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscribe
'''

# Python 2/3 compatibility
from __future__ import print_function
import roslib
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import TransformStamped
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from tf.transformations import euler_from_matrix, quaternion_from_euler
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import image_geometry
import ros_numpy
import tf2_ros
import rospy
import rosbag

# Built-in modules
import os
import sys
import time
import threading
import multiprocessing
from typing import List

# External modules
import cv2
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ROS modules
PKG = 'lidar_camera_calibration'
roslib.load_manifest(PKG)
# ! give a initial guess before estimating
left_r = [-1.7687376706546014, -0.863365583690817, 3.0802177953498013]
left_t = [-0.01177157, -0.19403029,  1.10526402]
rvec_guess = np.array([left_r]).transpose()
tvec_guess = np.array([left_t]).transpose()

# * Parameters
FLAG_GUESS = False
EMPTY_DISTORTION = True

# Global variables
OUSTER_LIDAR = True
FLAG_PAUSE = False
FLAG_FIRST_TIME = True
KEY_LOCK = threading.Lock()
CV_BRIDGE = CvBridge()

# Global paths
PKG_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CALIB_PATH = 'calibration_data/lidar_camera_calibration'


x = 0
y = -8.4
z = -4.86
a = 10
b = 10
c = 10

# x = 1.10
# y = 0.03
# z = -2.98
# a = 8.17
# b = 10.61
# c = 8.24


'''
Keyboard handler thread
Inputs: None
Outputs: None
'''


def handle_keyboard():
    global KEY_LOCK, FLAG_PAUSE
    key = input('Press [ENTER] to FLAG_pause and pick points\n')
    with KEY_LOCK:
        FLAG_PAUSE = True


'''
Start the keyboard handler thread
Inputs: None
Outputs: None
'''


def start_keyboard_handler():
    keyboard_t = threading.Thread(target=handle_keyboard)
    keyboard_t.daemon = True
    keyboard_t.start()


'''
Save the point correspondences and image data
Points data will be appended if file already exists

Inputs:
    data - [numpy array] - points or opencv image
    filename - [str] - filename to save
    folder - [str] - folder to save at
    is_image - [bool] - to specify whether points or image data

Outputs: None
'''


def save_data(data, filename, folder, is_image=False, overwrite=False):
    # Empty data
    if not len(data):
        return

    # Handle filename
    filename = os.path.join(PKG_PATH, os.path.join(folder, filename))

    # Create folder
    try:
        os.makedirs(os.path.join(PKG_PATH, folder))
    except OSError:
        if not os.path.isdir(os.path.join(PKG_PATH, folder)):
            raise

    # Save image
    if is_image:
        cv2.imwrite(filename, data)
        return

    # overwrite file
    if overwrite:
        np.savetxt(filename, data)
        return

    # Save points data
    if os.path.isfile(filename):
        rospy.logwarn('Updating file: %s' % filename)
        data = np.vstack((np.loadtxt(filename), data))
    np.savetxt(filename, data)


'''
Runs the image point selection GUI process

Inputs:
    img_msg - [sensor_msgs/Image] - ROS sensor image message
    now - [int] - ROS bag time in seconds
    rectify - [bool] - to specify whether to rectify image or not

Outputs:
    Picked points saved in PKG_PATH/CALIB_PATH/img_corners.npy
'''


def extract_points_2D(img_msg, now, rectify=False):
    # Log PID
    rospy.loginfo('2D Picker PID: [%d]' % os.getpid())

    # Read image using CV bridge
    try:
        img = CV_BRIDGE.imgmsg_to_cv2(img_msg, 'bgr8')
    except CvBridgeError as e:
        rospy.logerr(e)
        return

    # Rectify image
    if rectify:
        CAMERA_MODEL.rectifyImage(img, img)
    disp = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    # Setup matplotlib GUI
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Select 2D Image Points - %d' % now.secs)
    ax.set_axis_off()
    ax.imshow(disp)

    # Pick points
    picked, corners = [], []

    def onclick(event):
        if event.key != 'x':
            return

        x = event.xdata
        y = event.ydata
        if (x is None) or (y is None):
            return

        # Display the picked point
        picked.append((x, y))
        corners.append((x, y))
        rospy.loginfo('IMG: %s', str(picked[-1]))

        if len(picked) > 1:
            # Draw the line
            temp = np.array(picked)
            ax.plot(temp[:, 0], temp[:, 1])
            ax.figure.canvas.draw_idle()

            # Reset list for future pick events
            del picked[0]

    # Display GUI
    fig.canvas.mpl_connect('key_press_event', onclick)
    plt.show()

    # Save corner points and image
    rect = '_rect' if rectify else ''
    if len(corners) > 1:
        del corners[-1]  # Remove last duplicate
    save_data(corners, 'img_corners.txt', CALIB_PATH)
    save_data(img, 'image_color-%d.jpg' % (now.secs),
              os.path.join(CALIB_PATH, 'images'), True)


'''
Runs the LiDAR point selection GUI process

Inputs:
    velodyne - [sensor_msgs/PointCloud2] - ROS velodyne PCL2 message
    now - [int] - ROS bag time in seconds

Outputs:
    Picked points saved in PKG_PATH/CALIB_PATH/pcl_corners.npy
'''


def extract_points_3D(pointcloud_msg, now):
    # Log PID
    rospy.loginfo('3D Picker PID: [%d]' % os.getpid())

    # Extract points data
    points = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud_msg)
    points = np.asarray(points.tolist())

    # Group all beams together and pick the first 4 columns for X, Y, Z, intensity.
    if OUSTER_LIDAR:
        points = points.reshape(-1, 9)[:, :4]

    # Select points within chessboard range
    global x, y, z, a, b, c
    inrange = np.where((points[:, 0] > x) & (points[:, 0] < x+a) &  # param for "camera left" and "lidar_1"
                       (points[:, 1] > y) & (points[:, 1] < y+b) &
                       (points[:, 2] > z) & (points[:, 2] < z+c))

    points = points[inrange[0]]
    print(points.shape)
    if points.shape[0] > 5:
        rospy.loginfo('PCL points available: %d', points.shape[0])
    else:
        rospy.logwarn('Very few PCL points available in range')
        return

    # Color map for the points
    cmap = matplotlib.cm.get_cmap('hsv')
    colors = cmap(points[:, -1] / np.max(points[:, -1]))

    # Setup matplotlib GUI
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Select 3D LiDAR Points - %d' % now.secs, color='white')
    # ax.set_axis_off()
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_facecolor((0, 0, 0))
    ax.scatter(points[:, 0], points[:, 1],
               points[:, 2], c=colors, s=2, picker=5)

    # Equalize display aspect ratio for all axes
    max_range = (np.array([points[:, 0].max() - points[:, 0].min(),
                           points[:, 1].max() - points[:, 1].min(),
                           points[:, 2].max() - points[:, 2].min()]).max() / 2.0)
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Pick points
    picked, corners = [], []

    def onpick(event):
        ind = event.ind[0]
        x, y, z = event.artist._offsets3d

        # Ignore if same point selected again
        if picked and (x[ind] == picked[-1][0] and y[ind] == picked[-1][1] and z[ind] == picked[-1][2]):
            return

        # Display picked point
        picked.append((x[ind], y[ind], z[ind]))
        corners.append((x[ind], y[ind], z[ind]))
        rospy.loginfo('PCL: %s', str(picked[-1]))

        if len(picked) > 1:
            # Draw the line
            temp = np.array(picked)
            ax.plot(temp[:, 0], temp[:, 1], temp[:, 2])
            ax.figure.canvas.draw_idle()

            # Reset list for future pick events
            del picked[0]

    # Display GUI
    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()

    # Save corner points
    if len(corners) > 1:
        del corners[-1]  # Remove last duplicate
    save_data(corners, 'pcl_corners.txt', CALIB_PATH)


'''
Calibrate the LiDAR and image points using OpenCV PnP RANSAC
Requires minimum 5 point correspondences

Inputs:
    points2D - [numpy array] - (N, 2) array of image points
    points3D - [numpy array] - (N, 3) array of 3D points

Outputs:
    Extrinsics saved in PKG_PATH/CALIB_PATH/extrinsics.npz
'''


def remove_points():
    # Load corresponding points
    folder = os.path.join(PKG_PATH, CALIB_PATH)

    points2D = np.loadtxt(os.path.join(folder, 'img_corners.txt'))
    points3D = np.loadtxt(os.path.join(folder, 'pcl_corners.txt'))

    tobe_removed_list = []

    cmap = matplotlib.cm.get_cmap('hsv')
    colors = cmap(points3D[:, -1] / np.max(points3D[:, -1]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Remove 3D LiDAR Points', color='white')
    ax.set_axis_off()
    ax.set_facecolor((0, 0, 0))
    ax.scatter(points3D[:, 0], points3D[:, 1],
               points3D[:, 2], c=colors, s=2, picker=5)

    def onpick(event):
        ind = event.ind[0]
        x, y, z = event.artist._offsets3d

        tobe_removed_list.append(ind)

        # Display picked point
        print([x[ind], y[ind], z[ind]])

    # Display GUI
    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()

    if len(tobe_removed_list) != 0:
        # remove points
        rospy.logwarn(f"deleting index {tobe_removed_list}")

        points2D = np.delete(points2D, tobe_removed_list, axis=0)
        points3D = np.delete(points3D, tobe_removed_list, axis=0)

        save_data(points2D, 'img_corners.txt', CALIB_PATH, overwrite=True)
        save_data(points3D, 'pcl_corners.txt', CALIB_PATH, overwrite=True)
    else:
        rospy.logwarn("no need to delete")


def calibrate(tf_publisher: tf2_ros.StaticTransformBroadcaster):
    # Load corresponding points
    folder = os.path.join(PKG_PATH, CALIB_PATH)

    points2D = np.loadtxt(os.path.join(folder, 'img_corners.txt'))
    points3D = np.loadtxt(os.path.join(folder, 'pcl_corners.txt'))

    # Check points shape
    if not (points2D.shape[0] >= 5):
        rospy.logwarn('PnP RANSAC Requires minimum 5 points')
        return
    assert(points2D.shape[0] == points3D.shape[0])

    # Obtain camera matrix and distortion coefficients
    camera_matrix = CAMERA_MODEL.intrinsicMatrix()
    dist_coeffs = CAMERA_MODEL.distortionCoeffs()
    print(dist_coeffs)

    empty = np.zeros((5, 1), np.float32)
    if EMPTY_DISTORTION:
        dist_coeffs = empty
    # empty = np.expand_dims(empty, 0)
    # breakpoint()
    global rvec_guess, tvec_guess, FLAG_GUESS
    # Estimate extrinsics
    if FLAG_GUESS:
        rospy.loginfo("use method SOLVEPNP_ITERATIVE with init guess")
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            points3D, points2D, camera_matrix, dist_coeffs,
            rvec=rvec_guess.copy(), tvec=tvec_guess.copy(),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        rospy.loginfo("use method SOLVEPNP_EPNP")
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            points3D, points2D, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)

    rvec_guess = rotation_vector
    tvec_guess = translation_vector

    # Compute re-projection error.
    points2D_reproj = cv2.projectPoints(points3D, rotation_vector,
                                        translation_vector, camera_matrix, dist_coeffs)[0].squeeze(1)
    assert(points2D_reproj.shape == points2D.shape)

    # Compute error only over inliers.
    error = (points2D_reproj - points2D)[inliers]
    error = error.squeeze()
    rmse = np.sqrt(np.mean(error[:, 0] ** 2 + error[:, 1] ** 2))
    rospy.loginfo(
        'Re-projection error before LM refinement (RMSE) in px: ' + str(rmse))

    # Refine estimate using LM
    if not success:
        rospy.logwarn('Initial estimation unsuccessful, skipping refinement')
    elif not hasattr(cv2, 'solvePnPRefineLM'):
        rospy.logwarn(
            'solvePnPRefineLM requires OpenCV >= 4.1.1, skipping refinement')
    else:
        assert len(
            inliers) >= 3, 'LM refinement requires at least 3 inlier points'
        rotation_vector, translation_vector = cv2.solvePnPRefineLM(points3D[inliers],
                                                                   points2D[inliers], camera_matrix, dist_coeffs, rotation_vector, translation_vector)

        # Compute re-projection error.
        points2D_reproj = cv2.projectPoints(points3D, rotation_vector,
                                            translation_vector, camera_matrix, dist_coeffs)[0].squeeze(1)
        assert(points2D_reproj.shape == points2D.shape)
        # Compute error only over inliers.
        error = (points2D_reproj - points2D)[inliers]
        error = error.squeeze()
        rmse = np.sqrt(np.mean(error[:, 0] ** 2 + error[:, 1] ** 2))
        rospy.loginfo(
            'Re-projection error after LM refinement (RMSE) in px: ' + str(rmse))

    # Convert rotation vector
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    euler = euler_from_matrix(rotation_matrix)
    q = quaternion_from_euler(euler[0], euler[1], euler[2])

    # Publish newest tf
    global PARENT_FRAME_ID, CHILD_FRAME_ID
    tf_msg = TransformStamped()
    tf_msg.header.frame_id = PARENT_FRAME_ID
    tf_msg.child_frame_id = CHILD_FRAME_ID
    tf_msg.transform.translation.x = translation_vector.squeeze()[0]
    tf_msg.transform.translation.y = translation_vector.squeeze()[1]
    tf_msg.transform.translation.z = translation_vector.squeeze()[2]
    tf_msg.transform.rotation.x = q[0]
    tf_msg.transform.rotation.y = q[1]
    tf_msg.transform.rotation.z = q[2]
    tf_msg.transform.rotation.w = q[3]
    tf_publisher.sendTransform(tf_msg)

    # Save extrinsics
    np.savez(os.path.join(folder, 'extrinsics.npz'),
             euler=euler, R=rotation_matrix, T=translation_vector.T)

    # Display results
    def rad2degree(rad: float) -> float:
        return rad * 180 / np.pi
    print('Euler angles (RPY degree): ', rad2degree(
        euler[0]), rad2degree(euler[1]), rad2degree(euler[2]))
    print('Euler angles (RPY):', euler)
    print('Rotation Matrix:', rotation_matrix)
    print('Translation Offsets:', translation_vector.T)


'''
Callback function to publish project image and run calibration

Inputs:
    image - [sensor_msgs/Image] - ROS sensor image message
    camera_info - [sensor_msgs/CameraInfo] - ROS sensor camera info message
    velodyne - [sensor_msgs/PointCloud2] - ROS velodyne PCL2 message
    image_pub - [sensor_msgs/Image] - ROS image publisher

Outputs: None
'''


def callback(image, camera_info, pointcloud, tf_pub):
    global CAMERA_MODEL, FLAG_FIRST_TIME, FLAG_PAUSE, PARENT_FRAME_ID, CHILD_FRAME_ID

    # Setup the pinhole camera model
    if FLAG_FIRST_TIME:
        FLAG_FIRST_TIME = False

        CAMERA_MODEL = image_geometry.PinholeCameraModel()

        # Setup camera model
        rospy.loginfo('Setting up camera model')
        CAMERA_MODEL.fromCameraInfo(camera_info)

        # set parameters
        PARENT_FRAME_ID = image.header.frame_id
        CHILD_FRAME_ID = pointcloud.header.frame_id

    # # Projection/display mode
    # if PROJECT_MODE:
    #     project_point_cloud(velodyne, image, image_pub)

    # Calibration mode
    elif FLAG_PAUSE:
        # Create GUI processes to pick 3D, 2D points
        now = rospy.get_rostime()
        img_p = multiprocessing.Process(
            target=extract_points_2D, args=[image, now])
        pcl_p = multiprocessing.Process(
            target=extract_points_3D, args=[pointcloud, now])
        img_p.start()
        pcl_p.start()
        img_p.join()
        pcl_p.join()

        # Create process to check and remove 3D, 2D points
        remove_p = multiprocessing.Process(target=remove_points)
        remove_p.start()
        remove_p.join()

        calibrate(tf_pub)

        with KEY_LOCK:
            FLAG_PAUSE = False

        start_keyboard_handler()


'''
The main ROS node which handles the topics

Inputs:
    camera_info - [str] - ROS sensor camera info topic
    image_color - [str] - ROS sensor image topic
    velodyne - [str] - ROS velodyne PCL2 topic
    camera_lidar - [str] - ROS projected points image topic

Outputs: None
'''


def listener(image_topic, camera_info_topic, pointcloud_topic):
    # Start node
    rospy.loginfo('Current PID: [%d]' % os.getpid())
    rospy.loginfo('Image Topic: %s' % image_topic)
    rospy.loginfo('Camera Info Topic: %s' % camera_info_topic)
    rospy.loginfo('PointCloud2 Topic: %s' % pointcloud_topic)

    # Subscribe to topics
    image_sub = message_filters.Subscriber(image_topic, Image)
    camera_info_sub = message_filters.Subscriber(camera_info_topic, CameraInfo)
    pointcloud_sub = message_filters.Subscriber(pointcloud_topic, PointCloud2)

    # Publish output topic
    tf_pub = tf2_ros.StaticTransformBroadcaster()

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer(
        [image_sub, camera_info_sub, pointcloud_sub],
        queue_size=5, slop=0.1)
    ats.registerCallback(callback, tf_pub)

    # Keep python from exiting until this node is stopped
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')


if __name__ == '__main__':
    rospy.init_node('calibrator', anonymous=True)

    # ! for calibration mode, don't forget to change 3 topics
    rospy.set_param("~test", "test")
    image_topic = '/camera_0/image' if not rospy.has_param(
        "~image_topic") else rospy.get_param("~image_topic")
    camera_info_topic = '/camera_0/camera_info' if not rospy.has_param(
        "~camera_info_topic") else rospy.get_param("~camera_info_topic")
    pointcloud_topic = '/lidar_0/lidar_node/pointcloud' if not rospy.has_param(
        "~pointcloud_topic") else rospy.get_param("~pointcloud_topic")

    # Start keyboard handler thread
    start_keyboard_handler()

    # Start subscriber
    listener(image_topic, camera_info_topic, pointcloud_topic)
