#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import yaml
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
from geometry_msgs.msg import PoseStamped,PoseWithCovarianceStamped, TransformStamped
from graphviz import Graph
from copy import copy, deepcopy
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist,PointStamped
import math
import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener
from scipy.ndimage import convolve
import heapq

# Import other python packages that you think necessary


class Task1(Node):
    """
    Environment mapping task.
    """
    def __init__(self):
        super().__init__('task1_node')
        self.time_period = 0.1
    # Define function(s) that complete the (automatic) mapping task
        self.timer_path = self.create_timer(self.time_period,self.path_timer_callback)
        self.timer_cmd_vel = self.create_timer(self.time_period,self.cmd_vel_timer_callback)
        self.timer_path_follow = self.create_timer(self.time_period,self.path_follow_callback)
        self.timer_transform = self.create_timer(self.time_period, self.get_pose)
        self.timer_rotate = self.create_timer(self.time_period, self.rotate_tb3)
        self.map_subscriber = self.create_subscription(OccupancyGrid,'map',self.map_callback,10)
        self.map_data = OccupancyGrid()
        self.map_data_2d = [[]]

        self.odom = Odometry()
        self.path = Path()
        self.twist = Twist()
        self.scan_data = LaserScan()
        self.pixel_world_coversion = 0.049

        # Transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.path_publisher = self.create_publisher(Path,'path',10)
        self.publisher_cmd_vel = self.create_publisher(Twist,'cmd_vel',10)
        self.subscription_scan = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )
        self.prev_error1 = 0.0
        self.integral_error1 = 0.0
        self.kp1 = 1.8
        self.kd1 = 0.05
        self.ki1 = 0.01
        self.max_velocity = 0.1

        self.point_index = 0

        #pid controller 2
        self.prev_error2 = 0.0
        self.integral_error2 = 0.0
        self.kp2 = 0.4
        self.kd2 = 0.04
        self.ki2 = 0.004
        self.max_velocity_angular = 0.5
        self.transform_confirmed = False

        self.rotate = True
        self.rotate_start_time = None
        self.kernel = 17
        self.collision_check = 1.0
        self.previous_path = []
        self.back_track = False
    
    def scan_callback(self,msg):
        self.scan_data = msg
        if self.scan_data.ranges[0] <= 0.3 and self.back_track == False:
            self.back_track = True
            self.point_index = len(self.previous_path.poses) - self.point_index - 1
            self.previous_path.poses.reverse()
            self.path = deepcopy(self.previous_path)
            print(min(self.scan_data.ranges))

    def rotate_tb3(self):
        if self.rotate:
           if self.rotate_start_time is None:
              self.rotate_start_time = self.get_clock().now()
           self.twist.angular.z = 0.5
           elapsed_time = (self.get_clock().now() - self.rotate_start_time).nanoseconds * 1e-9
           target_duration = 2*math.pi / abs(self.twist.angular.z)
           if elapsed_time >= target_duration:
               self.twist.angular.z = 0.0
               self.rotate = False
               self.rotate_start_time = None
        

    def get_pose(self):
            try:
                # Get the transform from 'map' to 'base_link'
                transform = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
                self.odom.pose.pose.position.x = transform.transform.translation.x
                self.odom.pose.pose.position.y = transform.transform.translation.y
                self.odom.pose.pose.position.z = transform.transform.translation.z
                self.odom.pose.pose.orientation.x = transform.transform.rotation.x
                self.odom.pose.pose.orientation.y = transform.transform.rotation.y
                self.odom.pose.pose.orientation.z = transform.transform.rotation.z
                self.odom.pose.pose.orientation.w = transform.transform.rotation.w    
                self.transform_confirmed = True
            except Exception as e:
                print('not recieved transform')
                self.transform_confirmed = False
    
    def cmd_vel_timer_callback(self):
        self.publisher_cmd_vel.publish(self.twist)
    def path_timer_callback(self): 
        if len(self.path.poses)!=0:
            self.path_publisher.publish(self.path)
    
    def odom_callback(self,msg):
        self.odom = msg

    def world_to_pixel_conversion(self,width,height,map_origin_x,map_origin_y,val,val1):
        x_limit_end = width*self.pixel_world_coversion + map_origin_x
        y_limit_end = height*self.pixel_world_coversion + map_origin_y
        t = (width*(val - map_origin_x)/(x_limit_end - map_origin_x))
        y = -height*(val1 - map_origin_y)/(y_limit_end - map_origin_y) + height
        return int(t),int(y)


    def pixel_to_world_conversion(self,width,height,map_origin_x,map_origin_y,val,val1):
        x_limit_end = width*self.pixel_world_coversion + map_origin_x
        y_limit_end = height*self.pixel_world_coversion + map_origin_y
        t = ((x_limit_end - map_origin_x)*(val/width)) + map_origin_x
        y = ((y_limit_end - map_origin_y)*(val1 - (height))/(-height)) + map_origin_y
        return t,y

    
    def map_callback(self,msg):
        self.map_data = msg
        self.map_data_2d = np.reshape(self.map_data.data, (self.map_data.info.height, self.map_data.info.width))
        self.map_data_2d = np.flipud(self.map_data_2d)
        self.map_data_2d = MapProcessor(self.map_data_2d,'map_data')
        kr = self.map_data_2d.rect_kernel(self.kernel ,0.58823529)
        self.map_data_2d.inflate_map(kr,True)
        self.map_data_2d.get_graph_from_map()
        if len(self.path.poses) == 0 and self.rotate == False and self.transform_confirmed:
            max_neighbour_distance = {}
            start_point_x =  self.odom.pose.pose.position.x
            start_point_y =  self.odom.pose.pose.position.y
            print(self.map_data_2d.map.image_array.shape)
            start_point_x,start_point_y = self.world_to_pixel_conversion(self.map_data_2d.map.image_array.shape[1],self.map_data_2d.map.image_array.shape[0],self.map_data.info.origin.position.x,self.map_data.info.origin.position.y,start_point_x,start_point_y)
            print(start_point_x,start_point_y)
            for i in range(self.map_data_2d.map.image_array.shape[0]):
                for j in range(self.map_data_2d.map.image_array.shape[1]):
                    if self.map_data_2d.inf_map_img_array[i][j] == 255:
                        if '%d,%d'%(i,j) not in max_neighbour_distance:
                            max_neighbour_distance['%d,%d'%(i,j)] = [[],np.inf,[],[]]
                        pixel_dist = math.sqrt((i - start_point_y)**2 + (j - start_point_x)**2)
                        if (i > 0):
                            if self.map_data_2d.inf_map_img_array[i-1][j] == 0:
                                
                                max_neighbour_distance['%d,%d'%(i,j)][0].append('%d,%d'%(i-1,j))
                                max_neighbour_distance['%d,%d'%(i,j)][1] = pixel_dist
                            elif self.map_data_2d.inf_map_img_array[i-1][j] == 255:
                                max_neighbour_distance['%d,%d'%(i,j)][2].append('%d,%d'%(i-1,j))
                            else:
                                max_neighbour_distance['%d,%d'%(i,j)][3].append('%d,%d'%(i-1,j))

                                # add an edge up
                                
                        if (i < (self.map_data_2d.map.image_array.shape[0] - 1)):
                            
                            if self.map_data_2d.inf_map_img_array[i+1][j] == 0:
                                

                                # add an edge down
                                max_neighbour_distance['%d,%d'%(i,j)][0].append('%d,%d'%(i+1,j))
                                max_neighbour_distance['%d,%d'%(i,j)][1] = pixel_dist
                            elif self.map_data_2d.inf_map_img_array[i+1][j] == 255:
                                max_neighbour_distance['%d,%d'%(i,j)][2].append('%d,%d'%(i+1,j))
                            else:
                                max_neighbour_distance['%d,%d'%(i,j)][3].append('%d,%d'%(i+1,j))

                        if (j > 0):
                            if self.map_data_2d.inf_map_img_array[i][j-1] == 0:
                                # add an edge to the left
                                
                                max_neighbour_distance['%d,%d'%(i,j)][0].append('%d,%d'%(i,j-1))
                                max_neighbour_distance['%d,%d'%(i,j)][1] = pixel_dist
                            elif self.map_data_2d.inf_map_img_array[i][j-1] == 255:
                                max_neighbour_distance['%d,%d'%(i,j)][2].append('%d,%d'%(i,j-1))
                            else:
                                max_neighbour_distance['%d,%d'%(i,j)][3].append('%d,%d'%(i,j-1))

                        if (j < (self.map_data_2d.map.image_array.shape[1] - 1)):
                            
                            if self.map_data_2d.inf_map_img_array[i][j+1] == 0:
                                # add an edge to the right
                                
                                max_neighbour_distance['%d,%d'%(i,j)][0].append('%d,%d'%(i,j+1))
                                max_neighbour_distance['%d,%d'%(i,j)][1] = pixel_dist
                            elif self.map_data_2d.inf_map_img_array[i][j+1] == 255:
                                max_neighbour_distance['%d,%d'%(i,j)][2].append('%d,%d'%(i,j+1))
                            else:
                                max_neighbour_distance['%d,%d'%(i,j)][3].append('%d,%d'%(i,j+1))                            
                        if ((i > 0) and (j > 0)):
                            
                            if self.map_data_2d.inf_map_img_array[i-1][j-1] == 0:
                                # add an edge up-left
                                
                                max_neighbour_distance['%d,%d'%(i,j)][0].append('%d,%d'%(i-1,j-1))
                                max_neighbour_distance['%d,%d'%(i,j)][1] = pixel_dist
                            elif self.map_data_2d.inf_map_img_array[i-1][j-1] == 255:
                                max_neighbour_distance['%d,%d'%(i,j)][2].append('%d,%d'%(i-1,j-1))
                            else:
                                max_neighbour_distance['%d,%d'%(i,j)][3].append('%d,%d'%(i-1,j-1))
                        if ((i > 0) and (j < (self.map_data_2d.map.image_array.shape[1] - 1))):
                            
                            if self.map_data_2d.inf_map_img_array[i-1][j+1] == 0:
                                # add an edge up-right
                                
                                max_neighbour_distance['%d,%d'%(i,j)][0].append('%d,%d'%(i-1,j+1))
                                max_neighbour_distance['%d,%d'%(i,j)][1] = pixel_dist
                            elif self.map_data_2d.inf_map_img_array[i-1][j+1] == 255:
                                max_neighbour_distance['%d,%d'%(i,j)][2].append('%d,%d'%(i-1,j+1))
                            else:
                                max_neighbour_distance['%d,%d'%(i,j)][3].append('%d,%d'%(i-1,j+1))
                        if ((i < (self.map_data_2d.map.image_array.shape[0] - 1)) and (j > 0)):
                            
                            if self.map_data_2d.inf_map_img_array[i+1][j-1] == 0:
                                # add an edge down-left
                                
                                max_neighbour_distance['%d,%d'%(i,j)][0].append('%d,%d'%(i+1,j-1))
                                max_neighbour_distance['%d,%d'%(i,j)][1] = pixel_dist
                            elif self.map_data_2d.inf_map_img_array[i+1][j-1] == 255:
                                max_neighbour_distance['%d,%d'%(i,j)][2].append('%d,%d'%(i+1,j-1))
                            else:
                                max_neighbour_distance['%d,%d'%(i,j)][3].append('%d,%d'%(i+1,j-1))
                        if ((i < (self.map_data_2d.map.image_array.shape[0] - 1)) and (j < (self.map_data_2d.map.image_array.shape[1] - 1))):
                            if self.map_data_2d.inf_map_img_array[i+1][j+1] == 0:
                                

                                # add an edge down-right
                                max_neighbour_distance['%d,%d'%(i,j)][0].append('%d,%d'%(i+1,j+1))
                                max_neighbour_distance['%d,%d'%(i,j)][1] = pixel_dist  
                            elif self.map_data_2d.inf_map_img_array[i+1][j+1] == 255:
                                max_neighbour_distance['%d,%d'%(i,j)][2].append('%d,%d'%(i+1,j+1))
                            else:
                                max_neighbour_distance['%d,%d'%(i,j)][3].append('%d,%d'%(i+1,j+1))
                        if len(max_neighbour_distance['%d,%d'%(i,j)][0]) == 0 or max_neighbour_distance['%d,%d'%(i,j)][1] == np.inf:
                            del max_neighbour_distance['%d,%d'%(i,j)]
            max_neighbour_distance_backup = max_neighbour_distance.copy()
            result_key = ''
            max_neighbour_distance = dict(sorted(max_neighbour_distance.items(), key=lambda item: abs(len(item[1][3]))))
            max_obs = len(list(max_neighbour_distance.items())[0][1][3])
            del_key = []
            for key, value in max_neighbour_distance.items():
                if len(max_neighbour_distance[key][3])!=max_obs:
                    del_key.append(key)
            for i in del_key:
                del max_neighbour_distance[i]
            
            max_neighbour_distance = dict(sorted(max_neighbour_distance.items(), key=lambda item: abs(len(item[1][0])-len(item[1][2]))))
            min_len = list(max_neighbour_distance.items())
            min_len = abs(len(min_len[0][1][0]) - len(min_len[0][1][2]))

            del_key = []
            for key,value in max_neighbour_distance.items():
                if abs(len(max_neighbour_distance[key][0]) - len(max_neighbour_distance[key][2]))!=min_len:
                    del_key.append(key)
            for i in del_key:
                del max_neighbour_distance[i]
            
            
            max_neighbour_distance = dict(sorted(max_neighbour_distance.items(), key=lambda item: abs(item[1][1])))
            items = list(max_neighbour_distance.items())
            result_key = items[len(items)//2][1][0][0]
            print(self.map_data_2d.inf_map_img_array[start_point_y][start_point_x])
            
            if(len(max_neighbour_distance)!=0 and self.map_data_2d.inf_map_img_array[start_point_y][start_point_x]==0):
                self.kernel = 15
                while len(max_neighbour_distance)>0:    
                    try:       
                            self.point_index = 0
                            result_key = items[len(items)//2][1][0][0]
                            print('median') 
                            self.map_data_2d.map_graph.root = str(start_point_y)+','+str(start_point_x)
                            self.map_data_2d.map_graph.end = result_key
                            self.as_maze = BFS(self.map_data_2d.map_graph)
                            self.as_maze.solve(self.map_data_2d.map_graph.g[self.map_data_2d.map_graph.root],self.map_data_2d.map_graph.g[self.map_data_2d.map_graph.end])
                            path_as,dist_as = self.as_maze.reconstruct_path(self.map_data_2d.map_graph.g[self.map_data_2d.map_graph.root],self.map_data_2d.map_graph.g[self.map_data_2d.map_graph.end])
                            pix_path = self.map_data_2d.draw_path(path_as)
                            if len(path_as) == 0:
                                raise ValueError("rasing error to use minimum value")
                           
                            poses_stamped = []
                            self.path = Path()
                            for i in path_as:
                                pose = PoseStamped()
                                pose.header.stamp = self.get_clock().now().to_msg()
                                pose.header.frame_id = 'map'
                                x = i.split(',')
                                pose.pose.position.x,pose.pose.position.y = self.pixel_to_world_conversion(self.map_data_2d.map.image_array.shape[1],self.map_data_2d.map.image_array.shape[0],self.map_data.info.origin.position.x,self.map_data.info.origin.position.y,float(x[1]),float(x[0]))
                                poses_stamped.append(pose)
                            print("Path Planned -- Starting execusion")
                            self.path.header.stamp =  self.get_clock().now().to_msg()
                            self.path.header.frame_id = 'map'
                            self.path.poses = poses_stamped
                            # print(self.path.poses)
                            self.previous_path = deepcopy(self.path)
                            if len(path_as) !=0:
                            #    self.previous_path = self.path
                               break

                    except:
                            result_key = items[0][1][0][0]    
                            print('minimum')
                            print(result_key)
                            self.map_data_2d.map_graph.root = str(start_point_y)+','+str(start_point_x)
                            self.map_data_2d.map_graph.end = result_key
                            self.as_maze = BFS(self.map_data_2d.map_graph)
                            self.as_maze.solve(self.map_data_2d.map_graph.g[self.map_data_2d.map_graph.root],self.map_data_2d.map_graph.g[self.map_data_2d.map_graph.end])
                            path_as,dist_as = self.as_maze.reconstruct_path(self.map_data_2d.map_graph.g[self.map_data_2d.map_graph.root],self.map_data_2d.map_graph.g[self.map_data_2d.map_graph.end])
                            pix_path = self.map_data_2d.draw_path(path_as)
                            
                            poses_stamped = []
                            self.path = Path()
                            for i in path_as:
                                pose = PoseStamped()
                                pose.header.stamp = self.get_clock().now().to_msg()
                                pose.header.frame_id = 'map'
                                x = i.split(',')
                                pose.pose.position.x,pose.pose.position.y = self.pixel_to_world_conversion(self.map_data_2d.map.image_array.shape[1],self.map_data_2d.map.image_array.shape[0],self.map_data.info.origin.position.x,self.map_data.info.origin.position.y,float(x[1]),float(x[0]))
                                poses_stamped.append(pose)
                            
                            print("Path Planned -- Starting execusion")
                            self.path.header.stamp =  self.get_clock().now().to_msg()
                            self.path.header.frame_id = 'map'
                            self.path.poses = poses_stamped
                            # print(self.path.poses)
                            self.previous_path = deepcopy(self.path)
                            if len(path_as) != 0:
                                # self.previous_path = self.path
                                break
                            max_neighbour_distance_backup = dict(sorted(max_neighbour_distance_backup.items(), key=lambda item: abs(item[1][1])))
                            items = list(max_neighbour_distance_backup.items())
                            # result_key = items[len(items)//2][1][0][0]
                            del max_neighbour_distance_backup[list(max_neighbour_distance_backup.items())[0][0]]
                            # print(max_neighbour_distance_backup)


                            
            else:
                # self.rotate = True
                self.back_track = True
                self.previous_path.poses.reverse()
                self.point_index = len(self.previous_path.poses) - self.point_index - 1
                self.path = deepcopy(self.previous_path)
                print('Starting Recovery')
                
                # if self.kernel !=5:
                #    self.kernel -=1
    
    def pid_controller_angular(self,error):
        op = self.kp2*error + (self.kd2 * (error - self.prev_error1)/0.1) + self.ki2 * self.integral_error1
        self.prev_error1 = error
        self.integral_error1+=(error*0.1)
        if op > self.max_velocity_angular:
            op = self.max_velocity_angular
        elif op < -self.max_velocity_angular:
            op = -self.max_velocity_angular
        self.twist = Twist()
        self.twist.angular.z = float(op)

    

    def pid_controller_linear(self,error):
        op = self.kp1*error + (self.kd1 * (error - self.prev_error2)/0.1) + self.ki1 * self.integral_error2 
        self.prev_error2 = error
        self.integral_error2+=(error*0.1)
        if op <= -self.max_velocity:
            op = -self.max_velocity_angular
        elif op >= self.max_velocity:
            op = self.max_velocity
        # print(op)
        self.twist = Twist()
        self.twist.linear.x = float(op)

    def quaternion_to_euler(self,qx, qy, qz, qw):
    # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def path_follow_callback(self):
        if (len(self.path.poses)!=0 and self.rotate == False):
            
            l = 1
            
            check = 0


            stop_condition = True

            if self.back_track == True:
                print('point index',self.point_index)
                x,y = self.world_to_pixel_conversion(self.map_data_2d.map.image_array.shape[1],self.map_data_2d.map.image_array.shape[0],self.map_data.info.origin.position.x,self.map_data.info.origin.position.y,self.odom.pose.pose.position.x,self.odom.pose.pose.position.y)
                stop_condition = (self.map_data_2d.inf_map_img_array[y][x] == 100 or self.map_data_2d.inf_map_img_array[y][x] == 255) or (self.scan_data.ranges[0]) <= 0.3
                print(self.map_data_2d.inf_map_img_array[y][x])
                check = 1
                print('stop in back_track',stop_condition)
                if not(stop_condition):
                    self.back_track = False
            
            print(self.back_track)


            if self.point_index < len(self.path.poses)-l:
                if stop_condition or check == 0: 

                
                    linear_error = math.sqrt((self.odom.pose.pose.position.x  - self.path.poses[self.point_index+1].pose.position.x)**2 +(self.odom.pose.pose.position.y - self.path.poses[self.point_index+1].pose.position.y)**2) 

                        # Convert quaternion to Euler angles
                    roll, pitch, yaw = self.quaternion_to_euler(self.odom.pose.pose.orientation.x,self.odom.pose.pose.orientation.y,self.odom.pose.pose.orientation.z,self.odom.pose.pose.orientation.w)
                    orientation_target = (math.atan2(self.path.poses[self.point_index+1].pose.position.y - self.path.poses[self.point_index].pose.position.y ,self.path.poses[self.point_index+1].pose.position.x - self.path.poses[self.point_index].pose.position.x ))
                    orientation_error = orientation_target - yaw
                    if orientation_error > math.pi:
                        orientation_error -= 2*math.pi
                    elif orientation_error < -math.pi:
                        orientation_error += 2*math.pi
                    self.pid_controller_angular(orientation_error)
                    if abs(orientation_error) < 0.02:
                        self.twist = Twist()
                        self.twist.angular.z = 0.0

                            # self.point_index +=1
                        self.pid_controller_linear(linear_error)
                        if abs(linear_error) < 0.07 and abs(orientation_error) < 0.02:
                            self.twist = Twist()
                            self.twist.linear.x = 0.0
                            self.twist.angular.z = 0.0
                            self.prev_error1 = 0.0
                            self.integral_error1 = 0.0
                            self.prev_error2 = 0.0
                            self.integral_error2 = 0.0
                            self.point_index +=1
                                # break
                else:
                    self.twist = Twist()
                    print('Back Track complete')
                    check = 0
                    self.prev_error1 = 0.0
                    self.integral_error1 = 0.0
                    self.prev_error2 = 0.0
                    self.integral_error2 = 0.0
                    self.path = Path()

            else:
                self.twist = Twist()
                print('Goal Reached')
                self.path = Path()
                # self.point_index = 0
                self.rotate = True
                check = 0
                self.prev_error1 = 0.0
                self.integral_error1 = 0.0
                self.prev_error2 = 0.0
                self.integral_error2 = 0.0


        # print(self.map_data_2d)


class Map():
    def __init__(self, map_name):
        self.map_im = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map()
    def __open_map(self,map_name):
        im = Image.fromarray(map_name)

        return im

    def __get_obstacle_map(self):
        img_array = np.reshape(list(self.map_im.getdata()),(self.map_im.size[1],self.map_im.size[0]))
        
        return img_array
    

class Queue():
    def __init__(self, init_queue = []):
        self.queue = copy(init_queue)
        self.start = 0
        self.end = len(self.queue)-1

    def __len__(self):
        numel = len(self.queue)
        return numel

    def __repr__(self):
        q = self.queue
        tmpstr = ""
        for i in range(len(self.queue)):
            flag = False
            if(i == self.start):
                tmpstr += "<"
                flag = True
            if(i == self.end):
                tmpstr += ">"
                flag = True

            if(flag):
                tmpstr += '| ' + str(q[i]) + '|\n'
            else:
                tmpstr += ' | ' + str(q[i]) + '|\n'

        return tmpstr

    def __call__(self):
        return self.queue

    def initialize_queue(self,init_queue = []):
        self.queue = copy(init_queue)

    def sort(self,key=str.lower):
        self.queue = sorted(self.queue,key=key)

    def push(self,data):
        self.queue.append(data)
        self.end += 1

    def pop(self):
        p = self.queue.pop(self.start)
        self.end = len(self.queue)-1
        return p

class Node():
    def __init__(self,name):
        self.name = name
        self.children = []
        self.weight = []

    def __repr__(self):
        return self.name

    def add_children(self,node,w=None):
        if w == None:
            w = [1]*len(node)
        self.children.extend(node)
        self.weight.extend(w)




class Tree():
    def __init__(self,name):
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}
        self.g_visual = Graph('G')

    def __call__(self):
        for name,node in self.g.items():
            if(self.root == name):
                self.g_visual.node(name,name,color='red')
            elif(self.end == name):
                self.g_visual.node(name,name,color='blue')
            else:
                self.g_visual.node(name,name)
            for i in range(len(node.children)):
                c = node.children[i]
                w = node.weight[i]
                #print('%s -> %s'%(name,c.name))
                if w == 0:
                    self.g_visual.edge(name,c.name)
                else:
                    self.g_visual.edge(name,c.name,label=str(w))
        return self.g_visual

    def add_node(self, node, start = False, end = False):
        self.g[node.name] = node
        if(start):
            self.root = node.name
        elif(end):
            self.end = node.name

    def set_as_root(self,node):
        # These are exclusive conditions
        self.root = True
        self.end = False

    def set_as_end(self,node):
        # These are exclusive conditions
        self.root = False
        self.end = True
class Tree():
    def __init__(self,name):
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}
        self.g_visual = Graph('G')

    def __call__(self):
        for name,node in self.g.items():
            if(self.root == name):
                self.g_visual.node(name,name,color='red')
            elif(self.end == name):
                self.g_visual.node(name,name,color='blue')
            else:
                self.g_visual.node(name,name)
            for i in range(len(node.children)):
                c = node.children[i]
                w = node.weight[i]
                #print('%s -> %s'%(name,c.name))
                if w == 0:
                    self.g_visual.edge(name,c.name)
                else:
                    self.g_visual.edge(name,c.name,label=str(w))
        return self.g_visual

    def add_node(self, node, start = False, end = False):
        self.g[node.name] = node
        if(start):
            self.root = node.name
        elif(end):
            self.end = node.name

    def set_as_root(self,node):
        # These are exclusive conditions
        self.root = True
        self.end = False

    def set_as_end(self,node):
        # These are exclusive conditions
        self.root = False
        self.end = True



class MapProcessor():
    def __init__(self,name,data):
        self.map = Map(name)
        self.inf_map_img_array = deepcopy(self.map.image_array)
        self.map_graph = Tree(data)

    def __modify_map_pixel(self,map_array,i,j,value,absolute):
        if( (i >= 0) and
            (i < map_array.shape[0]) and
            (j >= 0) and
            (j < map_array.shape[1]) ):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] += value

    def __inflate_obstacle(self,kernel,map_array,i,j,absolute):
        dx = int(kernel.shape[0]//2)
        dy = int(kernel.shape[1]//2)
        if (dx == 0) and (dy == 0):
            self.__modify_map_pixel(map_array,i,j,kernel[0][0],absolute)
        else:
            for k in range(i-dx,i+dx):
                for l in range(j-dy,j+dy):
                    if( (k >= 0) and
                        (k < self.map.image_array.shape[0]) and
                        (l >= 0) and
                        (l < self.map.image_array.shape[1]) ):
                        self.inf_map_img_array[k][l] = 100


    def inflate_map(self,kernel,absolute=True):
        c = 0
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.map.image_array[i][j] == 100:
                    c+=1
                    self.__inflate_obstacle(kernel,self.inf_map_img_array,i,j,absolute)
                
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array)
        if r == 0:
            r = 1

    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = Node('%d,%d'%(i,j))
                    self.map_graph.add_node(node)
        # Connect the nodes through edges
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    if (i > 0):
                        if self.inf_map_img_array[i-1][j] == 0:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d'%(i-1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up],[1])
                    if (i < (self.map.image_array.shape[0] - 1)):
                        if self.inf_map_img_array[i+1][j] == 0:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d'%(i+1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw],[1])
                    if (j > 0):
                        if self.inf_map_img_array[i][j-1] == 0:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d'%(i,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_lf],[1])
                    if (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i][j+1] == 0:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d'%(i,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_rg],[1])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i-1][j-1] == 0:
                            # add an edge up-left
                            child_up_lf = self.map_graph.g['%d,%d'%(i-1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_lf],[np.sqrt(2)])
                    if ((i > 0) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i-1][j+1] == 0:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d'%(i-1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_rg],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i+1][j-1] == 0:
                            # add an edge down-left
                            child_dw_lf = self.map_graph.g['%d,%d'%(i+1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_lf],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i+1][j+1] == 0:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g['%d,%d'%(i+1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_rg],[np.sqrt(2)])

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        r = np.max(g)-np.min(g)
        sm = (g - np.min(g))*1/r
        return sm

    def rect_kernel(self, size, value):
        m = np.ones(shape=(size,size))
        return 100*m

    def draw_path(self,path):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        for idx in path:
            tup = tuple(map(int, idx.split(',')))
            path_tuple_list.append(tup)
            path_array[tup] = 0.5
        return path_array



class AStar():
    def __init__(self,in_tree):
        self.in_tree = in_tree
        self.q = Queue()
        self.dist = {name:np.Inf for name,node in in_tree.g.items()}
        self.h = {name:0 for name,node in in_tree.g.items()}

        for name,node in in_tree.g.items():
            start = tuple(map(int, name.split(',')))
            end = tuple(map(int, self.in_tree.end.split(',')))
            self.h[name] = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)

        self.via = {name:0 for name,node in in_tree.g.items()}
        for __,node in in_tree.g.items():
            self.q.push(node)

    def __get_f_score(self,node):
      return self.dist[node.name] + self.h[node.name]

    def solve(self, sn, en):
        self.dist[sn.name] = 0
        while len(self.q) > 0:
          self.q.sort(key=self.__get_f_score)
          current_node = self.q.queue[self.q.start]
          for i in current_node.children:
            if self.dist[i.name] > self.dist[current_node.name] + current_node.weight[current_node.children.index(i)]:
              self.dist[i.name] = self.dist[current_node.name] + current_node.weight[current_node.children.index(i)]
              self.via[i.name] = current_node.name
          popped_node = self.q.pop()
          if popped_node.name == en.name:
            print('done')
            break





        # Place code here (remove the pass
        # statement once you start coding)

    def reconstruct_path(self,sn,en):
        path = []
        dist = self.dist[en.name]
        path.append(en.name)
        current_node = en.name
        while True:
          path.append(self.via[current_node])
          print(current_node)
          current_node = self.via[current_node]
          if current_node == sn.name:
              path.append(sn.name)
              break
        path.reverse()
        # Place code here
        return path,dist
class Dijkstra():
    def __init__(self,in_tree):
        self.q = Queue()
        self.dist = {name:np.inf for name,node in in_tree.g.items()}
        self.via = {name:0 for name,node in in_tree.g.items()}
        self.visited = {name:False for name,node in in_tree.g.items()}
        for __,node in in_tree.g.items():
            self.q.push(node)

    def __get_dist_to_node(self,node):
        return self.dist[node.name]

    def solve(self, sn, en):
        self.dist[sn.name] = 0
        while len(self.q) > 0:
            self.q.sort(key=self.__get_dist_to_node)
            u = self.q.pop()
            #print(u.name,self.q.queue)
            if u.name == en.name:
                break
            for i in range(len(u.children)):
                c = u.children[i]
                w = u.weight[i]
                new_dist = self.dist[u.name] + w
                if new_dist < self.dist[c.name]:
                    self.dist[c.name] = new_dist
                    self.via[c.name] = u.name


    def reconstruct_path(self,sn,en):
        start_key = sn.name
        end_key = en.name
        dist = self.dist[end_key]
        u = end_key
        path = [u]
        while u != start_key:
            u = self.via[u]
            path.append(u)
        path.reverse()
        return path,dist
class BFS():
    def __init__(self,tree):
        self.q = Queue()
        self.visited = {name:False for name,node in tree.g.items()}
        self.via = {name:0 for name,node in tree.g.items()}
        self.dist = {name:0 for name,node in tree.g.items()}

    def solve(self,sn,en):
        self.q.push(sn)
        self.visited[sn.name] = True
        while len(self.q) > 0:
            node = self.q.pop()
            for i in range(len(node.children)):
                c = node.children[i]
                w = node.weight[i]
                if self.visited[c.name] == False:
                    self.q.push(c)
                    self.visited[c.name] = True
                    self.via[c.name] = node.name
                    self.dist[c.name] = self.dist[node.name] + w
            #print(node.name,self.q.queue)
            #print(self.dist)
        return self.via

    def reconstruct_path(self,sn=0,en=0):
        path = []
        node = en.name
        path.append(node)
        dist = self.dist[en.name]
        while True:
            node = self.via[node]
            if node == 0:
                break
            else:
                path.append(node)
        path.reverse()
        if path[0] != sn.name:
            path = []
        return path,dist

def main(args=None):
    rclpy.init(args=args)

    task1 = Task1()

    try:
        rclpy.spin(task1)
    except KeyboardInterrupt:
        pass
    finally:
        task1.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()