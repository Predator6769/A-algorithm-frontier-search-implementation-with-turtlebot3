#!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
import rclpy
from rclpy.node import Node
import yaml
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
from geometry_msgs.msg import PoseStamped,TransformStamped,PoseWithCovarianceStamped
from graphviz import Graph
from copy import copy, deepcopy
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math
from tf2_ros import Buffer, TransformListener
import heapq
# Import other python packages that you think necessary


class Task2(Node):
    """
    Environment localization and navigation task.
    """
    def __init__(self):
        super().__init__('task2_node')
        timer_period = 0.1  # seconds

        #timers
        self.timer_path_follow = self.create_timer(timer_period,self.path_follow_callback)
        self.timer_cmd_vel = self.create_timer(timer_period,self.cmd_vel_timer_callback)
        self.timer_path = self.create_timer(timer_period,self.path_timer_callback)
        # self.timer_transform = self.create_timer(timer_period, self.get_pose)


        # publishers
        self.publisher_cmd_vel = self.create_publisher(Twist,'cmd_vel',10)
        self.path_publisher = self.create_publisher(Path,'path',10)


        # subscribers
        self.subscription = self.create_subscription(
            PoseStamped,
            'move_base_simple/goal',
            self.goal_pose_callback,
            10
        )

        self.subscription_init_pose = self.create_subscription(
            PoseWithCovarianceStamped,
            'amcl_pose',
            self.odom_callback,
            10
        )

        # Transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Messages
        self.twist = Twist()
        self.odom = Odometry()
        self.path = Path()
        self.transform = TransformStamped()

        # pid controller 1
        self.prev_error1 = 0.0
        self.integral_error1 = 0.0
        self.kp1 = 0.8
        self.kd1 = 0.05
        self.ki1 = 0.01
        self.max_velocity = 0.5

        self.point_index = 0

        #pid controller 2
        self.prev_error2 = 0.0
        self.integral_error2 = 0.0
        self.kp2 = 0.4
        self.kd2 = 0.04
        self.ki2 = 0.004
        self.max_velocity_angular = 0.1
     
        # Odom to map transform
        self.odom_to_map_transform =[0,0,0,0,0]

        self.robot_orientation = 0.0

        # pixels to map frame conversion
        self.pixel_world_coversion = 0.049

        # Loading map and graph
        self.map_graph = MapProcessor('/home/prashanth/sim_ws_fp/src/sim_ws/src/turtlebot3_gazebo/maps/map')
        kr = self.map_graph.rect_kernel(11,1)
        self.map_graph.inflate_map(kr,True)
        # self.map_graph.get_graph_from_map()

        # Path index
        self.point_index = 0


        self.subscription  # prevent unused variable warning

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
            except Exception as e:
                pass
    def cmd_vel_timer_callback(self):
        self.publisher_cmd_vel.publish(self.twist)

    def euler_to_quaternion(self,roll, pitch, yaw):
        """
        Convert Euler angles to a quaternion.
        
        Parameters:
            roll (float): Rotation around the x-axis in radians.
            pitch (float): Rotation around the y-axis in radians.
            yaw (float): Rotation around the z-axis in radians.
            
        Returns:
            tuple: A tuple representing the quaternion (x, y, z, w).
        """
        # Calculate half angles
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        # Compute quaternion components
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return (x, y, z, w)
    def quaternion_multiply(self,q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)

    def quaternion_rotate(self,q, v):
        """Rotate vector v by quaternion q."""
        q_conjugate = (-q[1], -q[2], -q[3], q[0])
        v_as_quaternion = (0, v[0], v[1], v[2])
            
        return self.quaternion_multiply(self.quaternion_multiply(q, v_as_quaternion), q_conjugate)[1:]

        

    def odom_callback(self,msg):
        self.odom = msg

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
        print(op)
        self.twist = Twist()
        self.twist.linear.x = float(op)



    def path_timer_callback(self):
        if len(self.path.poses)!=0:
            self.path_publisher.publish(self.path)


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


    def path_follow_callback(self):
        if len(self.path.poses)!=0:
            if self.point_index < len(self.path.poses) - 1:
                
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
                print(yaw, orientation_target,linear_error)
                # self.point_index +=1
                if abs(orientation_error) < 0.02:
                    self.twist = Twist()
                    self.twist.angular.z = 0.0

                    # self.point_index +=1
                    self.pid_controller_linear(linear_error)
                    if abs(linear_error) < 0.07 and abs(orientation_error) < 0.05:
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
                print('Goal Reached')
                self.path = Path()
                self.point_index = 0
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.0
                self.prev_error1 = 0.0
                self.integral_error1 = 0.0
                self.prev_error2 = 0.0
                self.integral_error2 = 0.0
                   
                   
            

    def goal_pose_callback(self,msg):
        self.path = Path()
        print('Calculating Path')
        print(msg.pose.position.x,msg.pose.position.y)
        # print(msg.point.x,msg.point.y)
        start_point_x =  self.odom.pose.pose.position.x
        start_point_y =  self.odom.pose.pose.position.y
        print(start_point_x,start_point_y)
        print(self.map_graph.map.image_array.shape[1],self.map_graph.map.image_array.shape[0],self.map_graph.map.map_df.origin[0][0],self.map_graph.map.map_df.origin[0][1],start_point_x,start_point_y)
        start_point_x,start_point_y = self.world_to_pixel_conversion(self.map_graph.map.image_array.shape[1],self.map_graph.map.image_array.shape[0],self.map_graph.map.map_df.origin[0][0],self.map_graph.map.map_df.origin[0][1],start_point_x,start_point_y)
        self.map_graph.map_graph.root = str(start_point_y)+","+str(start_point_x)

        goal_point_x,goal_point_y = self.world_to_pixel_conversion(self.map_graph.map.image_array.shape[1],self.map_graph.map.image_array.shape[0],self.map_graph.map.map_df.origin[0][0],self.map_graph.map.map_df.origin[0][1],msg.pose.position.x,msg.pose.position.y)
        print(goal_point_y,goal_point_x)
        print(self.map_graph.inf_map_img_array[goal_point_y][goal_point_x])
        self.map_graph.map_graph.end = str(goal_point_y)+","+str(goal_point_x)

        self.as_rrt = RRT(self.map_graph, self.map_graph.map_graph.root, self.map_graph.map_graph.end)
        path_as = self.as_rrt.solve()

        poses_stamped = []
        for i in path_as:
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            x = i.split(',')
            pose.pose.position.x,pose.pose.position.y = self.pixel_to_world_conversion(self.map_graph.map.image_array.shape[1],self.map_graph.map.image_array.shape[0],self.map_graph.map.map_df.origin[0][0],self.map_graph.map.map_df.origin[0][1],float(x[1]),float(x[0]))
            poses_stamped.append(pose)
        print("Path Planned -- Starting execusion")
        self.path.header.stamp =  self.get_clock().now().to_msg()
        self.path.header.frame_id = 'map'
        self.path.poses = poses_stamped
        print(len(self.path.poses))
        print(self.path.poses)


# Map load class
class Map():
    def __init__(self, map_name):
        self.map_im, self.map_df, self.limits = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)
    def __open_map(self,map_name):
        f = open(map_name + '.yaml', 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        map_name = map_df.image[0]
        im = Image.open(map_name)
        # size = 200, 200
        # im.thumbnail(size)
        im = ImageOps.grayscale(im)
        # Get the limits of the map. This will help to display the map
        # with the correct axis ticks.
        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]

        return im, map_df, [xmin,xmax,ymin,ymax]

    def __get_obstacle_map(self,map_im, map_df):
        img_array = np.reshape(list(self.map_im.getdata()),(self.map_im.size[1],self.map_im.size[0]))
        up_thresh = self.map_df.occupied_thresh[0]*255
        low_thresh = self.map_df.free_thresh[0]*255
        # img_array = np.fliplr(img_array)
        # img_array = np.flipud(img_array)
        for j in range(self.map_im.size[0]):
            for i in range(self.map_im.size[1]):
                if img_array[i,j] > up_thresh:
                    img_array[i,j] = 255
                else:
                    img_array[i, j] = 0
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




class MapProcessor():
    def __init__(self,name):
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree(name)

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
                    self.__modify_map_pixel(map_array,k,l,kernel[k-i+dx][l-j+dy],absolute)

    def inflate_map(self,kernel,absolute=True):
        # Perform an operation like dilation, such that the small wall found during the mapping process
        # are increased in size, thus forcing a safer path.
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.map.image_array[i][j] == 0:
                    self.__inflate_obstacle(kernel,self.inf_map_img_array,i,j,absolute)
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r

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
        return m

    def draw_path(self,path):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        for idx in path:
            tup = tuple(map(int, idx.split(',')))
            path_tuple_list.append(tup)
            path_array[tup] = 0.5
        return path_array
    
class RRT():
    def __init__(self,in_tree,start_point,end_point):
        self.in_tree = in_tree
        self.parents = {start_point: None}
        self.start = (int(start_point.split(',')[0]),int(start_point.split(',')[1]))
        self.end = (int(end_point.split(',')[0]),int(end_point.split(',')[1]))
        self.d = 5
        self.iterations = 100000
        self.goal_found = 0

    def solve(self):
        new_node = Node('%d,%d'%(self.start[0],self.start[1]))

        self.in_tree.map_graph.add_node(new_node)
        for i in range(self.iterations):
            print(i)
            x, y = self.sample_point()
            self.extend_graph(x,y)
            if self.goal_found == 1:
                break
        print(len(self.in_tree.map_graph.g))
        path = []
        # print(self.parents)
        check = 0
        if self.goal_found:
            end = str(self.end[0]) + ',' + str(self.end[1])
            path.append(end)
            temp = None
            start_node = Node('%d,%d'%(self.start[0],self.start[1]))

            while True:
                check = 0
                if temp == None:
                    nn = Node('%d,%d'%(self.end[0],self.end[1]))
                    nn = nn.name
                else:
                    nn = temp
                for i in self.in_tree.map_graph.g.keys():
                    # print(i, self.in_tree.map_graph.g[i].children,nn.name)
                    for j in self.in_tree.map_graph.g[i].children:
                        if int(nn.split(',')[0]) == int(j.name.split(',')[0]) and int(nn.split(',')[1]) == int(j.name.split(',')[1]):
                            check = 1
                            path.append(i)
                            temp = i
                            break
                    if check:
                        break
                if int(temp.split(',')[0]) == self.start[0] and int(temp.split(',')[1]) == self.start[1]:
                    break
            path.reverse()
        return path
                
                      


    def sample_point(self):
        limx = self.in_tree.inf_map_img_array.shape[0] - 1
        limy = self.in_tree.inf_map_img_array.shape[1] - 1
        b_x = (0,limx)
        b_y  = (0,limy)
        x = np.random.uniform(*b_x)
        y = np.random.uniform(*b_y)
        return x,y
    
    def distance(self,node1,point):
      return math.sqrt((node1[0] - point[0])**2 + (node1[1] - point[1])**2)
    
    def world_to_pixel_conversion_1(self,width,height,map_origin_x,map_origin_y,val,val1,pixel_world_coversion):
        x_limit_end = width*pixel_world_coversion + map_origin_x
        y_limit_end = height*pixel_world_coversion + map_origin_y
        t = (width*(val - map_origin_x)/(x_limit_end - map_origin_x))
        y = -height*(val1 - map_origin_y)/(y_limit_end - map_origin_y) + height
        return int(t),int(y)


    def pixel_to_world_conversion_1(self,width,height,map_origin_x,map_origin_y,val,val1,pixel_world_coversion):
        x_limit_end = width*pixel_world_coversion + map_origin_x
        y_limit_end = height*pixel_world_coversion + map_origin_y
        t = ((x_limit_end - map_origin_x)*(val/width)) + map_origin_x
        y = ((y_limit_end - map_origin_y)*(val1 - (height))/(-height)) + map_origin_y
        return t,y
    
    def extend_graph(self, x, y):
        limx = self.in_tree.inf_map_img_array.shape[0] - 1
        limy = self.in_tree.inf_map_img_array.shape[1] - 1

        nodes = self.in_tree.map_graph.g.keys()
        min_d  = np.inf
        c_node = None
        nodes = [(int(node.split(',')[0]),int(node.split(',')[1])) for node in nodes]
        for node in nodes:
            if self.distance((x,y),node) < min_d:
                min_d = self.distance((x,y),node)
                c_node = node
        dx = x - c_node[0]
        dy = y - c_node[1]

        length = math.hypot(dx, dy)
        if length == 0:
            return
            # raise ValueError("Points are the same. Direction is undefined.")

        # Normalize direction
        dx /= length
        dy /= length

        # Move 'distance' units from p1 in direction of p2
        new_x = c_node[0] + dx * self.d
        new_y = c_node[1] + dy * self.d

        # Round to nearest grid cell
        sampled_point = (int(round(new_x)), int(round(new_y)))

        if sampled_point[0] > limx or sampled_point[0] < 0 or sampled_point[1] < 0 or sampled_point[1] > limy or self.in_tree.inf_map_img_array[sampled_point[0]][sampled_point[1]] != 0:
            return
        
        obs = self.check_obstarcle_btw_points(sampled_point, c_node)
        if obs:
            if self.distance(self.end, sampled_point) < self.d:
                new_node = Node('%d,%d'%(self.end[0],self.end[1]))
                if not(new_node.name in self.in_tree.map_graph.g.keys()):
                    self.in_tree.map_graph.add_node(new_node)
                new_child = self.in_tree.map_graph.g['%d,%d'%(self.end[0],self.end[1])]
                self.in_tree.map_graph.g['%d,%d'%(c_node[0],c_node[1])].add_children([new_child],[self.d])
                self.goal_found = 1
                self.parents['%d,%d'%(c_node[0],c_node[1])] = new_child.name
                print("goal reached")
            else:

                new_node = Node('%d,%d'%(sampled_point[0],sampled_point[1]))
                if not(new_node.name in self.in_tree.map_graph.g.keys()):
                    self.in_tree.map_graph.add_node(new_node)
                new_child = self.in_tree.map_graph.g['%d,%d'%(sampled_point[0],sampled_point[1])]
                self.in_tree.map_graph.g['%d,%d'%(c_node[0],c_node[1])].add_children([new_child],[self.d])
                self.parents['%d,%d'%(c_node[0],c_node[1])] = new_child.name
                


        
    
    def check_obstarcle_btw_points(self, start_point, end_point):
        
        limx = self.in_tree.inf_map_img_array.shape[0] - 1
        limy = self.in_tree.inf_map_img_array.shape[1] - 1
        # st_w_x, st_w_y  = self.pixel_to_world_conversion_1(self.in_tree.map.image_array.shape[1],self.in_tree.map.image_array.shape[0],self.in_tree.map.map_df.origin[0][0],self.in_tree.map.map_df.origin[0][1],float(start_point[1]),float(start_point[0]), 0.049)
        # ed_w_x, ed_w_y = self.pixel_to_world_conversion_1(self.in_tree.map.image_array.shape[1],self.in_tree.map.image_array.shape[0],self.in_tree.map.map_df.origin[0][0],self.in_tree.map.map_df.origin[0][1],float(end_point[1]),float(end_point[0]), 0.049)
        
        points = self.interpolate_with_spacing(start_point, end_point)
        for point in points:
            # w_x, w_y = self.world_to_pixel_conversion_1(self.in_tree.map.image_array.shape[1],self.in_tree.map.image_array.shape[0],self.in_tree.map.map_df.origin[0][0],self.in_tree.map.map_df.origin[0][1],point[0],point[1] ,0.049)
            if point[0] > limx or point[0] < 0 or point[1] > limy or point[1] < 0:
                return False
            if self.in_tree.inf_map_img_array[point[0]][point[1]] != 0:
                return False
        return True


    def interpolate_with_spacing(self,p1, p2, spacing=0.03, round_to_int=False):
        """
        Generates interpolated points between p1 and p2 with max spacing.

        Parameters:
            p1 (tuple): Start point (x1, y1)
            p2 (tuple): End point (x2, y2)
            spacing (float): Maximum distance between consecutive points
            round_to_int (bool): If True, rounds points to nearest integers

        Returns:
            list of tuple: Interpolated points
        """
        x1, y1 = p1
        x2, y2 = p2

        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        if dist == 0:
            return [p1]

        num_steps = max(1, int(dist / spacing))
        points = []
        num_steps = self.d

        for i in range(num_steps):
            # t = i / num_steps
            x = x1 + dx * i
            y = y1 + dy * i
            if round_to_int:
                x, y = int(round(x)), int(round(y))
            points.append((x, y))

        # Optionally remove duplicates if rounding caused repeats
        if round_to_int:
            points = list(dict.fromkeys(points))  # Removes duplicates while preserving order

        return points


    




def main(args=None):
    rclpy.init(args=args)

    task2 = Task2()

    try:
        rclpy.spin(task2)
    except KeyboardInterrupt:
        pass
    finally:
        task2.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
