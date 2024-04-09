#!/usr/bin/env python3

import rospy
import serial
from vn_driver.msg import Vectornav
import sys
import time
from std_msgs.msg import Header, Time
import rosbag
import math



class IMUDriver():

    def __init__(self):
        rospy.init_node('rtk_driver')


        # self.rate = rospy.Rate(10)

        self.port = rospy.get_param('~port', '/dev/ttyUSB0')
        rospy.loginfo(self.port)
        self.imu_pub = rospy.Publisher('imu', Vectornav, queue_size=1)
        self.serial_port = serial.Serial(self.port, baudrate=115200)
        rospy.loginfo("Connected to port " + self.port)
        self.serial_port.write(b'VNWRG,07,40*XX\r\n')
        print(self.serial_port.readline().decode('utf-8'))
        self.imu_msg = Vectornav()
        self.imu_msg.header = Header()
        self.imu_msg.header.frame_id = "imu1_frame"
        # self.rosbag = rosbag.Bag('imu.bag', 'w')



    # 1 Yaw float deg Calculated attitude heading angle in degrees
    # 2 Pitch float deg Calculated attitude pitch angle in degrees.
    # 3 Roll float deg Calculated attitude roll angle in degrees.
    # 4 MagX float Gauss Compensated magnetometer measurement in x-axis.
    # 5 MagY float Gauss Compensated magnetometer measurement in y-axis.
    # 6 MagZ float Gauss Compensated magnetometer measurement in z-axis.
    # 7 AccelX float m/s2 Compensated accelerometer measurement in x-axis.
    # 8 AccelY float m/s2 Compensated accelerometer measurement in y-axis.
    # 9 AccelZ float m/s2 Compensated accelerometer measurement in z-axis.
    # 10 GyroX float rad/s Compensated angular rate in x-axis.
    # 11 GyroY float rad/s Compensated angular rate in y-axis.
    # 12 GyroZ
    def parseVNYMR(self, line):
        vnymr_split = line.split(',')   

        quat_orientation = self.euler_to_quaternion(float(vnymr_split[1]), 
                                                    float(vnymr_split[2]), 
                                                    float(vnymr_split[3]))
        self.imu_msg.imu.orientation.x = quat_orientation[0]
        self.imu_msg.imu.orientation.y = quat_orientation[1]
        self.imu_msg.imu.orientation.z = quat_orientation[2]
        self.imu_msg.imu.orientation.w = quat_orientation[3]

        self.imu_msg.imu.angular_velocity.x = float(vnymr_split[10])
        self.imu_msg.imu.angular_velocity.y = float(vnymr_split[11])
        self.imu_msg.imu.angular_velocity.z = float(vnymr_split[12].split('*')[0])

        self.imu_msg.imu.linear_acceleration.x = float(vnymr_split[7])
        self.imu_msg.imu.linear_acceleration.y = float(vnymr_split[8])
        self.imu_msg.imu.linear_acceleration.z = float(vnymr_split[9])

        self.imu_msg.magnetic_field.magnetic_field.x = float(vnymr_split[4])
        self.imu_msg.magnetic_field.magnetic_field.y = float(vnymr_split[5])
        self.imu_msg.magnetic_field.magnetic_field.z = float(vnymr_split[6])

        self.imu_msg.vnymr_string = line
        self.imu_msg.header.stamp = rospy.Time.now()


        

    def euler_to_quaternion(self, yaw_deg, pitch_deg, roll_deg):
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)

        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = [0] * 4
        q[0] = cy * cp * cr + sy * sp * sr
        q[1] = cy * cp * sr - sy * sp * cr
        q[2] = sy * cp * sr + cy * sp * cr
        q[3] = sy * cp * cr - cy * sp * sr
        return q

    def main(self):

        try:
            while not rospy.is_shutdown():
                # rospy.loginfo("Reading line")
                line = self.serial_port.readline().decode('utf-8')
                # line = str(self.serial_port.read(20).decode('utf-8'))
                # rospy.loginfo(line)

                if "$VNYMR" in line:
                    try:
                        self.parseVNYMR(line)
                        # print(self.imu_msg)
                        # self.rosbag.write('imu', self.imu_msg)
                        self.imu_pub.publish(self.imu_msg)
                    except Exception as e:
                        rospy.loginfo(e)
                        rospy.loginfo("Error parsing GNGGA line")


                    
                # self.rate.sleep()
                
        except rospy.ROSInterruptException:

            self.serial_port.close()
        
        except serial.serialutil.SerialException as e:
            rospy.loginfo(e)
            rospy.loginfo("Shutting down imu node...")
        self.bag.close()


if __name__ == '__main__':
    driver = IMUDriver()
    driver.main()