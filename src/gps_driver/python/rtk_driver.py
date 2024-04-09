#!/usr/bin/env python3

import rospy
import serial
from gps_driver.msg import Customrtk
import sys
import utm
import time
from datetime import datetime
from std_msgs.msg import Header, Time
import rosbag


def convertToUTM(LatitudeSigned, LongitudeSigned):
    pass


class RTKDriver():

    def __init__(self):
        rospy.init_node('rtk_driver')


        self.rate = rospy.Rate(10)

        self.port = rospy.get_param('~port', '/dev/pts/7')
        rospy.loginfo(self.port)
        self.rtk_pub = rospy.Publisher('rtk', Customrtk, queue_size=1)
        self.serial_port = serial.Serial(self.port, baudrate=4800)
        rospy.loginfo("Connected to port " + self.port)
        self.rtk_msg = Customrtk()
        self.bag = rosbag.Bag('occluded_rtk.bag', 'w')

    def parseGNGGA(self, line):
        g = line.split(',')

        utc = g[1]
        lat = g[2]
        lat_dir = g[3]
        lon = g[4]
        lon_dir = g[5]
        fix_quality = g[6]
        hdop = g[8]
        alt = g[9]

        signed_lat = self.LatLongtoSigned(lat, lat_dir)
        signed_lon = self.LatLongtoSigned(lon, lon_dir)

        # print(g, signed_lat, signed_lon)
        # UTM Easting, Northing, Zone, Letter
        utm_vals = utm.from_latlon(signed_lat, signed_lon)
        # print("utm vals", utm_vals)

        utcepoch = self.UTCtoUTCEpoch(utc)

        
        self.rtk_msg.header.frame_id = "RTK1_FRAME"
        self.rtk_msg.header.stamp.secs = utcepoch[0]
        self.rtk_msg.header.stamp.nsecs = utcepoch[1]
        # self.rtk_msg.header.stamp = rospy.Time.now()

        self.rtk_msg.altitude = float(alt)
        self.rtk_msg.latitude = signed_lat
        self.rtk_msg.longitude = signed_lon
        self.rtk_msg.utm_easting = utm_vals[0]
        self.rtk_msg.utm_northing = utm_vals[1]
        self.rtk_msg.zone = utm_vals[2]
        self.rtk_msg.letter = utm_vals[3]

        self.rtk_msg.fix_quality = int(fix_quality)
        self.rtk_msg.hdop = float(hdop)
        self.rtk_msg.gngga_read = line

        print(self.rtk_msg)




    def LatLongtoSigned(self, L, dir):
        # Convert east, west, north, or south to a sign
        m = 0

        if dir == 'E' or dir == 'N':
            m = 1
        elif dir == 'W' or dir == 'S':
            m = -1

        DD = 3
        if dir == 'N' or dir == 'S':
            DD = 2
        
        # Convention DDDmm.mmm
        # minutes to degrees is m/60
        deg = int(L[:DD])
        mins = float(L[DD:]) / 60

        return m * (deg + mins)

    def UTCtoUTCEpoch(self, UTC):
        # Convention HHMMSS.SS
        UTCinSecs = 3600 * int(UTC[:2]) + 60 * int(UTC[2:4]) + float(UTC[4:])
        
        TimeSinceEpoch = time.time()
        TimeSinceEpochBOD = int(TimeSinceEpoch - (TimeSinceEpoch % 86400))
        CurrentTime = TimeSinceEpochBOD + UTCinSecs
        CurrentTimeSec = int(CurrentTime)
        # Need to round this because of floating point arithmetic in python
        CurrentTimeNsec = int((CurrentTime % 1) * (10**2)) * (10**7)

        
        return [CurrentTimeSec, CurrentTimeNsec]

    def main(self):

        try:
            while not rospy.is_shutdown():
                # rospy.loginfo("Reading line")
                line = self.serial_port.readline().decode('utf-8')
                # line = str(self.serial_port.read(20).decode('utf-8'))
                # rospy.loginfo(line)

                if "$GNGGA" in line:
                    try:
                        self.parseGNGGA(line)


                        self.rtk_pub.publish(self.rtk_msg)
                        self.bag.write('rtk', self.rtk_msg)
                    except Exception as e:
                        rospy.loginfo(e)
                        rospy.loginfo("Error parsing GNGGA line")


                    
                self.rate.sleep()
                
        except rospy.ROSInterruptException:

            self.serial_port.close()
        
        except serial.serialutil.SerialException as e:
            rospy.loginfo(e)
            rospy.loginfo("Shutting down rtk node...")
        self.bag.close()


if __name__ == '__main__':
    driver = RTKDriver()
    driver.main()