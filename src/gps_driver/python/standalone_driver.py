#!/usr/bin/env python3

import rospy
import serial
from gps_driver.msg import Customgps
import sys
import utm
import time
from datetime import datetime
from std_msgs.msg import Header, Time


def convertToUTM(LatitudeSigned, LongitudeSigned):
    pass


class StandaloneDriver():

    def __init__(self):
        rospy.init_node('standalone_driver')


        self.rate = rospy.Rate(10)

        self.port = rospy.get_param('~port', '/dev/ttyUSB0')
        rospy.loginfo(self.port)
        self.gps_pub = rospy.Publisher('gps', Customgps, queue_size=1)
        self.serial_port = serial.Serial(self.port, baudrate=4800)
        rospy.loginfo("Connected to port " + self.port)
        self.gps_msg = Customgps()

    def parseGPGGA(self, line):
        g = line.split(',')
        utc = g[1]
        lat = g[2]
        lat_dir = g[3]
        lon = g[4]
        lon_dir = g[5]
        hdop = g[8]
        alt = g[9]

        signed_lat = self.LatLongtoSigned(lat, lat_dir)
        signed_lon = self.LatLongtoSigned(lon, lon_dir)

        # print(g, signed_lat, signed_lon)
        # UTM Easting, Northing, Zone, Letter
        utm_vals = utm.from_latlon(signed_lat, signed_lon)
        # print("utm vals", utm_vals)

        utcepoch = self.UTCtoUTCEpoch(utc)

        
        self.gps_msg.header.frame_id = "GPS1_FRAME"
        self.gps_msg.header.stamp.secs = utcepoch[0]
        self.gps_msg.header.stamp.nsecs = utcepoch[1]
        # self.gps_msg.header.stamp = rospy.Time.now()

        self.gps_msg.altitude = float(alt)
        self.gps_msg.latitude = signed_lat
        self.gps_msg.longitude = signed_lon
        self.gps_msg.utm_easting = utm_vals[0]
        self.gps_msg.utm_northing = utm_vals[1]
        self.gps_msg.zone = utm_vals[2]
        self.gps_msg.letter = utm_vals[3]

        self.gps_msg.hdop = float(hdop)
        self.gps_msg.gpgga_read = line

        print(self.gps_msg)




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

                if "$GPGGA" in line:
                    self.parseGPGGA(line)

                    self.gps_pub.publish(self.gps_msg)


                    
                self.rate.sleep()
                
        except rospy.ROSInterruptException:
            self.serial_port.close()
        
        except serial.serialutil.SerialException as e:
            rospy.loginfo(e)
            rospy.loginfo("Shutting down gps node...")


if __name__ == '__main__':
    driver = StandaloneDriver()
    driver.main()