import cv2
import numpy as np
from copy import deepcopy
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from roboflow import Roboflow
import supervision as sv
import pandas as pd
import os

fruit_filename = 'fruit_data6.csv'

class FruitRipenessDetector:
    def __init__(self):
        os.system("pwd")
        self.fruit_df = pd.DataFrame(columns=["timestamp", "ripeness", "depth", 'object_offset', 'xmin', 'ymin', 'xmax', 'ymax'])
        self.fruit_df.to_csv(fruit_filename, index=False)
    def run(self):
        # while not rospy.is_shutdown():
        # cv2.waitKey(0)
        print("Waiting for input")
        input()
        image = rospy.wait_for_message('/camera/color/image_raw', Image)
        depth_image = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', Image)

        timestamp = str(image.header.stamp)
        cv_image = CvBridge().imgmsg_to_cv2(image, "bgr8")
        cv2.imwrite('./images/' + timestamp + ".jpg", cv_image)
        os.system("python ./infer.py --model_dir=yolov3_mobilenet_v3_large_voc --image_file=./images/" + timestamp + ".jpg --output_dir=./output --timestamp=" + timestamp)
        self.fruit_df = pd.read_csv(fruit_filename)
        idx = 0
        print(self.fruit_df)
        while (idx in self.fruit_df['timestamp']):
            print("Searching for timestamp: ", timestamp + '_' + str(idx))

            df_row = self.fruit_df.loc[self.fruit_df["timestamp"] == timestamp + '_' + str(idx)]
            object_offset, depth = self.get_depth(depth_image, int(df_row['xmin']), int(df_row['ymin']), int(df_row['xmax']), int(df_row['ymax']))
            self.fruit_df.loc[self.fruit_df["timestamp"] == timestamp + '_' + str(idx), "depth"] = depth
            self.fruit_df.loc[self.fruit_df["timestamp"] == timestamp + '_' + str(idx), "object_offset"] = object_offset

            idx += 1

        self.fruit_df.to_csv(fruit_filename, index=False)

    def get_depth(self, depth_image, xmin, ymin, xmax, ymax):
        depth_image = CvBridge().imgmsg_to_cv2(depth_image, desired_encoding="passthrough")
        camera_info = rospy.wait_for_message('/camera/aligned_depth_to_color/camera_info', CameraInfo)
        fx = camera_info.K[0]
        cx = camera_info.K[2]
        depth = depth_image[ymin:ymax, xmin:xmax]
        object_center = (xmin + xmax) / 2
        object_distance = np.mean(depth)
        object_offset = (object_center - cx) * object_distance / fx
        print("Object Offset: ", object_offset)
        print("Object Distance: ", object_distance)
        return object_offset, object_distance
        # depth_image = CvBridge().imgmsg_to_cv2(depth_image, desired_encoding="passthrough")
        # avg_depth = np.mean(depth_image[ymin:ymax, xmin:xmax])
        # return avg_depth


def main():
    rospy.init_node("apple_detector")

    # # Load an image for testing
    # image = cv2.imread("ripe_mango.jpg")

    detector = FruitRipenessDetector()
    detector.run()


if __name__ == "__main__":
    
    main()
