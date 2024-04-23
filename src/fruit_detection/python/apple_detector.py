import cv2
import numpy as np
from copy import deepcopy
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from roboflow import Roboflow
import supervision as sv
import pandas as pd
import os


class FruitRipenessDetector:
    def __init__(self):
        os.system("pwd")
        self.fruit_df = pd.DataFrame(columns=["timestamp", "ripeness", "depth", 'xmin', 'ymin', 'xmax', 'ymax'])
        self.fruit_df.to_csv("fruit_data.csv", index=False)
    def run(self):
        # while not rospy.is_shutdown():
        # cv2.waitKey(0)
        image = rospy.wait_for_message('/camera/color/image_raw', Image)
        depth_image = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', Image)

        timestamp = str(image.header.stamp)
        cv_image = CvBridge().imgmsg_to_cv2(image, "bgr8")
        cv2.imwrite('./images/' + timestamp + ".jpg", cv_image)
        os.system("python ./infer.py --model_dir=yolov3_mobilenet_v3_large_voc --image_file=./images/" + timestamp + ".jpg --output_dir=./output --timestamp=" + timestamp)
        self.fruit_df = pd.read_csv("fruit_data.csv")
        idx = 0
        print(self.fruit_df)
        while (idx in self.fruit_df['timestamp']):
            print("Searching for timestamp: ", timestamp + '_' + str(idx))

            df_row = self.fruit_df.loc[self.fruit_df["timestamp"] == timestamp + '_' + str(idx)]
            self.fruit_df.loc[self.fruit_df["timestamp"] == timestamp + '_' + str(idx), "depth"] = self.get_depth(depth_image, 
                                                                                                                  int(df_row['xmin']), 
                                                                                                                  int(df_row['ymin']), 
                                                                                                                  int(df_row['xmax']), 
                                                                                                                  int(df_row['ymax']))

            idx += 1

        self.fruit_df.to_csv("fruit_data.csv", index=False)

    def get_depth(self, depth_image, xmin, ymin, xmax, ymax):
        depth_image = CvBridge().imgmsg_to_cv2(depth_image, desired_encoding="passthrough")
        avg_depth = np.mean(depth_image[ymin:ymax, xmin:xmax])
        return avg_depth


def main():
    rospy.init_node("apple_detector")

    # # Load an image for testing
    # image = cv2.imread("ripe_mango.jpg")

    detector = FruitRipenessDetector()
    detector.run()


if __name__ == "__main__":
    
    main()
