import cv2
import numpy as np
from copy import deepcopy
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from roboflow import Roboflow
import supervision as sv
import pandas as pd

class FruitRipenessDetector:
    def __init__(self):
        self.kernelOpen = np.ones((5, 5), np.uint8)
        self.kernelClose = np.ones((20, 20), np.uint8)

        # Define HSV color ranges for different ripeness levels
        self.lower_green = np.array([40, 50, 50])
        self.upper_green = np.array([80, 255, 255])
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([40, 255, 255])
        self.lower_red = np.array([0, 100, 100])
        self.upper_red = np.array([10, 255, 255])
        # self.display_threshold_colors()

        
        self.fruits = pd.DataFrame(columns=["timestamp", "fruit", "ripeness"])

        # rf = Roboflow(api_key="1J1vsxLnwdelvzUNnDM9")
        # project = rf.workspace().project("yolov8-firsttry")
        # self.model = project.version(6).model

    def display_threshold_colors(self):
        # Create blank images for each color threshold
        red_threshold = np.zeros((100, 100, 3), np.uint8)
        green_threshold = np.zeros((100, 100, 3), np.uint8)
        yellow_threshold = np.zeros((100, 100, 3), np.uint8)


        # Fill the images with the respective color thresholds
        red_threshold[:, :] = self.lower_red  # Set red channel to maximum
        green_threshold[:, :] = self.lower_green  # Set green channel to maximum
        yellow_threshold[:, :] = self.lower_yellow  # Set green and blue channels to maximum

        # Display the threshold images
        cv2.imshow("Red Threshold", red_threshold)
        cv2.imshow("Green Threshold", green_threshold)
        cv2.imshow("Yellow Threshold", yellow_threshold)
        cv2.waitKey(0)

        # Fill the images with the respective color thresholds
        red_threshold[:, :] = self.upper_red  # Set red channel to maximum
        green_threshold[:, :] = self.upper_green  # Set green channel to maximum
        yellow_threshold[:, :] = self.upper_yellow  # Set green and blue channels to maximum

        # Display the threshold images
        cv2.imshow("Red Threshold", red_threshold)
        cv2.imshow("Green Threshold", green_threshold)
        cv2.imshow("Yellow Threshold", yellow_threshold)
        cv2.waitKey(0)
        

    def preprocess_image(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blurred_image = cv2.GaussianBlur(hsv_image, (7, 7), 0)
        return blurred_image
        
    def find_contours(self, image):
        edge_img = deepcopy(image)
        edged = cv2.Canny(edge_img, 50, 100)
        edged = cv2.dilate(edged, self.kernelOpen, iterations=1)
        edged = cv2.erode(edged, self.kernelClose, iterations=1)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        return contours


    def apply_color_mask(self, hsv, lower_color, upper_color):
        mask = cv2.inRange(hsv, lower_color, upper_color)
        mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernelOpen)
        mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, self.kernelClose)
        return mask_close

    def calculate_color_percentages(self, mask_close, crop_img):
        area = np.sum(mask_close == 255)
        total_area = crop_img.size / 3  # Divide by 3 for HSV channels
        return area / total_area if total_area else 0

    def determine_ripeness(self, red_percentage, yellow_percentage, green_percentage):
        if green_percentage > 0.5:
            return "Low Ripeness"
        elif red_percentage > 0.8:
            return "High Ripeness"
        else:
            return "Medium Ripeness"

    def process_image(self, image):
        # Create kernels for morphological operations
        kernelOpen = np.ones((5, 5), np.uint8)
        kernelClose = np.ones((20, 20), np.uint8)


        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges and create masks
        masks = {
            'red1': cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])),
            'red2': cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255])),
            'green': cv2.inRange(hsv, np.array([40, 40, 40]), np.array([70, 255, 255])),
            'yellow': cv2.inRange(hsv, np.array([20, 50, 50]), np.array([30, 255, 255]))
        }
        masks['red'] = masks['red1'] + masks['red2']

        # Calculate color percentages
        total_pixels = hsv.size / 3  # Number of pixels per channel
        color_counts = {color: np.sum(mask == 255) for color, mask in masks.items()}
        color_percents = {color: count / total_pixels * 100 for color, count in color_counts.items()}

        # Determine ripeness based on dominant color proportion
        ripeness = "Unknown"

        # Find the dominant color by finding the maximum percentage
        dominant_color = max(color_percents, key=color_percents.get)

        if dominant_color == 'green':
            ripeness = "Low Ripeness"
        elif dominant_color == 'yellow':
            ripeness = "Medium Ripeness"
        elif dominant_color == 'red':
            ripeness = "High Ripeness"

        # Print color percentages and ripeness in the console
        print(f"Ripeness: {ripeness}")
        print(f"Red: {color_percents['red']:.2f}%")
        print(f"Green: {color_percents['green']:.2f}%")
        print(f"Yellow: {color_percents['yellow']:.2f}%")

        # Find the largest contour in the combined mask
        total_mask = sum(masks.values())
        contours, _ = cv2.findContours(total_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, ripeness, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


        return image, ripeness
    
    def detect(self, image):
        p_image, ripeness = self.process_image(image)

        timestamp = rospy.Time.now()
        self.fruits = pd.concat([self.fruits, pd.DataFrame([[timestamp, "apple", ripeness]], columns=["timestamp", "fruit", "ripeness"])])
        self.fruits.to_csv("fruits.csv", index=False)

        cv2.imshow("Ripeness Detection", p_image)
        cv2.waitKey(0)


    def yolo_detect(self, image):
        
        cv2.imwrite("temp.jpg", image)

        result = self.model.predict('/home/drakemoore/fruitrobo/temp.jpg', confidence=40).json()

        labels = [item["class"] for item in result["predictions"]]

        detections = sv.Detections.from_roboflow(result)

        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()

        image = cv2.imread("temp.jpg")

        annotated_image = mask_annotator.annotate(
            scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels)

        sv.plot_image(image=annotated_image, size=(16, 16))

    def run(self):
        while not rospy.is_shutdown():
            # cv2.waitKey(0)
            image = rospy.wait_for_message('/camera/color/image_raw', Image)
            cv_image = CvBridge().imgmsg_to_cv2(image, "bgr8")
            self.detect(cv_image)
            # self.yolo_detect(cv_image)




def main():
    rospy.init_node("fruit_ripeness_detector")

    # # Load an image for testing
    # image = cv2.imread("ripe_mango.jpg")

    detector = FruitRipenessDetector()
    detector.run()
    # processed_image = detector.process_image(image)



if __name__ == "__main__":
    
    main()
