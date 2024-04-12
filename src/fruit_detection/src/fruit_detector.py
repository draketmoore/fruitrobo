import cv2
import numpy as np
from copy import deepcopy
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class FruitRipenessDetector:
    def __init__(self):
        self.kernelOpen = np.ones((5, 5), np.uint8)
        self.kernelClose = np.ones((20, 20), np.uint8)

        # Define HSV color ranges for different ripeness levels
        self.lower_green = np.array([35, 100, 50])
        self.upper_green = np.array([85, 255, 255])
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])
        self.lower_red = np.array([0, 100, 100])
        self.upper_red = np.array([10, 255, 255])
        

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
        elif yellow_percentage > 0.8:
            return "High Ripeness"
        else:
            return "Medium Ripeness"

    def process_image(self, image):
        hsv_image = self.preprocess_image(image)
        contours = self.find_contours(hsv_image)
        
        # Sort contours by area in descending order and remove small contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if contours:
            # Focus on the largest contour
            largest_contour = contours[0]
            x, y, w, h = cv2.boundingRect(largest_contour)
            crop_img = hsv_image[y:y+h, x:x+w]
        
            # Apply color masks for red, green, and yellow segments within the fruit
            red_mask = self.apply_color_mask(crop_img, self.lower_red, self.upper_red) + \
                   self.apply_color_mask(crop_img, np.array([170, 50, 50]), np.array([180, 255, 255]))
            green_mask = self.apply_color_mask(crop_img, self.lower_green, self.upper_green)
            yellow_mask = self.apply_color_mask(crop_img, self.lower_yellow, self.upper_yellow)
        
            # Calculate the percentage of each color present in the fruit
            red_percentage = self.calculate_color_percentages(red_mask, crop_img)
            green_percentage = self.calculate_color_percentages(green_mask, crop_img)
            yellow_percentage = self.calculate_color_percentages(yellow_mask, crop_img)
        

            ripeness = self.determine_ripeness(red_percentage, yellow_percentage, green_percentage)
        
            # Draw a rectangle around the fruit and label it with the ripeness level
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, ripeness, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
        return image
    
    def detect(self, image):
        p_image = self.process_image(image)
        cv2.imshow("Ripeness Detection", p_image)
        cv2.waitKey(0)

    def run(self):
        while not rospy.is_shutdown():
            # cv2.waitKey(0)

            image = rospy.wait_for_message('/camera/color/image_raw', Image)
            cv_image = CvBridge().imgmsg_to_cv2(image, "bgr8")
            self.detect(cv_image)




def main():
    rospy.init_node("fruit_ripeness_detector")

    # # Load an image for testing
    # image = cv2.imread("ripe_mango.jpg")

    detector = FruitRipenessDetector()
    detector.run()
    # processed_image = detector.process_image(image)



if __name__ == "__main__":
    
    main()
