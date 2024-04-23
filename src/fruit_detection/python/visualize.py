# coding: utf-8
# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from operator import itemgetter
import pandas as pd



def visualize_box_mask(im, results, labels, threshold=0.5, imname=None):
    """
    Args:
        im (str/np.ndarray): path of image/np.ndarray read by cv2
        results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                        matix element:[class, score, x_min, y_min, x_max, y_max]
                        MaskRCNN's results include 'masks': np.ndarray:
                        shape:[N, im_h, im_w]
        labels (list): labels:['class1', ..., 'classn']
        threshold (float): Threshold of score.
    Returns:
        im (PIL.Image.Image): visualized image
    """
    if isinstance(im, str):
        im = Image.open(im).convert('RGB')
    else:
        im = Image.fromarray(im)
    if 'boxes' in results:
        im = draw_box(im, results['boxes'], labels, threshold=threshold, imname=imname)

    return im


def get_color_map_list(num_classes):
    """
    Args:
        num_classes (int): number of class
    Returns:
        color_map (list): RGB color list
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def determine_ripeness(hsv):
    # Define color ranges for masks
    masks = {
        'red1': cv2.inRange(hsv, np.array([0, 0, 0]), np.array([10, 255, 255])),
        'red2': cv2.inRange(hsv, np.array([170, 0, 0]), np.array([180, 255, 255])),
        'green': cv2.inRange(hsv, np.array([36, 0, 0]), np.array([86, 255, 255])),
        'yellow': cv2.inRange(hsv, np.array([20, 0, 0]), np.array([30, 255, 255]))
    }
    masks['red'] = masks['red1'] + masks['red2']

    # Calculate color percentages
    total_pixels = hsv.size / 3  # Number of pixels per channel
    color_counts = {color: np.sum(mask == 255) for color, mask in masks.items()}
    color_percents = {color: count / total_pixels * 100 for color, count in color_counts.items()}

    red_percentage = color_percents['red']
    green_percentage = color_percents['green']
    yellow_percentage = color_percents['yellow']

    # Determine ripeness based on dominant color proportion
    dominant_color = max(color_percents, key=color_percents.get)
    print("Dominant color: ", dominant_color)
    
    print(f"red:{color_percents['red']}, green:{color_percents['green']}, yellow:{color_percents['yellow']}")

    # if red_percentage > 50 or (red_percentage > 30 and yellow_percentage > 30):
    #     return "High Ripeness"
    # elif green_percentage > 50:
    #     return "Low Ripeness"
    # elif yellow_percentage > 50:
    #     return "Medium Ripeness"
    # else:
    #     return "Unknown"

    if dominant_color == 'green':
       return "Low Ripeness"
    elif dominant_color == 'yellow':
        return "Medium Ripeness"
    elif dominant_color == 'red':
        return "High Ripeness"
    else:
        return "Unknown"


def draw_box(im, np_boxes, labels, threshold=0.5, imname=None):
    draw_thickness = min(im.size) // 320
    draw = ImageDraw.Draw(im)
    clsid2color = {}
    color_list = get_color_map_list(len(labels))
    expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]

    # Initialize SAM model to use reuse for each bounding box
    print("Loading SAM model")
    sam = sam_model_registry["vit_b"](checkpoint="/home/drakemoore/Downloads/sam_vit_b_01ec64.pth")
    sam.to("cuda")
    print("Creating mask generator")
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("Mask generated")

    fruit_df = pd.read_csv("fruit_data.csv")

    for idx, dt in enumerate(np_boxes):
        clsid, bbox, score = int(dt[0]), dt[2:], dt[1]
        xmin, ymin, xmax, ymax = map(int, bbox)  # Ensure the coordinates are integer

        # Extract the image within the bounding box
        im_bbox = im.crop((xmin, ymin, xmax, ymax))
        segment = cv2.cvtColor(np.array(im_bbox), cv2.COLOR_RGB2BGR)

        # Generate masks for the current segment
        masks = mask_generator.generate(segment)

        # Find the largest mask by area
        largest_mask_info = max(masks, key=lambda x: x['area'])
        largest_mask = largest_mask_info['segmentation'].astype('uint8')
        print(largest_mask)
        print(largest_mask.shape)
        
        # Apply the mask to the segment to get the color image of the largest segment
        segment_color = cv2.bitwise_and(segment, segment, mask=largest_mask)

        # Convert the largest mask to HSV and determine ripeness
        segment_hsv = cv2.cvtColor(segment_color, cv2.COLOR_BGR2HSV)
        ripeness_level = determine_ripeness(segment_hsv)

        # Visualize and annotate the image
        color = tuple(clsid2color.get(clsid, color_list[clsid]))

        # Draw bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=draw_thickness)

        # Label with class and ripeness level
        text = "{}: {}".format(labels[clsid] + "_" + str(idx), ripeness_level)
        # font = ImageFont.truetype("Arial.ttf", 20)
        font = ImageFont.load_default()
        text_size = draw.textsize(text, font=font)
        text_location = (xmin + 5, ymin - text_size[1])
        draw.rectangle([text_location[0] - 2, text_location[1] - 2, text_location[0] + text_size[0], text_location[1] + text_size[1]], fill=color)
        draw.text(text_location, text, fill='white', font=font)

        print("Imname: ", imname)
        timestamp = imname.split(".")[0]
        print("Timestamp: " , timestamp)
        fruit_df.loc[len(fruit_df.index)] = [timestamp + '_' + str(idx), ripeness_level, 0, xmin, ymin, xmax, ymax]
        fruit_df.to_csv("fruit_data.csv", index=False)





    return im



