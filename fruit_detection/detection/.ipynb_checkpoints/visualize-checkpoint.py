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



def visualize_box_mask(im, results, labels, threshold=0.5):
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
    if 'masks' in results and 'boxes' in results:
        im = draw_mask(
            im, results['boxes'], results['masks'], labels, threshold=threshold)
    if 'boxes' in results:
        im = draw_box(im, results['boxes'], labels, threshold=threshold)
    if 'segm' in results:
        im = draw_segm(
            im,
            results['segm'],
            results['label'],
            results['score'],
            labels,
            threshold=threshold)
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


def draw_mask(im, np_boxes, np_masks, labels, threshold=0.5):
    """
    Args:
        im (PIL.Image.Image): PIL image
        np_boxes (np.ndarray): shape:[N,6], N: number of box,
            matix element:[class, score, x_min, y_min, x_max, y_max]
        np_masks (np.ndarray): shape:[N, im_h, im_w]
        labels (list): labels:['class1', ..., 'classn']
        threshold (float): threshold of mask
    Returns:
        im (PIL.Image.Image): visualized image
    """
    color_list = get_color_map_list(len(labels))
    w_ratio = 0.4
    alpha = 0.7
    im = np.array(im).astype('float32')
    clsid2color = {}
    expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]
    np_masks = np_masks[expect_boxes, :, :]
    for i in range(len(np_masks)):
        clsid, score = int(np_boxes[i][0]), np_boxes[i][1]
        mask = np_masks[i]
        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color_mask = clsid2color[clsid]
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        idx = np.nonzero(mask)
        color_mask = np.array(color_mask)
        im[idx[0], idx[1], :] *= 1.0 - alpha
        im[idx[0], idx[1], :] += alpha * color_mask
    return Image.fromarray(im.astype('uint8'))

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
    #dominant_color = max(color_percents, key=color_percents.get)
    
    print(f"red:{color_percents['red']}, green:{color_percents['green']}, yellow:{color_percents['yellow']}")

    if red_percentage > 50 or (red_percentage > 30 and yellow_percentage > 30):
        return "High Ripeness"
    elif green_percentage > 50:
        return "Low Ripeness"
    elif yellow_percentage > 50:
        return "Medium Ripeness"
    else:
        return "Unknown"

    #if dominant_color == 'green':
     #   return "Low Ripeness"
    #elif dominant_color == 'yellow':
        #return "Medium Ripeness"
    #elif dominant_color == 'red':
        #return "High Ripeness"
    #else:
        #return "Unknown"


def draw_box(im, np_boxes, labels, threshold=0.5):
    draw_thickness = min(im.size) // 320
    draw = ImageDraw.Draw(im)
    clsid2color = {}
    color_list = get_color_map_list(len(labels))
    expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]

    # Initialize SAM model to use reuse for each bounding box
    print("Loading SAM model")
    sam = sam_model_registry["vit_b"](checkpoint="/mnt/c/NEU/Robotics Sensing and Navigation/project/sam_vit_b_01ec64.pth")
    sam.to("cuda")
    print("Creating mask generator")
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("Mask generated")

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
        text = "{}: {}".format(labels[clsid], ripeness_level)
        font = ImageFont.truetype("Arial.ttf", 20)
        text_size = draw.textsize(text, font=font)
        text_location = (xmin + 5, ymin - text_size[1])
        draw.rectangle([text_location[0] - 2, text_location[1] - 2, text_location[0] + text_size[0], text_location[1] + text_size[1]], fill=color)
        draw.text(text_location, text, fill='white', font=font)

    return im



def draw_segm(im,
              np_segms,
              np_label,
              np_score,
              labels,
              threshold=0.5,
              alpha=0.7):
    """
    Draw segmentation on image
    """
    mask_color_id = 0
    w_ratio = .4
    color_list = get_color_map_list(len(labels))
    im = np.array(im).astype('float32')
    clsid2color = {}
    np_segms = np_segms.astype(np.uint8)
    for i in range(np_segms.shape[0]):
        mask, score, clsid = np_segms[i], np_score[i], np_label[i]
        if score < threshold:
            continue

        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color_mask = clsid2color[clsid]
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        idx = np.nonzero(mask)
        color_mask = np.array(color_mask)
        im[idx[0], idx[1], :] *= 1.0 - alpha
        im[idx[0], idx[1], :] += alpha * color_mask
        sum_x = np.sum(mask, axis=0)
        x = np.where(sum_x > 0.5)[0]
        sum_y = np.sum(mask, axis=1)
        y = np.where(sum_y > 0.5)[0]
        x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
        cv2.rectangle(im, (x0, y0), (x1, y1),
                      tuple(color_mask.astype('int32').tolist()), 1)
        bbox_text = '%s %.2f' % (labels[clsid], score)
        t_size = cv2.getTextSize(bbox_text, 0, 0.3, thickness=1)[0]
        cv2.rectangle(im, (x0, y0), (x0 + t_size[0], y0 - t_size[1] - 3),
                      tuple(color_mask.astype('int32').tolist()), -1)
        cv2.putText(
            im,
            bbox_text, (x0, y0 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3, (0, 0, 0),
            1,
            lineType=cv2.LINE_AA)
    return Image.fromarray(im.astype('uint8'))