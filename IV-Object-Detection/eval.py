import numpy as np
import torch

def IoU(boxA, boxB):
    ## From https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
	yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) for two bounding boxes.

    Args:
        box1 (list): The first bounding box in the format [x0, y0, w, h].
        box2 (list): The second bounding box in the format [x0, y0, w, h].

    Returns:
        float: The IoU value.

    """
    # Extract coordinates from the boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = min(x1 + w1, x2 + w2) - x_intersection
    h_intersection = min(y1 + h1, y2 + h2) - y_intersection

    # Handle cases where there is no intersection
    if w_intersection <= 0 or h_intersection <= 0:
        return 0.0

    # Calculate the areas of the bounding boxes and intersection
    area_box1 = w1 * h1
    area_box2 = w2 * h2
    area_intersection = w_intersection * h_intersection

    # Calculate the IoU
    iou = area_intersection / float(area_box1 + area_box2 - area_intersection)

    return iou

def calculate_highest_iou(proposed_boxes, ground_truth_boxes):
    """
    Calculate the highest Intersection over Union (IoU) score for each proposed box
    with respect to the ground truth boxes.

    Args:
        proposed_boxes (list): List of proposed boxes in the format [[x0, y0, w, h], ...].
        ground_truth_boxes (list): List of ground truth boxes in the format [[x0, y0, w, h], ...].

    Returns:
        numpy.ndarray: Array of highest IoU scores for each proposed box.

    """
    proposed_boxes = np.array(proposed_boxes)
    ground_truth_boxes = np.array(ground_truth_boxes)

    # Extract coordinates from the boxes
    x1 = proposed_boxes[:, 0]
    y1 = proposed_boxes[:, 1]
    w1 = proposed_boxes[:, 2]
    h1 = proposed_boxes[:, 3]

    x2 = ground_truth_boxes[:, 0]
    y2 = ground_truth_boxes[:, 1]
    w2 = ground_truth_boxes[:, 2]
    h2 = ground_truth_boxes[:, 3]

    # Calculate the coordinates of the intersection rectangle
    x_intersection = np.maximum(x1[:, np.newaxis], x2)
    y_intersection = np.maximum(y1[:, np.newaxis], y2)
    w_intersection = np.maximum(0, np.minimum(x1[:, np.newaxis] + w1[:, np.newaxis], x2 + w2) - x_intersection)
    h_intersection = np.maximum(0, np.minimum(y1[:, np.newaxis] + h1[:, np.newaxis], y2 + h2) - y_intersection)

    # Calculate the areas of the bounding boxes and intersection
    area_box1 = w1 * h1
    area_box2 = w2 * h2
    area_intersection = w_intersection * h_intersection

    # Calculate the IoU
    iou = area_intersection / (area_box1[:, np.newaxis] + area_box2 - area_intersection)

    # Find the highest IoU score for each proposed box
    max_iou = np.max(iou, axis=1)

    return max_iou

def nms_pytorch(P : torch.tensor ,thresh_iou : float):
    """
    From https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/
    This might also be an option: https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the image 
            along with the class predscores, Shape: [num_boxes,5].
        thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
    Returns:
        A list of filtered boxes, Shape: [ , 5]
    """
    # Convert from list of tuples to list of lists
    P = [list(ele) for ele in P]
    P = torch.FloatTensor(P)
    # we extract coordinates for every 
    # prediction box present in P
    x1 = P[:, 0]
    y1 = P[:, 1]
    x2 = P[:, 2] + P[:, 0] 
    y2 = P[:, 3] + P[:, 1]
 
    # we extract the confidence scores as well
    #scores = P[:, 4]
 
    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)
     
    # sort the prediction boxes in P
    # according to their confidence scores
    order = torch.IntTensor(list(range(0,len(P))))
 
    # initialise an empty list for 
    # filtered prediction boxes
    keep = []
     
 
    while len(order) > 0:
         
        # extract the index of the 
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]
 
        # push S in filtered predictions list
        keep.append(P[idx])
 
        # remove S from P
        order = order[:-1]
 
        # sanity check
        if len(order) == 0:
            break
         
        # select coordinates of BBoxes according to 
        # the indices in order
        xx1 = torch.index_select(x1,dim = 0, index = order)
        xx2 = torch.index_select(x2,dim = 0, index = order)
        yy1 = torch.index_select(y1,dim = 0, index = order)
        yy2 = torch.index_select(y2,dim = 0, index = order)
 
        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])
 
        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1
         
        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
 
        # find the intersection area
        inter = w*h
 
        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim = 0, index = order) 
 
        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]
         
        # find the IoU of every prediction in P with S
        IoU = inter / union
 
        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou
        order = order[mask]
     
    return keep

def assign_labels():
    return None