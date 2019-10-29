import numpy as np
import cv2
import os
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error


green_center = (0, 0)
green_radius = 0.0
#green ball gimp hue is 90, sat 58-70, value 67-80
#corresponding opencv values are hue: 45, sat 148-178, val = 170-204
green_center = (0, 0)
green_radius = 0.0
greenLower = (38, 125, 120)
#greenLower = (20, 115, 95)
greenUpper = (60, 190, 210)
greenUpper = (46, 178, 204)

greenLower_relaxed = (22, 106, 104)
greenUpper_relaxed = (60, 196, 224)

gold_center = (0, 0)
gold_radius = 0.0
goldLower = (20, 148, 48)
goldUpper = (22, 245, 247)
goldLower_relaxed = (15, 142, 42)
goldUpper_relaxed = (28, 250, 252)

balls_info_pickle = open('data/target_size_data/balls_info.pickle', 'rb')
balls_info = pickle.load(balls_info_pickle)
print(len(balls_info))
balls_info_pickle.close()
balls_size_labels = ['tiny', 'small', 'medium', 'big1', 'huge']

#for displaying purpose -  colors and location info
target_prints = {'green': {'color' : (0, 255, 0), 'loc':(10,10)}, \
                 'gold' : {'color' : (0, 215, 255), 'loc' : (10, 20)}}

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def grab_contours(cnts):
    #if the length the contours tuple returned by cv2.findContours
    #is '2' then we are using either OpenCV v2.4, v4-beta, or
    #v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    #if the length of the contours tuple is '3' then we are using
    #either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    #otherwise OpenCV has changed their cv2.findContours return
    #signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    #return the actual contours array
    return cnts

def find_green_target(green_mask):
    green_target =  {'target_found': None, 'bbx':None, 'bby':None, 'bbwidth': None, 'bbheight': None }

    cnts = cv2.findContours(green_mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), green_radius) = cv2.minEnclosingCircle(c)
        bbx, bby, bbwidth, bbheight = cv2.boundingRect(c)
        M = cv2.moments(c)
        if M["m00"] == 0:
            # print('moments: ', M["m00"], M["m01"], M['m10'], M['m11'], ", radius = ", green_radius, ", contour length = ", len(c))
            if len(c) > 0:
                cx = 0
                cy = 0
                for p in c:
                    cx += p[0][0]
                    cy += p[0][1]
                cx = int(cx / len(c))
                cy = int(cy / len(c))
                green_center = (cx, cy)
                green_radius = 4.0
                # print("Green center = ", green_center)
                # input("observe")
            else:
                green_center = (42, 42)
                green_radius = 4.0
        else:
            green_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if green_radius > 1e-05:
            target_found = True
            # print("target radius = {0:.2f}".format(radius))
            # print("target center = ", center)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx):
                bloat_x = max(bbx - 10, 0)
                bloat_y = max(bby - 10, 0)
                bloat_x2 = min(bbx + bbwidth + 5, 83)
                bloat_y2 = min(bby + bbheight + 5, 83)
                #cv2.rectangle(bgr, (bbx, bby), (bbx + bbwidth, bby + bbheight), (0, 255, 0), 2)
                #dist_to_height_ratio = float(bbheight * (84 - (bby + bbheight)))
                #ratio_text = str(dist_to_height_ratio) + " _ " + str((bby + bbheight))
                #cv2.putText(bgr, ratio_text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), lineType=cv2.LINE_AA)
                green_target['target_found'] = target_found
                green_target['bbx'] = bbx
                green_target['bby'] = bby
                green_target['bbwidth'] = bbwidth
                green_target['bbheight'] = bbheight

    return green_target

def find_gold_target(gold_mask):
    gold_target = {'target_found': None, 'bbx': None, 'bby': None, 'bbwidth': None, 'bbheight': None}
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(gold_mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), gold_radius) = cv2.minEnclosingCircle(c)
        bbx, bby, bbwidth, bbheight = cv2.boundingRect(c)
        M = cv2.moments(c)
        if M["m00"] != 0.0:
            gold_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            gold_center = (0, 0)
            gold_radius = 0.0

        # only proceed if the radius meets a minimum size
        if gold_radius > 0.01:
            target_found = True
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx):
                bloat_x = max(bbx - 10, 0)
                bloat_y = max(bby - 10, 0)
                bloat_x2 = min(bbx + bbwidth + 5, 83)
                bloat_y2 = min(bby + bbheight + 5, 83)
                # cv2.rectangle(bgr, (bbx, bby), (bbx + bbwidth, bby + bbheight), (0, 255, 0), 2)
                # dist_to_height_ratio = float(bbheight * (84 - (bby + bbheight)))
                # ratio_text = str(dist_to_height_ratio) + " _ " + str((bby + bbheight))
                # cv2.putText(bgr, ratio_text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), lineType=cv2.LINE_AA)
                gold_target['target_found'] = target_found
                gold_target['bbx'] = bbx
                gold_target['bby'] = bby
                gold_target['bbwidth'] = bbwidth
                gold_target['bbheight'] = bbheight

    return gold_target

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

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
    return iou, boxAArea, boxBArea, interArea

def check_if_it_contains(boxA, boxB):
    contains_flag = False
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    if abs(boxA[0] - xA) < 4 and abs(boxA[1] - yA) < 4 and  abs(boxA[2] - xB) < 4 and abs(boxA[3] - yB) < 4 :
        contains_flag = True
    return contains_flag

def find_target_size(balls_info, ball_info):
    mse_list = []
    #get unique readings from the collected dist and height readings
    ball_info = np.unique(ball_info, axis=0)
    df = pd.DataFrame({'dist': ball_info[:, 0], 'height': ball_info[:, 1]})
    #group all the readings that are taken from distance and average them
    temp = df.groupby('dist').mean().reset_index()
    for i in range(0, len(balls_info)):
        # finding the matching rows with pickled ball_info and the new entries
        matching_rows = balls_info[i][balls_info[i]['dist'].isin(temp['dist'])]
        interesection_rows = temp[temp['dist'].isin(matching_rows['dist'])]
        if len(np.array(matching_rows['height'])) > 0 and len(np.array(interesection_rows['height'])) > 0:
            mse = mean_squared_error(np.array(matching_rows['height']), np.array(interesection_rows['height']))
            mse_list.append(mse)
    return np.array(mse_list)

def draw_targets(target_info, bgr):
    if (target_info['target_found']):
        bbx = target_info['bbx']
        bby = target_info['bby']
        bbwidth = target_info['bbwidth']
        bbheight = target_info['bbheight']
        cv2.rectangle(bgr, (bbx, bby), (bbx + bbwidth, bby + bbheight), (255, 255, 0), 2)

def get_target_size_and_draw(target_info, target_info_relaxed, bgr, ball_info, target_print):
    indx = -1
    if target_info['target_found']:
        # find the bboxes for strict and relaxed masks
        gt_x2 = target_info['bbx'] + target_info['bbwidth']
        gt_y2 = target_info['bby'] + target_info['bbheight']
        bboxA = np.array([target_info['bbx'], target_info['bby'], gt_x2, gt_y2])
        gt_x2 = target_info_relaxed['bbx'] + target_info_relaxed['bbwidth']
        gt_y2 = target_info_relaxed['bby'] + target_info_relaxed['bbheight']
        bboxB = np.array([target_info_relaxed['bbx'], target_info_relaxed['bby'], gt_x2, gt_y2])

        # find the overlap
        iou, bboxA_area, bboxB_area, inter_area = bb_intersection_over_union(bboxA, bboxB)
        # only if there is a overlap
        if inter_area > 0:
            strict_to_inter = float(bboxA_area / inter_area)
            # only if the overlap between the strict and relaxed is more than 70% consider the bbox legit
            if strict_to_inter > 0.7:
                bbx = target_info_relaxed['bbx']
                bby = int(target_info_relaxed['bby'])
                bbwidth = int(target_info_relaxed['bbwidth'])
                bbheight = int(target_info_relaxed['bbheight'])
                cv2.rectangle(bgr, (bbx, bby), (bbx + bbwidth, bby + bbheight), target_print['color'], 2)
                dist_to_height_ratio = float(bbheight * (84 - (bby + bbheight)))
                ratio_text = str(84 - (bby + bbheight)) + " , " + str(bbheight)
                # append the readings - dist  and height
                ball_info.append([84 - (bby + bbheight), bbheight])
                if ((84 - (bby + bbheight)) > 0) and (len(ball_info) > 0):
                    # print('ball info len : {} '.format(len(ball_info)))
                    # calculate the mse w.r.t to all the pickled ball size infos
                    mse_list = find_target_size(balls_info, ball_info)
                    if len(mse_list) > 0:
                        # choose the lowest MSE
                        indx = np.argmin(mse_list)
                # if rec_data:
                #     outfile.write(ratio_text + "\n")
    return indx

# function to find the green target
def green_target_finder(hsv, bgr, ball_info):
    global greenLower, greenUpper, greenLower_relaxed, greenUpper_relaxed
    # get the mask - stricter version
    green_mask = cv2.inRange(hsv, greenLower, greenUpper)
    # get the relaxed masked, reason to do this : strict version doesnt capture entire bbox sometimes
    # so height info may be wrong
    green_mask_relaxed = cv2.inRange(hsv, greenLower_relaxed, greenUpper_relaxed)
    # find the bboxes for both settings
    green_target = find_green_target(green_mask)
    green_target_relaxed = find_green_target(green_mask_relaxed)
    indx = get_target_size_and_draw(green_target, green_target_relaxed, bgr, ball_info, target_prints['green'])
    return indx

# similar function as  "green_target_finder()"
def gold_target_finder(hsv, bgr, ball_info_gold):
    global goldLower, goldUpper, goldLower_relaxed, goldUpper_relaxed
    gold_mask = cv2.inRange(hsv, goldLower, goldUpper)
    gold_mask_relaxed = cv2.inRange(hsv, goldLower_relaxed, goldUpper_relaxed)
    gold_target = find_gold_target(gold_mask)
    gold_target_relaxed = find_gold_target(gold_mask_relaxed)
    indx = get_target_size_and_draw(gold_target, gold_target_relaxed, bgr, ball_info_gold, target_prints['gold'])
    return indx
