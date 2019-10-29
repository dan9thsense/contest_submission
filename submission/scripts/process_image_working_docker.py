# important to remember that opencv image rows are the first index, cols are the second index
# For example, in the observation image, the top left pixel has index [0][0], the top center pixel index is [0][41], the top right pixel index = [0][83]
# the bottom left pixel index = [83][0], the bottom middle index is [83][41], and the bottom right index = [83][83]
# In the 40x40 arena maps, the top left pixel has index [0][0], the top center pixel index is [0][19], the top right pixel index = [0][39]
# the bottom left pixel index = [39][0], the bottom middle index is [39][19], and the bottom right index = [39][39]
# In the 80x80 arena maps, the top left pixel has index [0][0], the top center pixel index is [0][39], the top right pixel index = [0][79]
# the bottom left pixel index = [79][0], the bottom middle index is [79][39], and the bottom right index = [79][79]
# image is size (self.max_pixel x self.max_pixel), nominally 84x84, has a 60 degree field of view

import cv2
import numpy as np
import circle_fit as cf
from keras.models import load_model

X_MODEL_OFFSET = 50  # offset for target base X to ensure it is > 0
MODEL_MIN = 0.5  # min value of dataset for normalizing data, obtained during training
MODEL_MAX = 173.62371289336767  # max value for normalizing data
#TARGET_SIZE_MODEL_PATH = '../submission/trained_networks/target_size_model/120_adam_75.hdf5'
ORIGINAL_HOTZONE_CODE = False

#for docker
TARGET_SIZE_MODEL_PATH = '/aaio/trained_networks/target_size_model/120_adam_75.hdf5'


class ProcessImage:
    def __init__(self, max_pixel, min_green_radius, min_red_radius, watch_image):
        self.max_pixel = max_pixel
        self.min_green_radius = min_green_radius
        self.min_red_radius = min_red_radius
        self.watch_image = watch_image
        self.floorX = 0
        self.floorY = 0
        self.floor_slope = 0.0
        self.floor_size = 0
        self.top_of_floor = 0

        self.green_center = (0, 0)
        self.gold_center = (0, 0)
        self.red_center = (0, 0)
        self.green_radius = 0.
        self.gold_radius = 0.
        self.red_radius = 0.

        self.green_size_index = 0
        self.gold_size_index = 0
        self.red_size_index = 0

        self.target_size_model = load_model(TARGET_SIZE_MODEL_PATH)

    def __del__(self):
        cv2.destroyAllWindows()

    def reset_target_info(self):
        self.green_ball_info = []
        self.gold_ball_info = []

    # image is self.max_pixel x self.max_pixel, nominally 84x84, has a 60 degree field of view
    def grab_contours(self, cnts):
        # if the length the contours tuple returned by cv2.findContours
        # is '2' then we are using either OpenCV v2.4, v4-beta, or
        # v4-official
        if len(cnts) == 2:
            cnts = cnts[0]

        # if the length of the contours tuple is '3' then we are using
        # either OpenCV v3, v4-pre, or v4-alpha
        elif len(cnts) == 3:
            cnts = cnts[1]

        # otherwise OpenCV has changed their cv2.findContours return
        # signature yet again and I have no idea WTH is going on
        else:
            raise Exception(("Contours tuple must have length 2 or 3, "
                             "otherwise OpenCV changed their cv2.findContours return "
                             "signature yet again. Refer to OpenCV's documentation "
                             "in that case"))

        # return the actual contours array
        return cnts

    def find_green_target(self, green_mask, mask_type):
        # return dictionary for target information
        green_target = {'target_found': None, 'bbx': None, 'bby': None, 'bbwidth': None,
                        'bbheight': None, 'circle_radius': None}
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(green_mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = self.grab_contours(cnts)
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), green_radius) = cv2.minEnclosingCircle(c)
            if mask_type == 'strict':
                self.green_center = (int(x), int(y))
                self.green_radius = green_radius
            bbx, bby, bbwidth, bbheight = cv2.boundingRect(c)
            green_center = (int(x), int(y))
            # find the convex hull
            hull = cv2.convexHull(c)
            h = np.array(hull).reshape(len(hull), 2)
            h = h[h[:, 0] > 0]
            h = h[h[:, 1] > 0]

            # only proceed if the radius meets a minimum size
            if green_radius > 1e-05:
                green_target['target_found'] = True
                green_target['circle_radius'] = int(green_radius + 0.5)
                # bbox is needed to compare overlap between strict and relaxed target masks
                green_target['bbx'] = bbx
                green_target['bby'] = bby
                green_target['bbwidth'] = bbwidth
                green_target['bbheight'] = bbheight
                green_target['hull'] = h
                green_target['type'] = 'green'
                green_target['center'] = green_center
                #print("green contour found radius = ", green_radius)
        else:
            #print("no contour found that was green")
            pass

        return green_target

    def find_gold_target(self, gold_mask, mask_type):
        gold_target = {'target_found': None, 'bbx': None, 'bby': None, 'bbwidth': None,
                       'bbheight': None, 'circle_radius': None}
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(gold_mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = self.grab_contours(cnts)

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), gold_radius) = cv2.minEnclosingCircle(c)
            if mask_type == 'strict':
                self.gold_center = (int(x), int(y))
                self.gold_radius = gold_radius
            gold_center = (int(x), int(y))
            bbx, bby, bbwidth, bbheight = cv2.boundingRect(c)
            '''
            M = cv2.moments(c)
            if M["m00"] != 0.0:
                gold_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                gold_center = (int(x), int(y))  # (int(self.max_pixel / 2), int(self.max_pixel / 2))
                gold_radius = 1e-04
            '''
            hull = cv2.convexHull(c)
            h = np.array(hull).reshape(len(hull), 2)
            h = h[h[:, 0] > 0]
            h = h[h[:, 1] > 0]
            # only proceed if the radius meets a minimum size
            if gold_radius > 1e-05:
                gold_target['target_found'] = True
                gold_target['circle_radius'] = int(gold_radius + 0.5)
                gold_target['bbx'] = bbx
                gold_target['bby'] = bby
                gold_target['bbwidth'] = bbwidth
                gold_target['bbheight'] = bbheight
                gold_target['hull'] = h
                gold_target['type'] = 'gold'
                gold_target['center'] = gold_center

        return gold_target

    # this method is almost the same code as the original code for detecting hotzones. hotzone details are returned in a dict
    def find_hotzone(self, hotzone_mask, mask_type):
        # return dictionary for target information
        hotzone_info = {'target_found': None, 'bbx': None, 'bby': None, 'bbwidth': None,
                        'bbheight': None, 'radius': None, 'center': None, 'type': None}
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        hotzone_cnts = cv2.findContours(hotzone_mask.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        hotzone_cnts = self.grab_contours(hotzone_cnts)
        # cv2.drawContours(bgr, hotzone_cnts, -1, (255, 0, 0), 1)
        # only proceed if at least one contour was found
        if len(hotzone_cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(hotzone_cnts, key=cv2.contourArea)
            ((x, y), hotzone_radius) = cv2.minEnclosingCircle(c)
            bbx, bby, bbwidth, bbheight = cv2.boundingRect(c)
            M = cv2.moments(c)
            if M["m00"] != 0.0:
                hotzone_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                hotzone_center = (int(x), int(y))  # (int(self.max_pixel / 2), int(self.max_pixel / 2))
                hotzone_radius = 0.0

            hotzone_size = np.count_nonzero(hotzone_mask) / (self.max_pixel / 4)
            # print("hotzone size = ", hotzone_size, ", hotzone radius = ", hotzone_radius)
            hotzone_radius = hotzone_size

            # only proceed if the radius meets a minimum size
            if hotzone_radius > 0.01:
                hotzone_info['target_found'] = True
                hotzone_info['radius'] = hotzone_radius
                # bbox is needed to compare overlap between strict and relaxed target masks
                hotzone_info['bbx'] = bbx
                hotzone_info['bby'] = bby
                hotzone_info['bbwidth'] = bbwidth
                hotzone_info['bbheight'] = bbheight
                hotzone_info['type'] = 'hotzone'
                hotzone_info['center'] = hotzone_center
                hotzone_info['cnts'] = hotzone_cnts
                # print("green contour found radius = ", green_radius)
        else:
            # print("no contour found that was green")
            pass

        return hotzone_info

    def bb_intersection_over_union(self, boxA, boxB):
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

    def check_overlap(self, target_info, target_info_relaxed):
        # proceed only if strict version of masks detected a target
        overlap_flag = False
        if target_info['target_found']:
            # find the bboxes for strict and relaxed masks
            gt_x2 = target_info['bbx'] + target_info['bbwidth']
            gt_y2 = target_info['bby'] + target_info['bbheight']
            bboxA = np.array([target_info['bbx'], target_info['bby'], gt_x2, gt_y2])
            gt_x2 = target_info_relaxed['bbx'] + target_info_relaxed['bbwidth']
            gt_y2 = target_info_relaxed['bby'] + target_info_relaxed['bbheight']
            bboxB = np.array([target_info_relaxed['bbx'], target_info_relaxed['bby'], gt_x2, gt_y2])

            # calculate the overlap between strict and relaxed mask detections
            iou, bboxA_area, bboxB_area, inter_area = self.bb_intersection_over_union(bboxA, bboxB)
            # only if there is a overlap
            if inter_area > 0:
                strict_to_inter = float(bboxA_area / inter_area)
                # only if the overlap between the strict and relaxed is more than 70% consider the bbox legit
                if strict_to_inter > 0.7:
                    overlap_flag = True
        return overlap_flag

    def get_target_size_and_draw(self, target_info, target_info_relaxed, bgr, target_type):
        # proceed only if strict version of masks detected a target
        if target_info['target_found']:
            # find the bboxes for strict and relaxed masks
            gt_x2 = target_info['bbx'] + target_info['bbwidth']
            gt_y2 = target_info['bby'] + target_info['bbheight']
            bboxA = np.array([target_info['bbx'], target_info['bby'], gt_x2, gt_y2])
            gt_x2 = target_info_relaxed['bbx'] + target_info_relaxed['bbwidth']
            gt_y2 = target_info_relaxed['bby'] + target_info_relaxed['bbheight']
            bboxB = np.array([target_info_relaxed['bbx'], target_info_relaxed['bby'], gt_x2, gt_y2])

            # calculate the overlap between strict and relaxed mask detections
            iou, bboxA_area, bboxB_area, inter_area = self.bb_intersection_over_union(bboxA, bboxB)
            # only if there is a overlap
            if inter_area > 0:
                strict_to_inter = float(bboxA_area / inter_area)
                # only if the overlap between the strict and relaxed is more than 70% consider the bbox legit
                if strict_to_inter > 0.7:
                    if (len(target_info_relaxed['hull']) >= 3):
                        # fit a circle for the points obtained from convex hull on relaxed mask
                        xc, yc, r, _ = cf.least_squares_circle(target_info_relaxed['hull'])
                        # calculate the cordinates of 4 points: bottom, top, left and right
                        bottom_x = xc
                        bottom_y = yc + r
                        right_x = xc + r
                        right_y = yc
                        left_x = xc - r
                        left_y = yc
                        top_x = xc
                        top_y = yc - r
                        right = [xc + r, yc]
                        left = [xc - r, yc]
                        top = [top_x, top_y]
                        bottom = [bottom_x, bottom_y]
                        quadrants = [top, right, bottom, left]
                        #print(f'top = {[top_x, top_y]}, bottom = {[bottom_x, bottom_y]},'
                        #      f' right = {[right_x, right_y]}, left = {[left_x, left_y]}')
                        # add 50 to x to get rid of negative X values (helped in achieving higher accuracy)
                        dp = np.array([bottom_x + X_MODEL_OFFSET, bottom_y, r]).reshape((1, 3))
                        # normalize data point
                        dp = (dp - MODEL_MIN) / (MODEL_MAX - MODEL_MIN)
                        size_preds = self.target_size_model.predict(dp)
                        size_class = np.argmax(size_preds)
                        # valid target only if bottom Y is > 2, else it is chopped part of target
                        if bottom_y >= 41.0:
                            if target_type == 'green':
                                # self.green_radius = r
                                self.green_size_index = float(size_class + 1) / 2  # offset by 1 to avoid zero
                            else:
                                # self.gold_radius = r
                                self.gold_size_index = float(size_class + 1) / 2  # offset by 1 to avoid zero
                        # draw fitted circle
                        cv2.circle(bgr, (int(xc), int(yc)), int(r), (0, 0, 0), 2)
                        # draw lines each from center to the 4 points
                        cv2.line(bgr, (int(xc), int(yc)), (int(bottom_x), int(bottom_y)), (186, 72, 0), 2)
                        cv2.line(bgr, (int(xc), int(yc)), (int(top_x), int(top_y)), (186, 72, 0), 2)
                        cv2.line(bgr, (int(xc), int(yc)), (int(right_x), int(right_y)), (186, 72, 0), 2)
                        cv2.line(bgr, (int(xc), int(yc)), (int(left_x), int(left_y)), (186, 72, 0), 2)
                        cv2.line(bgr, (int(bottom_x - 5), int(bottom_y)), (int(bottom_x + 5), int(bottom_y)),
                                 (182, 72, 0), 2)
                        if target_type == "green":
                            cv2.putText(bgr, str(size_class + 1), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
                                        cv2.LINE_AA)
                        else:
                            cv2.putText(bgr, str(size_class + 1), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (32, 165, 218),
                                        1,
                                        cv2.LINE_AA)
                    else:
                        #if too less points to fit the circle
                        if target_type == 'green':
                                #self.green_radius = 0.5
                                self.green_size_index = 0.5
                                cv2.putText(bgr, str(1), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                                (0, 255, 0), 1,
                                                cv2.LINE_AA)
                        else:
                                #self.gold_radius = 0.5
                                self.gold_size_index = 0.5
                                cv2.putText(bgr, str(1), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                    (32, 165, 218),
                                    1,
                                    cv2.LINE_AA)

    def find_targets(self, frame, accept_green):
        target_found = False
        # check for blackout
        if np.count_nonzero(frame) < 100:
            return 0, -1, 0.0, 0.0, 0.0, (0, 0), 0, False, -1, False, True
            # radius, size_index, centerX, centerY, hotzone_radius, hotzone_center, color, obstacle_in_view, wall_in_view, red_in_view, blackout

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imshow("Array", bgr)
        # blurred = cv2.GaussianBlur(bgr, (11, 11), 0)
        # hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        get_hsv_values = 0  # int(input("print hsv?"))
        if get_hsv_values != 0:
            np.set_printoptions(threshold=np.inf)
            print(hsv)
        # cv2.imshow("HSV", hsv)

        # opencv has HSV ranges of 0-179 for Hue, and 0-255 for Saturation and Value
        # typical tables use a range of 0-100, so you have to be careful to convert
        # the conversion from gimp is:
        # opencvH = gimpH / 2
        # opencvS = (gimpS / 100) * 255
        # opencvV = (gimpV / 100) * 255

        # green ball gimp hue is 90, sat 58-70, value 67-80
        # corresponding opencv values are hue: 45, sat 148-178, val = 170-204
        self.green_center = (0, 0)
        self.green_radius = 0.0
        greenLower = (38, 125, 120)
        greenUpper = (46, 178, 204)
        # greenLower = (44, 148, 170)
        # greenUpper = (46, 178, 204)
        # greenLower = (25, 20, 0)
        # greenUpper = (70, 255, 255)
        # greenLower = (25, 20, 128)
        # greenUpper = (40, 255, 255)
        # second filter to let more green pixels, the first filter is strict: it is accurate but misses parts of target
        greenLower_relaxed = (28, 106, 104)
        greenUpper_relaxed = (60, 196, 224)

        self.gold_center = (0, 0)
        self.gold_radius = 0.0
        # gold ball gimp hue is 41-42, sat 58-96, value 19-97
        # corresponding opencv values are hue: 20-21, sat 148-245, val = 48-247
        goldLower = (20, 148, 48)
        goldUpper = (22, 245, 247)
        # goldLower = (15, 128, 128)
        # goldUpper = (30, 255, 255)
        # second filter to let more gold pixels, the first filter is strict: it is accurate but misses parts of target
        goldLower_relaxed = (15, 142, 42)
        goldUpper_relaxed = (25, 252, 251)

        # red floor gimp hue is 0, sat 70-90, value 58-75
        # corresponding opencv values are hue: 179 and 0-1, sat 178-230, val = 148-191
        self.red_center = (0, 0)
        self.red_radius = 0.0
        redLower1 = (0, 150, 100)
        redUpper1 = (5, 230, 191)
        redLower2 = (178, 150, 100)
        redUpper2 = (179, 230, 191)
        # redLower1 = (0, 100, 100)
        # redUpper1 = (2, 255, 255)
        # redLower2 = (177, 100, 100)
        # redUpper2 = (179, 255, 255)

        # hotzone gimp RGBA = 255, 173, 74, 255 HSV = 33, 72, 100
        # corresponding opencv values are hue 16.5, sat 184, val = 255
        hotzone_center = (0, 0)
        hotzone_radius = 0.0
        hotzoneLower = (16, 180, 250)
        hotzoneUpper = (17, 190, 255)
        hotzoneLower_relaxed = (13, 175, 245)
        hotzoneUpper_relaxed = (23, 200, 255)

        # arena walls gimp hue 29 - 33, sat 35 - 50, value 180 - 48
        # corresponding opencv values are hue: 14-17 , sat 90-128 val = 45-123
        # arena floor gimp hue 35 - 40, sat 34 - 40, value 47 - 78
        # corresponding opencv values are hue: 17-20, sat 87-102, val = 119-199

        # https: // mdcrosby.com / blog / animalaieval.html
        # by walls, I think he means obstacles that are walls, not the arena walls
        # Walls:Walls are used in two main ways. If they are used as barriers they will be grey RGB(153, 153, 153) HSV (0,0,63), opencv(0,0,153)
        # If they are used as platforms that cannot be climbed they will be blue RGB(0, 0, 255), HSV (240,0,0), opencv(120,0,0)
        # Ramps: Ramps will be purple RGB(255, 0, 255), HSV (300,100,100), opencv (150,255,255)
        # Tunnels: If they are used as barriers they will be grey RGB(153, 153, 153), HSV(0,0,63), opencv(0,0,153)

        floorLower = (17, 70, 100)
        floorUpper = (20, 120, 220)
        self.floor_mask = cv2.inRange(hsv, floorLower, floorUpper)
        self.floor_mask = cv2.erode(self.floor_mask, None, iterations=3)
        self.floor_mask = cv2.dilate(self.floor_mask, None, iterations=3)

        # floor_map = self.transform_floor_mask_to_map(self.floor_mask)
        #floor_map = self.transform_floor_mask_angles_to_map(self.floor_mask)
        # cv2.imshow("floor_mask", self.floor_mask)
        # cv2.imshow("Mask", floor_map)
        # print("floor mask:")
        # for vert in range(35, 45):
        #    for horiz in range(18, 22):
        #        print("x = ", horiz, ", rows up from the bottom = ", vert, ", rows down from the top = ", 83 - vert, ", value = ", self.floor_mask[83 - vert][horiz])

        laplacian = cv2.Laplacian(self.floor_mask, cv2.CV_8UC1)
        lines = cv2.HoughLines(laplacian, 1, np.pi / 180, 40)
        if lines is not None:
            for rho, theta in lines[0]:
                # print("rho, theta = ", rho, theta)
                a = np.cos(theta)
                b = np.sin(theta)
                self.floorX = int((a * rho) + 0.5) + 10
                self.floorY = int((b * rho) + 0.5)
                x1 = int((self.floorX - 30 * (-b)) + 0.5)
                y1 = int((self.floorY - 30 * (a)) + 0.5)
                if b != 0.0:
                    self.floor_slope = (y1 - self.floorY) / (x1 - self.floorX)
                else:
                    self.floor_slope = 1000.
                # cv2.line(bgr, (self.floorX, self.floorY), (x1, y1), (255, 255, 255), 2)
        else:
            self.floorX = 0
            self.floorY = 0
            self.floor_slope = 0.0

        self.floor_size = np.count_nonzero(self.floor_mask)
        row_sums = np.sum(self.floor_mask, axis=1)
        self.top_of_floor = self.max_pixel - 1
        for i in range(self.max_pixel):
            if float(row_sums[i]) / 255. > (self.max_pixel / 2):
                self.top_of_floor = i
                break
        # print("floor takes up {0:2d} rows from the bottom of the image".format(self.top_of_floor))
        # input("see edges and lines")

        # cv2.line(bgr, (0, self.top_of_floor), (83, self.top_of_floor), (255,0,0), 1)

        # if self.top_of_floor < 45:
        # input("see floor")

        wallLower = (14, 90, 45)
        wallUpper = (17, 128, 150)  # 123)
        wall_mask = cv2.inRange(hsv, wallLower, wallUpper)
        wall_in_view = -1
        if np.count_nonzero(wall_mask) > 3000 and self.floor_size < 2000:
            M = cv2.moments(wall_mask)
            if M["m00"] != 0.0:
                wall_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if wall_center[1] > 22:
                    if wall_center[0] < 40 or self.floor_slope < -0.7:
                        wall_in_view = 0  # wall is to the left
                        # cv2.circle(bgr, (5, 5), 5, (255, 255, 255), 2)
                    elif wall_center[0] > 44 or self.floor_slope > 0.7:
                        # cv2.circle(bgr, (79, 5), 5, (255, 255, 255), 2)
                        wall_in_view = 2  # wall is on the right
                    else:
                        # cv2.circle(bgr, (42, 5), 5, (255, 255, 255), 2)
                        wall_in_view = 1  # wall is in front
                else:
                    # wall is too far away to report
                    wall_in_view = -1

            # if wall_in_view == 0:
            # print("there is a wall to the left, floor size = ", self.floor_size, ", floor top = ", self.top_of_floor, "floorY = ", self.floorY, ", floor slope = ", self.floor_slope)
            # elif wall_in_view == 1:
            # print("there is a wall in front, floor size = ", self.floor_size, ", floor top = ", self.top_of_floor, "floorY = ", self.floorY, ", floor slope = ", self.floor_slope)
            # else:
            # print("there is a wall to the right, floor size = ", self.floor_size, ", floor top = ", self.top_of_floor, "floorY = ", self.floorY, ", floor slope = ", self.floor_slope)
            # if wall_in_view == 1:
            # if wall_center[1] > 38:
            # print("we are very close to the wall, floor size = ", self.floor_size, ", floor top = ", self.top_of_floor, "floorY = ", self.floorY, ", floor slope = ", self.floor_slope)
            # else:
            # print("we are between 2 and 8 blocks from the wall, floor size = ", self.floor_size, ", floor top = ", self.top_of_floor, "floorY = ", self.floorY, ", floor slope = ", self.floor_slope)

            # print("wall center = ", wall_center, "size = ", np.count_nonzero(wall_mask))
            # cv2.imshow("Mask", wall_mask)


        else:
            wall_in_view = -1
            # print("no wall in view")

        # if wall_in_view != -1:
        #    print("floorX = {0:2d}, floorY = {1:2d}, floor_slope = {2:.2f}".format(self.floorX, self.floorY, self.floor_slope))
        #    print("wall in view = {0:2d}, wall center x = {1:2d}, wall center y = {2:2d}".format(wall_in_view, wall_center[0], wall_center[1]))
        # input("see data")

        obstacleLower = (0, 0, 50)
        obstacleUpper = (0, 0, 70)
        obstacle_mask = cv2.inRange(hsv, obstacleLower, obstacleUpper)
        if np.count_nonzero(obstacle_mask) > 50:
            obstacle_in_view = True
        else:
            obstacle_in_view = False

        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        green_mask = cv2.inRange(hsv, greenLower, greenUpper)
        # green_mask = cv2.erode(green_mask, None, iterations=2)
        # green_mask = cv2.dilate(green_mask, None, iterations=2)
        # mask using relaxed filter
        green_mask_relaxed = cv2.inRange(hsv, greenLower_relaxed, greenUpper_relaxed)
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        
        
        green_target_info = self.find_green_target(green_mask, 'strict')
        
        
        
        if green_target_info['target_found']:
            green_target_info_relaxed = self.find_green_target(green_mask_relaxed, 'relaxed')
            self.get_target_size_and_draw(green_target_info, green_target_info_relaxed, bgr, 'green')
            #cv2.putText(bgr, str(self.green_size_index), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
            #            cv2.LINE_AA)
            #print("green target found, green target info relaxed = ", green_target_info_relaxed)
        else:
            self.green_size_index = 0
        '''
        M = cv2.moments(c)
        if M["m00"] == 0:
            #print('moments: ', M["m00"], M["m01"], M['m10'], M['m11'], ", radius = ", self.green_radius, ", contour length = ", len(c))
            if len(c) > 0:
                cx = 0
                cy = 0
                for p in c:
                    cx += p[0][0]
                    cy += p[0][1]
                cx = int(cx / len(c))
                cy = int(cy / len(c))
                self.green_center = (cx, cy)
                self.green_radius = 0.5
                #print("Green center = ", self.green_center)
                #input("observe")
            else:
                self.green_center = (int(x), int(y)) #(int(self.max_pixel / 2), int(self.max_pixel / 2))
                self.green_radius = 1e-04
        else:
            self.green_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        '''
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        gold_mask = cv2.inRange(hsv, goldLower, goldUpper)
        # gold_mask = cv2.erode(gold_mask, None, iterations=2)
        # gold_mask = cv2.dilate(gold_mask, None, iterations=2)
        gold_mask_relaxed = cv2.inRange(hsv, goldLower_relaxed, goldUpper_relaxed)
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        
        
        
        
        
        
        gold_target_info = self.find_gold_target(gold_mask, 'strict')
        
        
        
        
        
        
        
        if gold_target_info['target_found']:
            gold_target_info_relaxed = self.find_gold_target(gold_mask_relaxed, 'relaxed')
            self.get_target_size_and_draw(gold_target_info, gold_target_info_relaxed, bgr, 'gold')
        else:
            self.gold_size_index = 0
        '''
        M = cv2.moments(c)
        if M["m00"] != 0.0:
            self.gold_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            self.gold_center = (int(x), int(y)) #(int(self.max_pixel / 2), int(self.max_pixel / 2))
            self.gold_radius = 1e-04

        '''
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        red_mask1 = cv2.inRange(hsv, redLower1, redUpper1)
        red_mask2 = cv2.inRange(hsv, redLower2, redUpper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        # red_mask = cv2.erode(red_mask, None, iterations=2)
        # red_mask = cv2.dilate(red_mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        red_cnts = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        red_cnts = self.grab_contours(red_cnts)

        # only proceed if at least one contour was found
        red_in_view = False
        if len(red_cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(red_cnts, key=cv2.contourArea)
            ((x, y), self.red_radius) = cv2.minEnclosingCircle(c)
            self.red_center = (int(x), int(y))
            #self.red_radius = self.red_radius + 3
            # self.red_radius = np.count_nonzero(red_mask) / (self.max_pixel / 4)

            '''
            M = cv2.moments(c)
            if M["m00"] != 0.0:
                self.red_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                self.red_center = (int(x), int(y)) #(int(self.max_pixel / 2), int(self.max_pixel / 2))
                self.red_radius = 0.0
            '''

            # only proceed if the radius meets a minimum size
            if self.red_radius > 0.01:
                red_in_view = True
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(bgr, (int(x), int(y)), 5,
                           (255, 255, 255), 2)
                cv2.circle(bgr, self.red_center, 5, (0, 0, 255), -1)
                # target_found = True
                # print("target radius = {0:.2f}".format(radius))
                # print("target center = ", center)

        hotzone_mask = cv2.inRange(hsv, hotzoneLower, hotzoneUpper)
        # cv2.imshow("hotzone", hotzone_mask)
        # hotzone_mask = cv2.erode(hotzone_mask, None, iterations=2)
        # hotzone_mask = cv2.dilate(hotzone_mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball

        hotzone_cnts = cv2.findContours(hotzone_mask.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        hotzone_cnts = self.grab_contours(hotzone_cnts)

        # only proceed if at least one contour was found
        if len(hotzone_cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(hotzone_cnts, key=cv2.contourArea)
            ((x, y), hotzone_radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] != 0.0:
                hotzone_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                hotzone_center = (int(x), int(y))  # (int(self.max_pixel / 2), int(self.max_pixel / 2))
                hotzone_radius = 0.0

            hotzone_size = np.count_nonzero(hotzone_mask) / (self.max_pixel / 4)
            # print("hotzone size = ", hotzone_size, ", hotzone radius = ", hotzone_radius)
            hotzone_radius = hotzone_size

        # only proceed if the radius meets a minimum size
        if hotzone_radius > 0.01:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(bgr, (int(x), int(y)), 5,
                       (255, 255, 255), 2)
            cv2.circle(bgr, hotzone_center, 5, (0, 0, 255), -1)
            # target_found = True
            # print("target radius = {0:.2f}".format(radius))
            # print("target center = ", center)

        if self.watch_image:
            cv2.imshow("Frame", bgr)
            #cv2.imshow("mask", hotzone_mask_relaxed)
            #if hotzone_mask_relaxed_edge is not None:
            #    cv2.imshow("edged", hotzone_mask_relaxed_edge)
            cv2.waitKey(10)

        radius, size_index, centerX, centerY, color = self.process_targets(accept_green)
        # thought we might estimate target distance by its height in the image
        # but the center of the target is up off the floor, so as you approach it, the
        # center rises even though the whole target falls
        # if color != 0:
        #    height = self.max_pixel. - float(centerY)
        #    distance = height / (self.max_pixel / 2)
        #    print("radius = {0:.2f}, height = {1:.2f}, distance = {2:.2f}, ratio = {3:.2f}, color = {4:2d}".format(radius, height, distance, radius * distance, color))
        #    input("see target")
        return radius, size_index, centerX, centerY, hotzone_radius, hotzone_center, color, obstacle_in_view, wall_in_view, red_in_view, False

    def process_targets(self, accept_green):
        radius = 0.
        size_index = 0
        centerX = 0.
        centerY = 0.

        if self.green_radius < 1e-05 and self.red_radius < self.min_red_radius and self.gold_radius < 1e-05:
            color = 0
            # print("no target")

        # always avoid red
        elif self.red_radius >= self.min_red_radius and self.red_radius > self.green_radius and self.red_radius > self.gold_radius:
            color = 3
            # print("red radius = ", self.red_radius)
            radius = self.red_radius
            size_index = self.red_size_index
            centerX = self.red_center[0]
            centerY = self.red_center[1]
            # print("need to avoid red")

        # avoid green if we have lots of time
        # note that if there is a green target in view with accept_green = False when the first blackout happens
        # then the blackout will think there is no target in view, becuase we report color = 0 in that case
        elif self.green_radius >= self.min_green_radius and self.green_radius > self.gold_radius and (not accept_green):
            color = 1
            radius = self.green_radius
            size_index = self.green_size_index
            centerX = self.green_center[0]
            centerY = self.green_center[1]
            # print("need to avoid green")

        else:
            if self.gold_radius >= 1e-05:
                color = 2
                # print("gold radius = ", self.gold_radius)
                radius = self.gold_radius
                size_index = self.gold_size_index
                centerX = self.gold_center[0]
                centerY = self.gold_center[1]
            else:
                # if accept_green:
                # we want to report this even if we are not accepting green, since we may be in the looking around phase
                color = 1
                radius = self.green_radius
                size_index = self.green_size_index
                centerX = self.green_center[0]
                centerY = self.green_center[1]
                # else:
                #    #print("green target seen, but too many steps remaining for us to bother with it")
                #    color = 0

        return radius, size_index, centerX, centerY, color

    def transform_floor_mask_to_map(self, image):
        map = np.ones((40, 40), dtype=float)
        map *= 0.5
        pixel_values = np.zeros((40, 40), dtype=float)
        # for z in range(83, -1, -1): #image row 83 is at the bottom, row 0 is at the top
        for z in range(31):
            for x in range(84):
                # print("x, z, image: ", x, z, image[83-z][x])
                if z <= 11:
                    if x < 42:
                        pixel_values[39][19] += image[83 - z][x] / (42. * 12.)
                        # print("x, z, image, pixel_values: ", x, z, image[83 - z][x], pixel_values[39][19])
                    else:
                        pixel_values[39][20] += image[83 - z][x] / (42. * 12.)
                elif z <= 20:
                    if x < 10:
                        pixel_values[38][18] += image[83 - z][x] / (10. * 9.)
                    if x < 20:
                        pixel_values[38][19] += image[83 - z][x] / (10. * 9.)
                    elif x < 30:
                        pixel_values[38][20] += image[83 - z][x] / (10. * 9.)
                    else:
                        pixel_values[38][21] += image[83 - z][x] / (10. * 9.)
                elif z <= 25:
                    if x < 6:
                        pixel_values[37][17] += image[83 - z][x] / (6. * 5.)
                    if x < 13:
                        pixel_values[37][18] += image[83 - z][x] / (7. * 5.)
                    elif x < 20:
                        pixel_values[37][19] += image[83 - z][x] / (7 * 5.)
                    elif x < 27:
                        pixel_values[37][20] += image[83 - z][x] / (7. * 5.)
                    elif x < 34:
                        pixel_values[37][21] += image[83 - z][x] / (7. * 5.)
                    else:
                        pixel_values[37][22] += image[83 - z][x] / (6. * 5.)
                elif z <= 29:
                    if x < 5:
                        pixel_values[36][16] += image[83 - z][x] / (5. * 4.)
                    elif x < 10:
                        pixel_values[36][17] += image[83 - z][x] / (5. * 4.)
                    elif x < 15:
                        pixel_values[36][18] += image[83 - z][x] / (5. * 4.)
                    elif x < 20:
                        pixel_values[36][19] += image[83 - z][x] / (5. * 4.)
                    elif x < 25:
                        pixel_values[36][20] += image[83 - z][x] / (5. * 4.)
                    elif x < 30:
                        pixel_values[36][21] += image[83 - z][x] / (5. * 4.)
                    elif x < 35:
                        pixel_values[36][22] += image[83 - z][x] / (5. * 4.)
                    else:
                        pixel_values[36][23] += image[83 - z][x] / (5. * 4.)
                elif z <= 31:
                    if x < 4:
                        pixel_values[35][15] += image[83 - z][x] / (4. * 2.)
                    if x < 8:
                        pixel_values[35][16] += image[83 - z][x] / (4. * 2.)
                    if x < 12:
                        pixel_values[35][17] += image[83 - z][x] / (4. * 2.)
                    elif x < 16:
                        pixel_values[35][18] += image[83 - z][x] / (4. * 2.)
                    elif x < 20:
                        pixel_values[35][19] += image[83 - z][x] / (4. * 2.)
                    elif x < 24:
                        pixel_values[35][20] += image[83 - z][x] / (4. * 2.)
                    elif x < 28:
                        pixel_values[35][21] += image[83 - z][x] / (4. * 2.)
                    elif x < 32:
                        pixel_values[35][22] += image[83 - z][x] / (4. * 2.)
                    elif x < 36:
                        pixel_values[35][23] += image[83 - z][x] / (4. * 2.)
                    else:
                        pixel_values[35][24] += image[83 - z][x] / (4. * 2.)

                elif z == 32:
                    if x < 3:
                        pixel_values[34][13] += image[83 - z][x] / 3.
                        pixel_values[33][13] += image[83 - z][x] / 3.
                    if x < 6:
                        pixel_values[34][14] += image[83 - z][x] / 3.
                        pixel_values[33][14] += image[83 - z][x] / 3.
                    if x < 9:
                        pixel_values[34][15] += image[83 - z][x] / 3.
                        pixel_values[33][15] += image[83 - z][x] / 3.
                    if x < 12:
                        pixel_values[34][16] += image[83 - z][x] / 3.
                        pixel_values[33][16] += image[83 - z][x] / 3.
                    elif x < 15:
                        pixel_values[34][17] += image[83 - z][x] / 3.
                        pixel_values[33][17] += image[83 - z][x] / 3.
                    elif x < 18:
                        pixel_values[34][18] += image[83 - z][x] / 3.
                        pixel_values[33][18] += image[83 - z][x] / 3.
                    elif x < 21:
                        pixel_values[34][19] += image[83 - z][x] / 3.
                        pixel_values[33][19] += image[83 - z][x] / 3.
                    elif x < 24:
                        pixel_values[34][20] += image[83 - z][x] / 3.
                        pixel_values[33][20] += image[83 - z][x] / 3.
                    elif x < 27:
                        pixel_values[34][21] += image[83 - z][x] / 3.
                        pixel_values[33][21] += image[83 - z][x] / 3.
                    elif x < 30:
                        pixel_values[34][22] += image[83 - z][x] / 3.
                        pixel_values[33][22] += image[83 - z][x] / 3.
                    elif x < 33:
                        pixel_values[34][23] += image[83 - z][x] / 3.
                        pixel_values[33][23] += image[83 - z][x] / 3.
                    elif x < 36:
                        pixel_values[34][24] += image[83 - z][x] / 3.
                        pixel_values[33][24] += image[83 - z][x] / 3.
                    else:
                        pixel_values[34][25] += image[83 - z][x] / 3.
                        pixel_values[33][25] += image[83 - z][x] / 3.
                elif z == 33:
                    length = 9
                    if x < 3:
                        pixel_values[32][13] += image[83 - z][x] / 3.
                        pixel_values[31][13] += image[83 - z][x] / 3.
                    if x < 6:
                        pixel_values[32][14] += image[83 - z][x] / 3.
                        pixel_values[31][14] += image[83 - z][x] / 3.
                    if x < 9:
                        pixel_values[32][15] += image[83 - z][x] / 3.
                        pixel_values[31][15] += image[83 - z][x] / 3.
                    if x < 12:
                        pixel_values[32][16] += image[83 - z][x] / 3.
                        pixel_values[31][16] += image[83 - z][x] / 3.
                    elif x < 15:
                        pixel_values[32][17] += image[83 - z][x] / 3.
                        pixel_values[31][17] += image[83 - z][x] / 3.
                    elif x < 18:
                        pixel_values[32][18] += image[83 - z][x] / 3.
                        pixel_values[31][18] += image[83 - z][x] / 3.
                    elif x < 21:
                        pixel_values[32][19] += image[83 - z][x] / 3.
                        pixel_values[31][19] += image[83 - z][x] / 3.
                    elif x < 24:
                        pixel_values[32][20] += image[83 - z][x] / 3.
                        pixel_values[31][20] += image[83 - z][x] / 3.
                    elif x < 27:
                        pixel_values[32][21] += image[83 - z][x] / 3.
                        pixel_values[31][21] += image[83 - z][x] / 3.
                    elif x < 30:
                        pixel_values[32][22] += image[83 - z][x] / 3.
                        pixel_values[31][22] += image[83 - z][x] / 3.
                    elif x < 31:
                        pixel_values[32][23] += image[83 - z][x] / 3.
                        pixel_values[31][23] += image[83 - z][x] / 3.
                    elif x < 36:
                        pixel_values[32][24] += image[83 - z][x] / 3.
                        pixel_values[31][24] += image[83 - z][x] / 3.
                    else:
                        pixel_values[32][25] += image[83 - z][x] / 3.
                        pixel_values[31][25] += image[83 - z][x] / 3.

        for z in range(13, 26):
            for x in range(31, 40):
                if pixel_values[x][z] > 127.:  # if we see floor, mark the spot as unoccupied
                    # print("unoccupied coords, x, z = ", x, z)
                    map[x][z] = 0
        return map

    def transform_floor_mask_angles_to_map(self, image):
        map = np.ones((40, 40), dtype=float)
        map *= 0.5
        pixel_values = np.zeros((40, 40), dtype=float)
        field_of_view_ratio = .577  # this is the tan(30 degrees), since we have a +- 30 degree field of view
        # 84 columns of the image correspond to 60 degrees in the field of view
        # column_to_radian_factor = 60./(84. * 57.3)
        # for z in range(83, -1, -1): #image row 83 is at the bottom, row 0 is at the top

        for x in range(84):
            for z in range(43):
                num_rows = 1
                # calibrate the relationship of pixel position to distance in the arena
                if z <= 11:
                    start_length = 0  # distance from the agent that these z values in the image correspond to
                    end_length = 1
                    num_rows = 12
                    # x_offset = field_of_view_ratio * length
                    # if x < 42:
                    #    map_values[39][int((19.5 - x_offset) + 0.5)] += image[83-z][x] / (42. * 12.)
                    # else:
                    #    map_values[39][int((19.5 + x_offset) + 0.5)] += image[83-z][x] / (42. * 12.)
                elif z <= 20:
                    start_length = 1
                    end_length = 2
                    num_rows = 9
                    # x_offset = int((field_of_view_ratio * length) + 0.5)
                    # for i in range(x_offset):
                    #    if i == 0 and x < 42./ x_offset:
                    #        map_values[38][20 - i]
                    #    for j in range()
                    #    map_values[38][i] += image[83-z][x] / (42. * 9.)
                elif z <= 25:
                    start_length = 2
                    end_length = 3
                    num_rows = 5
                elif z <= 29:
                    start_length = 3
                    end_length = 4
                    num_rows = 4
                elif z == 31:
                    start_length = 4
                    end_length = 6
                    num_rows = 2
                elif z == 32:
                    start_length = 6
                    end_length = 8
                elif z == 33:
                    start_length = 8
                    end_length = 10
                elif z == 34:
                    start_length = 10
                    end_length = 13
                elif z == 35:
                    start_length = 13
                    end_length = 17
                elif z == 36:
                    start_length = 17
                    end_length = 21
                elif z == 37:
                    start_length = 21
                    end_length = 25
                elif z == 38:
                    start_length = 25
                    end_length = 29
                elif z == 39:
                    start_length = 29
                    end_length = 34
                elif z <= 41:
                    start_length = 34
                    end_length = 40
                    num_rows = 3  # suppress these last two a little

                x_offset = int((field_of_view_ratio * end_length) + 0.5)
                x_threshold = int((84. / (x_offset * 2)) + 0.5)
                num_x_thresholds = int(((84. / x_threshold) - 1) + 0.5)
                if num_x_thresholds > 39:
                    num_x_thresholds = 39
                    x_threshold = 2.15  # this is (84/78) * 2
                for i in range(num_x_thresholds + 1):
                    if x > i * x_threshold and x <= (i + 1) * x_threshold:
                        # pixel_values[34][19 - i] += image[83 - z][x] / x_threshold
                        # pixel_values[33][19 + i] += image[83 - z][x] / x_threshold
                        for j in range(start_length, end_length):
                            pixel_values[39 - j][int(19.5 - (num_x_thresholds / 2)) + i] += image[83 - z][x] / (
                                        x_threshold * num_rows)
                            # if z == 37:
                            #    print("i = ", i, ", j= ", j, ", x = ", x, ", i*threshold = ", i * x_threshold, ", int(19.5 - (num_x_thresholds / 2)) + i = ", int(19.5 - (num_x_thresholds / 2)) + i, ", value = ", pixel_values[39 - j][int(19.5 - (num_x_thresholds / 2)) + i])

        # note that these x and z correspond to map coordinates.  The x and z above correspond to image coordinates
        for x in range(40):
            for z in range(40):
                if pixel_values[x][z] > 127.:  # if we see floor, mark the spot as unoccupied
                    # print("unoccupied coords, x, z = ", x, z)
                    map[x][z] = 0
        return map

    def transform_image_pixel_to_map_floor_point(self, image_x, image_z):
        field_of_view_ratio = .577  # this is the tan(30 degrees), since we have a +- 30 degree field of view
        map_x = 0
        map_z = 0
        if image_z <= 11:
            start_length = 0  # distance from the agent that these z values in the image correspond to
            end_length = 1
        elif image_z <= 20:
            start_length = 1
            end_length = 2
        elif image_z <= 25:
            start_length = 2
            end_length = 3
        elif image_z <= 29:
            start_length = 3
            end_length = 4
        elif image_z == 31:
            start_length = 4
            end_length = 6
        elif image_z == 32:
            start_length = 6
            end_length = 8
        elif image_z == 33:
            start_length = 8
            end_length = 10
        elif image_z == 34:
            start_length = 10
            end_length = 13
        elif image_z == 35:
            start_length = 13
            end_length = 17
        elif image_z == 36:
            start_length = 17
            end_length = 21
        elif image_z == 37:
            start_length = 21
            end_length = 25
        elif image_z == 38:
            start_length = 25
            end_length = 29
        elif image_z == 39:
            start_length = 29
            end_length = 34
        elif image_z <= 45:
            start_length = 34
            end_length = 40
        else:
            print("unable to transform_image_pixel_to_map_floor_point, image_z = ", image_z)
            return 0, 0

        map_z = 39 - int(39.5 - ((start_length + end_length) / 2.))
        if map_z > 39:
            map_z = 39
        elif map_z < 0:
            map_z = 0
        x_offset = int((field_of_view_ratio * end_length) + 0.5)
        x_threshold = int((84. / (x_offset * 2)) + 0.5)
        num_x_thresholds = int(((84. / x_threshold) - 1) + 0.5)
        print("x_offset = ", x_offset, ", x_threshold = ", x_threshold, ", num_x_thresholds = ", num_x_thresholds)
        if num_x_thresholds > 39:
            num_x_thresholds = 39
            x_threshold = 2.15  # this is (84/78) * 2
        for i in range(num_x_thresholds + 1):
            # print("i = ", i, ", image_x = ", image_x, ", i*threshold = ", i * x_threshold, ", int(19.5 - (num_x_thresholds / 2)) + i = ", int(19.5 - (num_x_thresholds / 2)) + i)
            if image_x > i * x_threshold and image_x <= (i + 1) * x_threshold:
                map_x = int(19.5 - (num_x_thresholds / 2)) + i
                print("found a map_x = ", map_x)
                break
        if map_x > 39:
            map_x = 39
        elif map_x < 0:
            map_x = 0
        return map_x, map_z
