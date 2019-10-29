#making a map from animalai observations
#velocities are in meters/sec, map units are meters
#velocties are in the agent's frame of reference, not the map frame of reference

#looks like the sim puts in yaw after acceleration when doing diagonal actions
#that is based on the observation that we get an x velocity (not just z) when doing a diagonal action from rest


from animalai.envs.environment import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig
import time
import random
import numpy as np
import time
import cv2
import math

#ARENA_SIZE = 40
#DELTA_TIMESTEP = 0.0595 #this value is from the issues page
#DELTA_TIMESTEP = 0.0606 #this value was determined locally
#SHOW_MAP = True

Z_MAX = 37 #max number of rows to use in the image, starting immediately in front of the agent

USER_INPUT = False
if USER_INPUT:
    from process_image_working import ProcessImage

def init_environment():
    return UnityEnvironment(
        file_name='../test_submission/env/AnimalAI',          #Path to the environment
        worker_id=random.randint(1, 100),   #Unique ID for running the environment (used for connection)
        seed=10,                     #The random seed
        docker_training=False,      #Whether or not you are training inside a docker
        #no_graphics=False,          #Always set to False
        n_arenas=1,                 #Number of arenas in your environment
        play=False,                 #Set to False for training
        inference=True            #Set to true to watch your agent in action
        #resolution=None             #Int: resolution of the agent's square camera (in [4,512], default 84)
    )

#making a map from animalai observations
#velocities are in meters/sec, map units are meters
#velocties are in the agent's frame of reference, not the map frame of reference

#distance calculation is done with:
#delta_distance = 0.0595 * speed[0, 2]
#note that the agent is a sphere with radius = 0.5, so when going from one end to the other of the arena
#it goes from 0.5 to 39.5
#see: https://github.com/beyretb/AnimalAI-Olympics/issues/43

# in the search image, the open list is in green and the closed list is in gold

# our astar search tree has nodes, each node has
# full cost, cost from start node, underestimated cost to goal (the end node)
# for underestimated cost to goal, we will just use the euclidean distance from the node to the goal,
# it is a reasonable estimate and it will never overestimate
# position = location on the map of the node
# parent is the node's parent node
class Node():
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.cost_from_start = 0
        self.estimated_cost_to_goal = 0
        self.full_cost = 0

    def __eq__(self, other):
        return self.position == other.position
        
class Mapper:
    def __init__(self, arena_size, delta_timestep, use_occupancy_map = False, use_target_maps = False, show_map = True):
        self.image_size = 84
        self.arena_size = arena_size
        self.delta_timestep = delta_timestep
        self.use_occupancy_map = use_occupancy_map
        self.use_target_maps = use_target_maps
        self.display_map = show_map
        self.user_input = USER_INPUT
        if self.user_input:
            self.process_image = ProcessImage(84, 20.0, 13.0, True)
            self.frame = None
        self.occupancy_map_window_created = False
        self.path_map_window_created = False
        self.target_map_window_created = False
        self.astar_map_window_created = False
        self.astar_search_progress_window_created = False
        self.exploration_window_created = False
        self.green_seen_window_created = False
        self.z_max = Z_MAX       
        
        #focalLength = (picWidth * knownDistance) / knownWidth
        #our field of view is 60 degrees.  For 30 degrees, at 40 meters, we see tan(30/57.3) * 40 = 0.5773 * 40 = 23.1 meters horizontally
        #so our knownWidth = 46.2 meters, knownDistance is 40 meters, and picture width is 84 pixels
        #so the focal length = (84 * 40)/46.1 = 72.885
        # alternative method: camera focal length = (0.5 * image_width) / tan(0.5 * fov_radians)
        # = (0.5 * self.arena_size) / tan(0.5 * (60/57.3) = 42 / tan(30/57.3) = 42. / .5774 = 72.74
        #self.focal_length = (0.5 * self.image_size) / .5773  # = tan(0.5 * (60/57.3)
        self.focal_length = 72.8
        #self.base_location = np.array([0.0, 0.0, self.focal_length])
        #np.linalg computes the magnitude of a vector
        #self.base_location_magnitude = np.linalg.norm(self.base_location)
        #print("self.focal_length = ", self.focal_length, ", self.base_location = ", self.base_location, ", base_location_magnitude", self.base_location_magnitude)
        self.reset()

    def __del__(self):
        cv2.destroyAllWindows()
        '''
        if self.occupancy_map_window_created:
            cv2.destroyWindow('Occupancy_map')
        if self.path_map_window_created:
            cv2.destroyWindow('Path_map')
        if self.target_map_window_created:
            cv2.destroyWindow('Target_map')
        '''

    def reset(self):
        self.occupancy_map = np.ones((self.arena_size * 2, self.arena_size * 2))
        self.occupancy_map *= 0.5
        #for A* we need some boundaries
        for i in range(self.arena_size * 2):
            self.occupancy_map[i][0] = 1
            self.occupancy_map[i][(self.arena_size * 2) - 1] = 1
            self.occupancy_map[0][i] = 1
            self.occupancy_map[(self.arena_size * 2) - 1][i] = 1           
        self.path_map = np.ones((self.arena_size * 2, self.arena_size * 2))
        self.path_map *= 0.5
        if self.use_target_maps:
            self.target_map = np.zeros((self.arena_size * 2, self.arena_size * 2, 3))
            self.green_target_map = np.ones((self.arena_size * 2, self.arena_size * 2))
            self.green_target_map *= 0.5
            self.gold_target_map = np.ones((self.arena_size * 2, self.arena_size * 2))
            self.gold_target_map *= 0.5
            self.red_target_map = np.ones((self.arena_size * 2, self.arena_size * 2))
            self.red_target_map *= 0.5
            self.green_seen_map = np.zeros((self.arena_size * 2, self.arena_size * 2))
        self.max_map_row_floor = np.zeros(self.arena_size)
        self.origin_x = float(self.arena_size * 2) / 2.0
        self.origin_z = float(self.arena_size * 2) / 2.0
        self.x = float(self.arena_size * 2) / 2.0
        self.z = float(self.arena_size * 2) / 2.0
        self.yaw = 0.0
        self.previous_x = float(self.arena_size * 2) / 2.0
        self.previous_z = float(self.arena_size * 2) / 2.0
        self.previous_yaw = 0.0
        self.previous_heading = 0.0
        self.previous_vel_x = 0.0
        self.previous_vel_z = 0.0
        self.wall_points = []
        self.previous_wall = False
        self.target_list = []
        self.green_occupancy_list = []
        self.gold_occupancy_list = []
        self.red_occupancy_list = []
        self.green_seen_locations = []
        self.facing_perpendicular_to_wall = False
        self.long_wall = False

    def add_green_seen(self):       
        self.green_seen_locations.append([self.x, self.z, self.yaw])
        print("green seen from: ", len(self.green_seen_locations), " locations.") #  They are : ", self.green_seen_locations)
        if self.display_map and self.use_target_maps:
            if not self.green_seen_window_created:
                self.green_seen_window_created = True
                cv2.namedWindow('Green_seen', cv2.WINDOW_NORMAL)
            if int(self.z + 0.5) > 0 and int(self.z + 0.5) < len(self.green_seen_map) and int(self.x + 0.5) > 0 and int(self.x + 0.5) < len(self.green_seen_map):
                self.green_seen_map[int(self.z + 0.5)][int(self.x + 0.5)] = 1
            if self.display_map:
                cv2.imshow("Green_seen", self.green_seen_map)
        
    def update(self, velocities, action, floor_mask, blackout, wall_in_view):                
        delta_yaw = 0.0
        if action[1] == 1:
            delta_yaw = -0.1047  # 6 degrees cw
        elif action[1] == 2:
            delta_yaw = 0.1047  # 6 degrees ccw
        self.yaw = self.previous_yaw + delta_yaw

        vel_x = velocities[0]
        vel_z = velocities[2]
        #deltaT = time.time() - self.previous_time
        #deltaT = DELTA_TIMESTEP
        #self.previous_time = time.time()
        #need to convert from agent reference frame to arena reference frame
        #we use negative vel_z because opencv increases index values as you move from top to bottom
        #but the arena increases index values as you move bottom to top
        arena_vel_x = (vel_x * math.cos(self.previous_yaw)) + (-vel_z * math.sin(self.yaw))
        arena_vel_z = (-vel_z * math.cos(self.previous_yaw)) + (vel_x * math.sin(self.yaw))
        #print("vel_x = {0:.2f}, vel_z = {1:.2f}, arena vel_x = {2:.2f}, arena vel_z = {3:.2f}".format(vel_x, vel_z, arena_vel_x, arena_vel_z))

        delta_x = ((self.previous_vel_x + arena_vel_x) / 2.0) * self.delta_timestep
        delta_z = ((self.previous_vel_z + arena_vel_z) / 2.0) * self.delta_timestep
        #delta_distance = ((velocities[2] + previous_velocity) / 2.0) * 0.0606
        #use converted coordinates to calculate the move
        self.x = self.previous_x + delta_x
        self.z = self.previous_z + delta_z
        
        #delta_distance = math.sqrt((delta_x * delta_x) + (delta_z*delta_z))
        #total_distance_from_origin = math.sqrt(((self.x - self.origin_x)*(self.x - self.origin_x)) + ((self.z - self.origin_z)*(self.z - self.origin_z)))
        #yaw_degrees = yaw * 57.3
        #print("arena vel x = {0:.4f}, arena vel z = {1:.4f}, yaw = {2:.4f}, delta_distance = {3:.4f}, total_distance from origin = {4:.4f}, x = {5:.2f}, z = {6:.2f}".format(
        #    arena_vel_x, arena_vel_z, self.yaw, delta_distance, total_distance_from_origin, self.x, self.z))

        #if abs(vel_x) < 0.01 and abs(vel_z) < 0.01:
        #    if abs(self.previous_x) > 2.0 or abs(self.previous_z) > 2.0 or action[0] != 0:
                #we found a wall (or this is the first point)
                #the matrix is (rows, cols) while the image has x has the horizontal axis (cols)
                #so we have to swap the indicies if we want to plot this on the image directly
        #        self.wall_points.append((int(self.z + 0.5), int(self.x + 0.5)))

        if self.user_input:
            radius, centerX, centerY, hotzone_radius, hotzone_center, color, obstacle_in_view, wall_in_view, red_in_view, blackout = self.process_image.find_targets(self.frame, True)
            floor_mask = self.process_image.floor_mask
        
            if self.process_image.green_radius > 1e-05:
                #print("image green center = ", self.process_image.green_center[0], " pixels from the left and ", self.process_image.green_center[1], " rows from the top. radius = ", self.process_image.green_radius)
                if self.use_target_maps:
                    self.update_target_map(self.process_image.green_center[0], self.process_image.green_center[1], self.process_image.green_radius, 1)
                else:
                    self.update_target_list(self.process_image.green_center[0], self.process_image.green_center[1], self.process_image.green_radius, 1)
            if self.process_image.gold_radius > 1e-05:
                if self.use_target_maps:
                    self.update_target_map(self.process_image.gold_center[0], self.process_image.gold_center[1], self.process_image.gold_radius, 2)
                else:
                    self.update_target_list(self.process_image.gold_center[0], self.process_image.gold_center[1], self.process_image.gold_radius, 2)
            if self.process_image.red_radius > 1e-05:
                if self.use_target_maps:
                    self.update_target_map(self.process_image.red_center[0], self.process_image.red_center[1], self.process_image.red_radius, 3)
                else:
                    self.update_target_list(self.process_image.red_center[0], self.process_image.red_center[1], self.process_image.red_radius, 3)
        #add 0.5 because the int() function truncates the float
        if not blackout and self.use_occupancy_map:
            self.update_occupancy_map(int(self.x + 0.5), int(self.previous_x + 0.5), int(self.z + 0.5), int(self.previous_z + 0.5), floor_mask, wall_in_view)
        self.update_path_map(int(self.x + 0.5), int(self.previous_x + 0.5), int(self.z + 0.5), int(self.previous_z + 0.5))

        print("current map location: x = {0:.2f}, z = {1:.2f}, yaw degrees = {2:.1f}, action = [{3:.0f}, {4:.0f}]".format(self.x, self.z, self.yaw * 57.3, action[0], action[1]))
        if self.x < 0.:
            self.x = 0.
        elif self.x > (self.arena_size * 2) - 1:
            x = (self.arena_size * 2) - 1
        self.previous_x = self.x
        self.previous_z = self.z
        if self.yaw >= 6.28318:
            self.yaw -= 6.28318
        elif self.yaw <= -6.28318:
            self.yaw += 6.28318
        self.previous_yaw = self.yaw
        self.previous_vel_x = arena_vel_x
        self.previous_vel_z = arena_vel_z
        if self.display_map:
            if self.use_occupancy_map:
                self.show_occupancy_map()
            self.show_path_map()

    def update_occupancy_map(self, x, previous_x, z, previous_z, floor_mask, wall_in_view):
        self.max_map_row_floor = np.zeros(self.arena_size)
        #cv2.imshow("floor_mask", floor_mask)
        #fill in the floor points that we can see as unoccupied
        floor_points = self.transform_floor_mask_angles_to_map(floor_mask, wall_in_view)
        #print("floor points size = ", floor_points.shape[:2])
        #cv2.namedWindow('Floor_points', cv2.WINDOW_NORMAL)
        #cv2.imshow("Floor_points", floor_points)
        floor_points_translated_to_center = self.translate_map_to_center(floor_points)
        
        #cv2.imshow("translated to center", floor_points_translated_to_center)
        #print("floor points translated size = ", floor_points_translated.shape[:2])
        #floor_points is oriented directly in front of the agent
        #transform it into the arena frame
        floor_points_rotated = self.rotate_bound(floor_points_translated_to_center, -self.yaw)
        #cv2.imshow("translated and rotated to big map", floor_points_rotated)
        (h, w) = floor_points_rotated.shape[:2]
        #print("floor points rotated size = ", floor_points_rotated.shape[:2])
        border = int(((w - (self.arena_size * 2)) / 2.) - 0.5)
        if border > 0:
            floor_points_cropped = floor_points_rotated[border:(self.arena_size * 2) + border, border:(self.arena_size * 2) + border]
        else:
            floor_points_cropped = floor_points_rotated
        #print("floor points cropped size = ", floor_points_cropped.shape[:2])
        #cv2.imshow("cropped", floor_points_cropped)               
        #print("border = ", border)
        floor_points_transformed = self.translate_map(floor_points_cropped)
        #print("floor points last translation size = ", floor_points_transformed.shape[:2])
        #cleanup the borders.  we set the unoccupied locations to be 0.001, so we can just look for locations that are 0
        #turns out that after cropping, there is a tiny border of intermediate values remaining that show up
        #as a light black box in floor_points_transformed.  Those values are almost all between .001 and 0.5, so we can 
        #get rid of them in the same step
        for i in range(self.arena_size * 2):
            for j in range(self.arena_size * 2):
                if floor_points_transformed[i][j] == 0 or (floor_points_transformed[i][j] > 0.001 and floor_points_transformed[i][j] < 0.9):
                    floor_points_transformed[i][j] = 0.5
        #cv2.imshow("floor_points_transformed", floor_points_transformed)
        for i in range(self.arena_size * 2):
            for j in range(self.arena_size * 2):
                #points where we saw a clear floor will reduce the occupancy map value by 0.1
                if floor_points_transformed[i][j] <= 0.01:
                    self.occupancy_map[i][j] -= 0.1
                    if self.use_target_maps:
                        self.green_target_map[i][j] -= .1
                        self.gold_target_map[i][j] -= .1
                        self.red_target_map[i][j] -= .1
                    if self.occupancy_map[i][j] < 0:
                        self.occupancy_map[i][j] = 0
                    if self.use_target_maps:
                        if self.green_target_map[i][j] < 0:
                            self.green_target_map[i][j] = 0
                        if self.gold_target_map[i][j] < 0:
                            self.gold_target_map[i][j] = 0
                        if self.red_target_map[i][j] < 0:
                            self.red_target_map[i][j] = 0
                elif floor_points_transformed[i][j] >= 0.9:
                    if self.long_wall:
                        self.occupancy_map[i][j] = 1.0
                    else:
                        self.occupancy_map[i][j] += 0.1
                    if self.use_target_maps:
                        self.green_target_map[i][j] -= .1
                        self.gold_target_map[i][j] -= .1
                        self.red_target_map[i][j] -= .1
                    if self.occupancy_map[i][j] > 1.0:
                        self.occupancy_map[i][j] = 1.0
                    if self.use_target_maps:
                        if self.green_target_map[i][j] > 1.0:
                            self.green_target_map[i][j] = 1.0
                        if self.gold_target_map[i][j] > 1.0:
                            self.gold_target_map[i][j] = 1.0
                        if self.red_target_map[i][j] > 1.0:
                            self.red_target_map[i][j] = 1.0
                #the occupancy map decays toward 0.5 over time
                if self.occupancy_map[i][j] > 0.5:
                    self.occupancy_map[i][j] -= 0.0001
                elif self.occupancy_map[i][j] < 0.5:
                    self.occupancy_map[i][j] += 0.0001
                if self.use_target_maps:
                    if self.green_target_map[i][j] > 0.5:
                        self.green_target_map[i][j] -= 0.001
                    elif self.green_target_map[i][j] < 0.5:
                        self.green_target_map[i][j] += 0.001
                    if self.gold_target_map[i][j] > 0.5:
                        self.gold_target_map[i][j] -= 0.001
                    elif self.gold_target_map[i][j] < 0.5:
                        self.gold_target_map[i][j] += 0.001
                    if self.red_target_map[i][j] > 0.5:
                        self.red_target_map[i][j] -= 0.001
                    elif self.red_target_map[i][j] < 0.5:
                        self.red_target_map[i][j] += 0.001

        self.long_wall = False
        #find all the points between our previous position and our current position
        cv2.line(self.occupancy_map, (previous_x, previous_z), (x, z), (0, 0, 0), 1)
        #print("\ncurrent location: x = {0:.2f}, z = {1:.2f}, yaw degrees = {2:.1f}".format(self.x, self.z, self.yaw * 57.3))

    def get_current_location(self):
        return (self.x, self.z, self.yaw)

    def show_occupancy_map(self):
        if self.display_map and not self.occupancy_map_window_created:
            cv2.namedWindow('Occupancy_map', cv2.WINDOW_NORMAL)
            self.occupancy_map_window_created = True
        arrow_length = 10.0
        show_map = self.occupancy_map.copy()
        current_location = (int(self.previous_x + 0.5), int(self.previous_z + 0.5))
        #opencv positive angle goes cw but our convention is for positive angle to go ccw
        #so we use negative angle for the display (sin only, since cos does not change with sign)
        yaw_arrow_end_x = int((self.previous_x + arrow_length * math.sin(-self.previous_yaw)) + 0.5)
        yaw_arrow_end_z = int((self.previous_z - arrow_length * math.cos(self.previous_yaw)) + 0.5)
        yaw_arrow_end = (yaw_arrow_end_x, yaw_arrow_end_z)
        show_map = cv2.arrowedLine(show_map, current_location, yaw_arrow_end, 255)
        for i in self.wall_points:
            if i[0] <= (self.arena_size * 2) - 1 and i[0] >= 0 and i[1] <= (self.arena_size * 2) - 1 and i[1] >= 0:
                show_map[i[0], i[1]] = 1.0
        if self.display_map and self.use_occupancy_map:
            cv2.imshow('Occupancy_map', show_map)
            cv2.waitKey(10)

    def update_path_map(self, x, previous_x, z, previous_z):
        # find all the points between our previous position and our current position
        cv2.line(self.path_map, (previous_x, previous_z), (x, z), (0, 0, 0), 1)
        
    def show_path_map(self):
        if (not self.path_map_window_created) and self.display_map:
            cv2.namedWindow('Path_map', cv2.WINDOW_NORMAL)
            self.path_map_window_created = True
        arrow_length = 10.0
        show_map = self.path_map.copy()
        current_location = (int(self.previous_x + 0.5), int(self.previous_z + 0.5))
        #opencv positive angle goes cw but our convention is for positive angle to go ccw
        #so we use negative angle for the display (sin only, since cos does not change with sign)
        yaw_arrow_end_x = int((self.previous_x + arrow_length * math.sin(-self.previous_yaw)) + 0.5)
        yaw_arrow_end_z = int((self.previous_z - arrow_length * math.cos(self.previous_yaw)) + 0.5)
        yaw_arrow_end = (yaw_arrow_end_x, yaw_arrow_end_z)
        show_map = cv2.arrowedLine(show_map, current_location, yaw_arrow_end, 255)
        for i in self.wall_points:
            if i[0] <= (self.arena_size * 2) - 1 and i[0] >= 0 and i[1] <= (self.arena_size * 2) - 1 and i[1] >= 0:
                show_map[i[0], i[1]] = 1.0
        if self.display_map:
            cv2.imshow('Path_map', show_map)
            cv2.waitKey(10)

    def update_target_list(self, center_x, center_z, radius, color):
        # temp_map = np.zeros((self.arena_size * 2, self.arena_size * 2, 3))
        # circle_radius compensates for the difference between the center of the target
        # and the bottom of the target.  
        # As the agent moves toward the target, even though the target is getting closer
        # and the bottom of the target is moving down in the image, the center
        # of the target moves up in the image because the apparent radius is getting larger

        # a second correction is needed because the target is raised up in the air and, when it
        # gets close to the agent, the image cannot see its bottom

        circle_radius = int(radius + 0.5)
        if circle_radius < 1:
            circle_radius = 1
        # a circle_radius value of +6 above the radius works well when the target is less than 10m from the agent
        if 83 - center_z > 55:
            circle_radius += 2
        # as the target moves farther away, the effect diminishes, so we can increase circle_radius by less
        elif 83 - center_z > 48:
            circle_radius += 1
        # print("before limits, circle_radius = ", circle_radius, ", 83 - (center_z + circle_radius) = ", 83 - (center_z + circle_radius))
        # for large radius targets, circle_radius can get too big
        

        # print("center_z = ", center_z, ", 83 - center_z = ", 83 - center_z, ", radius = ", radius, ", circle_radius = ", circle_radius, ", 83 - (center_z + circle_radius) = ", 83 - (center_z + circle_radius))
        # map_x is in the range 0 to 39 and represents where the target is in the agent frame, from far left (0) to far right (39)
        # map_z is in the range 0 to 39 and represents where the target is in the agent frame, from immediately in front (0) to far away (39)
        # 83 - (center_z + circle_radius) is how far the bottom of the target is from the agent, in the image (not the map!)
        map_x, map_z = self.transform_image_pixel_to_map_floor_point(center_x, 83 - (center_z + circle_radius))
        if map_x == 0 and map_z == 0:
            print("image could not be interpreted for determining target location, skipping this one")
            return
        # print("center_x (image columns from the left) = ", center_x, ", center_z (image rows from the top) = ", center_z, ", radius = ", radius, ", circle_radius = ", circle_radius)

        # print("The bottom of the target is located in front of the agent by this many image rows: ", 83 - (center_z + circle_radius), ", circle_radius = ", circle_radius)
        # print("map_x, (map columns from the left in the agent frame): ", map_x, ", map_z (map rows from the agent) = ", map_z)

        # translate the point to correspond to agent location:
        # in the horizontal direction, the zero offset (center value) point of map_x is self.arena_size / 2, so we need to subtract that out
        # For example, if map_x = 20, then we do not want any x offset from self.x, so if self.x = 40, target_x should also = 40
        target_x = int((map_x - (self.arena_size / 2)) + self.x + 0.5)
        # in the vertical direction, the zero offset point just 0, so we do not need to subtract out any fixed value
        # however, since self.z is the distance from the top of the image, we need to subtract the value of map_z
        # from self.z to properly place the target in the image
        target_z = int(self.z + 0.5 - map_z)
        if target_x < 0:
            target_x = 0
        elif target_x > (self.arena_size * 2) - 1:
            target_x = (self.arena_size * 2) - 1
        if target_z < 0:
            target_z = 0
        elif target_z > (self.arena_size * 2) - 1:
            target_z = (self.arena_size * 2) - 1

        map_target_x, map_target_z = self.rotate_point((self.x, self.z), (target_x, target_z), -self.yaw)
        if map_target_x < 0 or map_target_x > (self.arena_size * 2) - 1 or map_target_z < 0 or map_target_z > (self.arena_size * 2) - 1:
            print("at least one map_target value is out of bounds: map_target_x = ", map_target_x, ", map_target_z", map_target_z)
            if map_target_x < 0:
                map_target_x = 0
            elif map_target_x > (self.arena_size * 2.) - 1:
                map_target_x = (self.arena_size * 2.) - 1
            if map_target_z < 0:
                map_target_z = 0
            elif map_target_z > (self.arena_size * 2.) - 1:
                map_target_z = (self.arena_size * 2.) - 1
        target_center_arena_frame_x = int(map_target_x + 0.5)
        target_center_arena_frame_z = int(map_target_z + 0.5)
        distance = np.sqrt(((map_x - (self.arena_size / 2.)) * (map_x - (self.arena_size / 2.))) + (map_z * map_z))
        target_diameter = (circle_radius * distance) / 40.  # at a distance of 39 meters, radius = 5; at 4 meters, radius = 39 for a size 5 target

        # print("self.x (agent columns from the left in map frame = ", self.x, ", self.z (agent rows from the top in map frame) = ", self.z)
        # print("x_target (map columns from the left in map frame = ", target_x, ", target_z (map rows from the top in the map frame = ", target_z)
        # print("map_target_x = ", map_target_x, ", map_target_z = ", map_target_z, ", target radius = ", target_diameter / 2)

        if target_center_arena_frame_x < 0:
            target_center_arena_frame_x = 0
        elif target_center_arena_frame_x > (self.arena_size * 2) - 1:
            target_center_arena_frame_x = (self.arena_size * 2) - 1
        if target_center_arena_frame_z < 0:
            target_center_arena_frame_z = 0
        elif target_center_arena_frame_z > (self.arena_size * 2) - 1:
            target_center_arena_frame_z = (self.arena_size * 2) - 1

        new_target = True
        if len(self.target_list) > 0:
            print("target list: x, z, diameter, color, number of times seen")
        for i in range(len(self.target_list)):
            if (self.target_list[i][0] - 10 < target_center_arena_frame_x and self.target_list[i][0] + 10 > target_center_arena_frame_x
                    and self.target_list[i][1] - 10 < target_center_arena_frame_z and self.target_list[i][1] + 10 > target_center_arena_frame_z
                    and self.target_list[i][3] == color):
                # there is a previous entry for this target
                # print("this target was seen ", self.target_list[i][4], " times before" )
                self.target_list[i][0] = ((self.target_list[i][0] * self.target_list[i][4]) + target_center_arena_frame_x) / (self.target_list[i][4] + 1)
                self.target_list[i][1] = ((self.target_list[i][1] * self.target_list[i][4]) + target_center_arena_frame_z) / (self.target_list[i][4] + 1)
                self.target_list[i][2] = ((self.target_list[i][2] * self.target_list[i][4]) + target_diameter) / (self.target_list[i][4] + 1)
                self.target_list[i][4] += 1
                new_target = False

            print("{0:.1f}, {1:.1f}, {2:.1f}, {3:1d}, {4:2d}".format(self.target_list[i][0], self.target_list[i][1], self.target_list[i][2], self.target_list[i][3], self.target_list[i][4]))

        if new_target:
            self.target_list.append([target_center_arena_frame_x, target_center_arena_frame_z, target_diameter, color, 1])
            print("new target found, x, z, diameter, color, number of times seen:")
            print("{0:.1f}, {1:.1f}, {2:.1f}, {3:1d}, {4:2d}".format(self.target_list[-1][0], self.target_list[-1][1], self.target_list[-1][2], self.target_list[-1][3], self.target_list[-1][4]))

        # print("target location: x = {0:2d}, z = {1:2d}, 83 - z = {2:2d}, target diameter = {3:.1f}, color = {4:1d}".format(
        #    target_center_arena_frame_x, target_center_arena_frame_z, 83 - target_center_arena_frame_z, target_diameter, color))

        # combine targets that have become close due to averaging
        for i in range(len(self.target_list)):
            j = i + 1
            while j < len(self.target_list):
                if (self.target_list[i][0] - 10 < self.target_list[j][0] and self.target_list[i][0] + 10 > self.target_list[j][0]
                        and self.target_list[i][1] - 10 < self.target_list[j][1] and self.target_list[i][1] + 10 > self.target_list[j][1]
                        and self.target_list[i][3] == self.target_list[j][3]):
                    # these two entries are close enough to be combined.  They converged through averaging values
                    # print("this target was seen ", self.target_list[i][4], " times before" )
                    self.target_list[i][0] = ((self.target_list[i][0] * self.target_list[i][4]) + (self.target_list[j][0] * self.target_list[j][4])) / (self.target_list[i][4] + self.target_list[j][4])
                    self.target_list[i][1] = ((self.target_list[i][1] * self.target_list[i][4]) + (self.target_list[j][1] * self.target_list[j][4])) / (self.target_list[i][4] + self.target_list[j][4])
                    self.target_list[i][2] = ((self.target_list[i][2] * self.target_list[i][4]) + (self.target_list[j][2] * self.target_list[j][4])) / (self.target_list[i][4] + self.target_list[j][4])
                    self.target_list[i][4] = self.target_list[i][4] + self.target_list[j][4]
                    print("combined target lists:  list number {0:2d} had {1:2d} entries and list number {2:2d} had {3:2d} entries".format(i, self.target_list[i][4], j, self.target_list[j][4]))
                    del self.target_list[j]
                else:
                    j += 1

    def update_target_map(self, center_x, center_z, radius, color):
        #temp_map = np.zeros((self.arena_size * 2, self.arena_size * 2, 3))
        #circle_radius compensates for the difference between the center of the target
        #and the bottom of the target.  
        #As the agent moves toward the target, even though the target is getting closer
        #and the bottom of the target is moving down in the image, the center
        #of the target moves up in the image because the apparent radius is getting larger

        #a second correction is needed because the target is raised up in the air and, when it
        #gets close to the agent, the image cannot see its bottom

        circle_radius = int(radius + 0.5)
        if circle_radius < 1:
            circle_radius = 1
        # a circle_radius value of +6 above the radius works well when the target is less than 10m from the agent
        if 83 - center_z > 55:
            circle_radius += 2
        # as the target moves farther away, the effect diminishes, so we can increase circle_radius by less
        elif 83 - center_z > 48:
            circle_radius += 1
        #print("before limits, circle_radius = ", circle_radius, ", 83 - (center_z + circle_radius) = ", 83 - (center_z + circle_radius))
        # for large radius targets, circle_radius can get too big
        '''
        if circle_radius > 20:
            circle_radius = 20
        if 83 - (center_z + circle_radius) < 5:
            circle_radius = 5 - center_z
        if 83 - (center_z + circle_radius) > 41:
            circle_radius = 42 - center_z
        '''
        '''
        circle_radius = int(radius + 0.5)
        if circle_radius < 1:
            circle_radius = 1
        #a circle_radius value of +6 above the radius works well when the target is less than 10m from the agent
        if 83 - center_z < 10:
            circle_radius += 4
        #as the target moves farther away, the effect diminishes, so we can increase circle_radius by less
        if 83 - center_z < 20:
            circle_radius += 1
        if 83 - center_z < 31:
            circle_radius += 1
        if 83 - center_z < 35:
            circle_radius += 1
        #for large radius targets, circle_radius can get too big
        if circle_radius > 8:
            circle_radius -= 1
        if circle_radius > 20:
            circle_radius =20
        if 83 - (center_z + circle_radius) < 5:
            circle_radius = 0
        if 83 - (center_z + circle_radius) > 41:
            circle_radius = 42 - center_z 
        '''
        
        #print("center_z = ", center_z, ", 83 - center_z = ", 83 - center_z, ", radius = ", radius, ", circle_radius = ", circle_radius, ", 83 - (center_z + circle_radius) = ", 83 - (center_z + circle_radius))
        #map_x is in the range 0 to 39 and represents where the target is in the agent frame, from far left (0) to far right (39)
        #map_z is in the range 0 to 39 and represents where the target is in the agent frame, from immediately in front (0) to far away (39)
        # 83 - (center_z + circle_radius) is how far the bottom of the target is from the agent, in the image (not the map!)
        map_x, map_z = self.transform_image_pixel_to_map_floor_point(center_x, 83 - (center_z + circle_radius))
        if map_x == 0 and map_z == 0:
            print("image could not be interpreted for determining target location, skipping this one")
            return
        #print("center_x (image columns from the left) = ", center_x, ", center_z (image rows from the top) = ", center_z, ", radius = ", radius, ", circle_radius = ", circle_radius)
        
        #print("The bottom of the target is located in front of the agent by this many image rows: ", 83 - (center_z + circle_radius), ", circle_radius = ", circle_radius)
        #print("map_x, (map columns from the left in the agent frame): ", map_x, ", map_z (map rows from the agent) = ", map_z)

        #translate the point to correspond to agent location:
        #in the horizontal direction, the zero offset (center value) point of map_x is self.arena_size / 2, so we need to subtract that out
        #For example, if map_x = 20, then we do not want any x offset from self.x, so if self.x = 40, target_x should also = 40
        target_x = int((map_x - (self.arena_size / 2)) + self.x + 0.5)
        #in the vertical direction, the zero offset point just 0, so we do not need to subtract out any fixed value
        #however, since self.z is the distance from the top of the image, we need to subtract the value of map_z
        #from self.z to properly place the target in the image
        target_z = int(self.z + 0.5 - map_z)
        if target_x < 0:
            target_x = 0
        elif target_x > (self.arena_size * 2) - 1:
            target_x = (self.arena_size * 2) - 1
        if target_z < 0:
            target_z = 0
        elif target_z > (self.arena_size * 2) - 1:
            target_z = (self.arena_size * 2) - 1
            
        map_target_x, map_target_z = self.rotate_point((self.x, self.z), (target_x, target_z), -self.yaw)
        if map_target_x < 0 or map_target_x > (self.arena_size * 2) - 1 or map_target_z < 0 or map_target_z > (self.arena_size * 2) - 1:
            print("at least one map_target value is out of bounds: map_target_x = ", map_target_x, ", map_target_z", map_target_z)
            if map_target_x < 0:
                map_target_x = 0
            elif map_target_x > (self.arena_size * 2.) - 1:
                map_target_x = (self.arena_size * 2.) - 1
            if map_target_z < 0:
                map_target_z = 0
            elif map_target_z > (self.arena_size * 2.) - 1:
                map_target_z = (self.arena_size * 2.) - 1
        target_center_arena_frame_x = int(map_target_x + 0.5)
        target_center_arena_frame_z = int(map_target_z + 0.5)
        distance = np.sqrt(((map_x - (self.arena_size / 2.)) * (map_x - (self.arena_size / 2.))) + (map_z * map_z))
        target_diameter = (circle_radius * distance) / 40.  # at a distance of 39 meters, radius = 5; at 4 meters, radius = 39 for a size 5 target
        
        #print("self.x (agent columns from the left in map frame = ", self.x, ", self.z (agent rows from the top in map frame) = ", self.z)
        #print("x_target (map columns from the left in map frame = ", target_x, ", target_z (map rows from the top in the map frame = ", target_z)
        #print("map_target_x = ", map_target_x, ", map_target_z = ", map_target_z, ", target radius = ", target_diameter / 2)
        '''
        #calculate the angle of the target in the agent frame
        # Vectors subtending image center and pixel from optical center in image
        
        #target_location = np.array([center_x - (self.image_size / 2.), (self.image_size - center_z), self.focal_length])
        #target_location_magnitude = np.linalg.norm(target_location)
        #print("self.target_location = ", target_location, ", target_location_magnitude", target_location_magnitude)
        #angle between agent_location and target location
        #dot = np.dot(self.base_location, target_location)       
        #view_angle = np.arccos(dot / (self.base_location_magnitude * target_location_magnitude))

        #angle from agent to target
        # focalLength = (picWidth * knownDistance) / knownWidth
        view_angle = -np.arctan((center_x - (self.image_size / 2.)) / self.focal_length)
        
        total_rotation_angle = self.yaw + view_angle
        print("target view angle degrees = ", view_angle * 57.3, ", self.yaw degrees = ", self.yaw * 57.3, ", total rotation angle degrees = ", total_rotation_angle * 57.3)
            
        #rotate to transform from agent frame to big arena frame
        #delta_x = int((target_x - self.x) * np.cos(total_rotation_angle) + (self.z - target_z) * np.sin(total_rotation_angle) + 0.5)
        #delta_z = int((target_x - self.x) * np.sin(total_rotation_angle) + (self.z - target_z) * np.cos(total_rotation_angle) + 0.5)
        #delta_x = int(((map_x - (self.arena_size / 2)) * np.cos(total_rotation_angle)) + ((map_z * np.sin(total_rotation_angle)) + 0.5))
        #delta_z = int(((map_x - (self.arena_size / 2)) * np.sin(total_rotation_angle)) + ((map_z * np.cos(total_rotation_angle)) + 0.5))
        #delta_x = map_x - (self.x - self.arena_size)
        #delta_x = (map_x + (self.arena_size / 2.0)) - self.x
        delta_z = map_z
        delta_x = map_x - (self.arena_size / 2.)
        #some of the delta_x may be due to the agent yaw
        delta_z_agent_yaw = delta_x * np.sin(total_rotation_angle)
        delta_x_agent_yaw = delta_x * (np.cos(view_angle) - np.cos(self.yaw))
        #delta_x_agent_yaw = -(self.arena_size * (np.sin(view_angle))) * np.cos(self.yaw)
        #target_center_arena_frame_x = int((delta_x - delta_x_agent_yaw) + self.arena_size + 0.5)
        target_center_arena_frame_x = int(self.x + delta_x_agent_yaw + 0.5)
        target_center_arena_frame_z = int((self.z - (map_z + delta_z_agent_yaw)) + 0.5)
        distance = np.sqrt( (delta_x * delta_x) + (delta_z*delta_z))
        target_diameter = (circle_radius * distance) / 40.  #at a distance of 39 meters, radius = 5; at 4 meters, radius = 39 for a size 5 target
        
        #target_center_arena_frame_x = int((((map_x + (0.5 * self.arena_size)) - self.x) * np.sin(self.yaw)) + 0.5) + self.arena_size
        #target_center_arena_frame_z = int(((self.z - delta_z) * np.cos(self.yaw))+ 0.5)
        print("target located at x = {0:.2f}, z = {1:.2f}, radius = {2:.2f}".format(target_center_arena_frame_x, target_center_arena_frame_z, target_diameter/ 2.))
        print("self.x = ", self.x, ", self.z =", self.z, ", deltax = ", delta_x, ", delta_z = ", delta_z, ", view angle degrees = ", view_angle * 57.3, ", total rotation angle degrees = ", total_rotation_angle * 57.3)
        print("target distance from agent = ", distance, ", target diameter = ", target_diameter)
        print("delta_x_agent_yaw = ", delta_x_agent_yaw, ", delta_z_agent_yaw = ", delta_z_agent_yaw, ", sin(self.yaw) = ", np.sin(self.yaw), ", cos(self.yaw) = ", np.cos(self.yaw))
        '''
        
        if target_center_arena_frame_x < 0:
            target_center_arena_frame_x = 0
        elif target_center_arena_frame_x > (self.arena_size * 2) - 1:
            target_center_arena_frame_x = (self.arena_size * 2) - 1
        if target_center_arena_frame_z < 0:
            target_center_arena_frame_z = 0
        elif target_center_arena_frame_z > (self.arena_size * 2) - 1:
            target_center_arena_frame_z = (self.arena_size * 2) - 1
            
        new_target = True
        if len(self.target_list) > 0:
            print("target list: x, z, diameter, color, number of times seen")
        for i in range(len(self.target_list)):
            if (self.target_list[i][0] - 10 < target_center_arena_frame_x and self.target_list[i][0] + 10 > target_center_arena_frame_x
                and self.target_list[i][1] - 10 < target_center_arena_frame_z and self.target_list[i][1] + 10 > target_center_arena_frame_z
                and self.target_list[i][3] == color):
                #there is a previous entry for this target
                #print("this target was seen ", self.target_list[i][4], " times before" )
                self.target_list[i][0] = ((self.target_list[i][0] * self.target_list[i][4]) + target_center_arena_frame_x) / (self.target_list[i][4] + 1)
                self.target_list[i][1] = ((self.target_list[i][1] * self.target_list[i][4]) + target_center_arena_frame_z) / (self.target_list[i][4] + 1)
                self.target_list[i][2] = ((self.target_list[i][2] * self.target_list[i][4]) + target_diameter) / (self.target_list[i][4] + 1)
                self.target_list[i][4] += 1
                new_target = False
            
            print("{0:.1f}, {1:.1f}, {2:.1f}, {3:1d}, {4:2d}".format(self.target_list[i][0], self.target_list[i][1], self.target_list[i][2], self.target_list[i][3], self.target_list[i][4]))

        if new_target:
            self.target_list.append([target_center_arena_frame_x, target_center_arena_frame_z, target_diameter, color, 1])
            print("new target found:")
            print("{0:.1f}, {1:.1f}, {2:.1f}, {3:1d}, {4:2d}".format(self.target_list[-1][0], self.target_list[-1][1], self.target_list[-1][2], self.target_list[-1][3], self.target_list[-1][4]))

        #print("target location: x = {0:2d}, z = {1:2d}, 83 - z = {2:2d}, target diameter = {3:.1f}, color = {4:1d}".format(
        #    target_center_arena_frame_x, target_center_arena_frame_z, 83 - target_center_arena_frame_z, target_diameter, color))

        #combine targets that have become close due to averaging
        for i in range(len(self.target_list)):
            j = i + 1
            while j < len(self.target_list):
                if (self.target_list[i][0] - 10 < self.target_list[j][0] and self.target_list[i][0] + 10 > self.target_list[j][0]
                        and self.target_list[i][1] - 10 < self.target_list[j][1] and self.target_list[i][1] + 10 > self.target_list[j][1]
                        and self.target_list[i][3] == self.target_list[j][3]):
                    #these two entries are close enough to be combined.  They converged through averaging values
                    # print("this target was seen ", self.target_list[i][4], " times before" )
                    self.target_list[i][0] = ((self.target_list[i][0] * self.target_list[i][4]) + (self.target_list[j][0] * self.target_list[j][4])) / (self.target_list[i][4] + self.target_list[j][4])
                    self.target_list[i][1] = ((self.target_list[i][1] * self.target_list[i][4]) + (self.target_list[j][1] * self.target_list[j][4])) / (self.target_list[i][4] + self.target_list[j][4])
                    self.target_list[i][2] = ((self.target_list[i][2] * self.target_list[i][4]) + (self.target_list[j][2] * self.target_list[j][4])) / (self.target_list[i][4] + self.target_list[j][4])
                    self.target_list[i][4] = self.target_list[i][4] + self.target_list[j][4]
                    print("combined target lists:  list number {0:2d} had {1:2d} entries and list number {2:2d} had {3:2d} entries".format(i, self.target_list[i][4], j, self.target_list[j][4]))
                    del self.target_list[j]
                else:
                    j+= 1

        if color == 1:
            #print("green target distance = {0:.2f}, diameter = {1:.2f}".format(distance, target_diameter))
            cv2.circle(self.target_map, (target_center_arena_frame_x, target_center_arena_frame_z), int((target_diameter/ 2.) + 0.5), (0, 255, 0), -1)
            #cv2.circle(self.target_map, (target_x, target_z), int((target_diameter / 2.) + 0.5), (0, 0, 255), -1)           
            for i in range(-int((target_diameter) + 0.5), int((target_diameter) + 0.5) + 1):
               for j in range(-int((target_diameter) + 0.5), int((target_diameter) + 0.5) + 1):
                   if ((i + target_center_arena_frame_x >= 0 and i + target_center_arena_frame_x < self.arena_size * 2)
                       and (j + target_center_arena_frame_z >= 0 and j + target_center_arena_frame_z < self.arena_size * 2)):
                        self.green_target_map[j + target_center_arena_frame_z][i + target_center_arena_frame_x] += .1
                        if self.green_target_map[j + target_center_arena_frame_z][i + target_center_arena_frame_x] > 1.0:
                            self.green_target_map[j + target_center_arena_frame_z][i + target_center_arena_frame_x] = 1.0
                        #print("j + target_center_arena_frame_z][i + target_center_arena_frame_x = ", j + target_center_arena_frame_z, i + target_center_arena_frame_x)
                        #print("-int((2 * target_diameter) + 0.5), int((2 * target_diameter) + 0.5) + 1 = ", -int((2 * target_diameter) + 0.5), int((2 * target_diameter) + 0.5) + 1)
                        #print("i , j =", i, j)
        elif color == 2:
            #print("gold target distance = {0:.2f}, diameter = {1:.2f}".format(distance, target_diameter))
            cv2.circle(self.target_map, (target_center_arena_frame_x, target_center_arena_frame_z), int((target_diameter/ 2.) + 0.5), (0, 215, 255), -1)
            for i in range(-int((target_diameter) + 0.5), int((target_diameter) + 0.5) + 1):
               for j in range(-int((target_diameter) + 0.5), int((target_diameter) + 0.5) + 1):
                   if ((i + target_center_arena_frame_x >= 0 and i + target_center_arena_frame_x < self.arena_size * 2)
                       and (j + target_center_arena_frame_z >= 0 and j + target_center_arena_frame_z < self.arena_size * 2)):
                        self.gold_target_map[j + target_center_arena_frame_z][i + target_center_arena_frame_x] += .1
                        if self.gold_target_map[j + target_center_arena_frame_z][i + target_center_arena_frame_x] > 1.0:
                            self.gold_target_map[j + target_center_arena_frame_z][i + target_center_arena_frame_x] = 1.0
        elif color == 3:
            #print("red target distance = {0:.2f}, diameter = {1:.2f}".format(distance, target_diameter))
            cv2.circle(self.target_map, (target_center_arena_frame_x, target_center_arena_frame_z), int((target_diameter/ 2.) + 0.5), (0, 0, 255), -1)
            for i in range(-int((target_diameter) + 0.5), int((target_diameter) + 0.5) + 1):
               for j in range(-int((target_diameter) + 0.5), int((target_diameter) + 0.5) + 1):
                   if ((i + target_center_arena_frame_x >= 0 and i + target_center_arena_frame_x < self.arena_size * 2)
                       and (j + target_center_arena_frame_z >= 0 and j + target_center_arena_frame_z < self.arena_size * 2)):
                        self.red_target_map[j + target_center_arena_frame_z][i + target_center_arena_frame_x] += .1
                        #we show red targets on the main occupancy map because they are effectively obstacles
                        self.occupancy_map[j + target_center_arena_frame_z][i + target_center_arena_frame_x] += .1
                        if self.red_target_map[j + target_center_arena_frame_z][i + target_center_arena_frame_x] > 1.0:
                            self.red_target_map[j + target_center_arena_frame_z][i + target_center_arena_frame_x] = 1.0
                        if self.occupancy_map[j + target_center_arena_frame_z][i + target_center_arena_frame_x] > 1.0:
                            self.occupancy_map[j + target_center_arena_frame_z][i + target_center_arena_frame_x] = 1.0
        if self.display_map:
            self.show_target_map()
            
        self.target_list_from_occupancy_maps()
        
    def target_list_from_occupancy_maps(self):
        green_list = []
        gold_list = []
        red_list = []
        for i in range(5, (self.arena_size * 2) - 5):
            for j in range(5, (self.arena_size * 2) - 5):
                green_sum = 0
                for k in range(-5, 6):
                    for m in range(-5,6):
                        green_sum += self.green_target_map[i+k][j+m]
                if green_sum > 75:
                    #there may be a target here
                    green_list.append([i,j])
        for i in range(5, (self.arena_size * 2) - 5):
            for j in range(5, (self.arena_size * 2) - 5):
                gold_sum = 0
                for k in range(-5, 6):
                    for m in range(-5,6):
                        gold_sum += self.gold_target_map[i+k][j+m]
                if gold_sum > 75:
                    #there may be a target here
                    gold_list.append([i,j])
        for i in range(5, (self.arena_size * 2) - 5):
            for j in range(5, (self.arena_size * 2) - 5):
                red_sum = 0
                for k in range(-5, 6):
                    for m in range(-5,6):
                        red_sum += self.red_target_map[i+k][j+m]
                if red_sum > 75:
                    #there may be a target here
                    red_list.append([i,j])
        for k in range(3):  #some targets are too far apart to combine with the first sweep or two
            i = 0
            green_sum = [0,0,0]
            #print("length of green list before reducing = ", len(green_list), ", k = ", k)
            while i < len(green_list):
                j = i + 1
                while j < len(green_list):
                    #print(green_list[i][0], green_list[j][0], green_list[i][1], green_list[i][1])
                    if (green_list[i][0] - 10 < green_list[j][0] and green_list[i][0] + 10 > green_list[j][0]
                        and green_list[i][1] - 10 < green_list[j][1] and green_list[j][1] + 10 > green_list[j][1]):
                        green_sum[0] += green_list[j][0]
                        green_sum[1] += green_list[j][1]
                        green_sum[2] += 1
                        #print("deleted entry ", j)
                        del green_list[j]
                    else:
                        j += 1
                if green_sum[2] > 0:
                    green_list[i][0] = green_sum[0] / green_sum[2]
                    green_list[i][1] = green_sum[1] / green_sum[2]
                i += 1        
            for i in range(len(self.green_occupancy_list)):
                j = 0
                while j < len(green_list):
                    #print(green_list[i][0], green_list[j][0], green_list[i][1], green_list[i][1])
                    if (self.green_occupancy_list[i][0] - 10 < green_list[j][0] and self.green_occupancy_list[i][0] + 10 > green_list[j][0]
                        and self.green_occupancy_list[i][1] - 10 < green_list[j][1] and self.green_occupancy_list[i][1] + 10 > green_list[j][1]):
                        del green_list[j]
                    else:
                        j += 1

        if len(green_list) > 0:
            for values in green_list:
                print("adding a new target to the green_occupancy_list: (x,z): ({0:.1f}, {1:.1f})".format(values[1], values[0]))
                self.green_occupancy_list.append(values)
        if len(self.green_occupancy_list) > 0:
            print("green occupancy map target list (x,z): ")
            for values in self.green_occupancy_list:
                print("{0:.1f}, {1:.1f}".format(values[1], values[0]))
                
        for k in range(3):  #some targets are too far apart to combine with the first sweep or two
            i = 0
            gold_sum = [0,0,0]
            #print("length of gold list before reducing = ", len(gold_list), ", k = ", k)
            while i < len(gold_list):
                j = i + 1
                while j < len(gold_list):
                    #print(gold_list[i][0], gold_list[j][0], gold_list[i][1], gold_list[i][1])
                    if (gold_list[i][0] - 10 < gold_list[j][0] and gold_list[i][0] + 10 > gold_list[j][0]
                        and gold_list[i][1] - 10 < gold_list[j][1] and gold_list[j][1] + 10 > gold_list[j][1]):
                        gold_sum[0] += gold_list[j][0]
                        gold_sum[1] += gold_list[j][1]
                        gold_sum[2] += 1
                        #print("deleted entry ", j)
                        del gold_list[j]
                    else:
                        j += 1
                if gold_sum[2] > 0:
                    gold_list[i][0] = gold_sum[0] / gold_sum[2]
                    gold_list[i][1] = gold_sum[1] / gold_sum[2]
                i += 1
        for i in range(len(self.gold_occupancy_list)):
            j = 0
            while j < len(gold_list):
                #print(gold_list[i][0], gold_list[j][0], gold_list[i][1], gold_list[i][1])
                if (self.gold_occupancy_list[i][0] - 10 < gold_list[j][0] and self.gold_occupancy_list[i][0] + 10 > gold_list[j][0]
                    and self.gold_occupancy_list[i][1] - 10 < gold_list[j][1] and self.gold_occupancy_list[i][1] + 10 > gold_list[j][1]):
                    del gold_list[j]
                else:
                    j += 1
        if len(gold_list) > 0:
            for values in gold_list:
                print("adding a new target to the gold_occupancy_list: (x,z): ({0:.1f}, {1:.1f})".format(values[1], values[0]))
                self.gold_occupancy_list.append(values)
        if len(self.gold_occupancy_list) > 0:
            print("gold occupancy map target list (x,z):")
            for values in self.gold_occupancy_list:
                print("{0:.1f}, {1:.1f}".format(values[1], values[0]))
 
        for k in range(3):  #some targets are too far apart to combine with the first sweep or two
            i = 0
            red_sum = [0,0,0]
            #print("length of red list before reducing = ", len(red_list), ", k = ", k)
            while i < len(red_list):
                j = i + 1
                while j < len(red_list):
                    #print(red_list[i][0], red_list[j][0], red_list[i][1], red_list[i][1])
                    if (red_list[i][0] - 10 < red_list[j][0] and red_list[i][0] + 10 > red_list[j][0]
                        and red_list[i][1] - 10 < red_list[j][1] and red_list[j][1] + 10 > red_list[j][1]):
                        red_sum[0] += red_list[j][0]
                        red_sum[1] += red_list[j][1]
                        red_sum[2] += 1
                        #print("deleted entry ", j)
                        del red_list[j]
                    else:
                        j += 1
                if red_sum[2] > 0:
                    red_list[i][0] = red_sum[0] / red_sum[2]
                    red_list[i][1] = red_sum[1] / red_sum[2]
                i += 1
        for i in range(len(self.red_occupancy_list)):
            j = 0
            while j < len(red_list):
                #print(red_list[i][0], red_list[j][0], red_list[i][1], red_list[i][1])
                if (self.red_occupancy_list[i][0] - 10 < red_list[j][0] and self.red_occupancy_list[i][0] + 10 > red_list[j][0]
                    and self.red_occupancy_list[i][1] - 10 < red_list[j][1] and self.red_occupancy_list[i][1] + 10 > red_list[j][1]):
                    del red_list[j]
                else:
                    j += 1
        if len(red_list) > 0:
            for values in red_list:
                print("adding a new target to the red_occupancy_list: (x,z): ({0:.1f}, {1:.1f})".format(values[1], values[0]))
                self.red_occupancy_list.append(values)
        if len(self.red_occupancy_list) > 0:
            print("red occupancy map target list (x,z):")
            for values in self.red_occupancy_list:
                print("{0:.1f}, {1:.1f}".format(values[1], values[0]))
                
    def gold_target_captured(self):
        if self.use_target_maps:
            for i in range(-10, 11):
                for j in range(-10, 11):
                    if ((int(i + self.x + 0.5) >= 0 and int(i + self.x + 0.5) < self.arena_size * 2)
                            and (int(j + self.z + 0.5) >= 0 and int(j + self.z + 0.5) < self.arena_size * 2)):
                        self.gold_target_map[int(j + self.z + 0.5)][int(i + self.x + 0.5)] = 0.0
        i = 0
        while i < len(self.target_list):
            if (self.target_list[i][0] - 10 < self.x and self.target_list[i][0] + 10 > self.x
                and self.target_list[i][1] - 10 < self.z and self.target_list[i][1] + 10 > self.z
                and self.target_list[i][3] == 2):
                print("captured gold target removed from the target list, x = {0:.1f}, z = {1:.1f}, self.x = {2:.1f}, self.z = {3:.1f}".format(self.target_list[i][0], self.target_list[i][1], self.x, self.z))
                del self.target_list[i]
            else:
                i += 1
           
    def show_target_map(self):
        if (not self.target_map_window_created) and self.display_map:
            cv2.namedWindow('Targets', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Green_targets', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Gold_targets', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Red_targets', cv2.WINDOW_NORMAL)
            self.target_map_window_created = True
        arrow_length = 10.0
        show_map = self.target_map.copy()
        show_green_map = self.green_target_map.copy()
        show_gold_map = self.gold_target_map.copy()
        show_red_map = self.red_target_map.copy()
        current_location = (int(self.previous_x + 0.5), int(self.previous_z + 0.5))
        #opencv positive angle goes cw but our convention is for positive angle to go ccw
        #so we use negative angle for the display (sin only, since cos does not change with sign)
        yaw_arrow_end_x = int((self.previous_x + arrow_length * math.sin(-self.previous_yaw)) + 0.5)
        yaw_arrow_end_z = int((self.previous_z - arrow_length * math.cos(self.previous_yaw)) + 0.5)
        yaw_arrow_end = (yaw_arrow_end_x, yaw_arrow_end_z)
        show_map = cv2.arrowedLine(show_map, current_location, yaw_arrow_end, 255)
        show_green_map = cv2.arrowedLine(show_green_map, current_location, yaw_arrow_end, 255)
        show_gold_map = cv2.arrowedLine(show_gold_map, current_location, yaw_arrow_end, 255)
        show_red_map = cv2.arrowedLine(show_red_map, current_location, yaw_arrow_end, 255)
        for i in self.wall_points:
            if i[0] <= (self.arena_size * 2) - 1 and i[0] >= 0 and i[1] <= (self.arena_size * 2) - 1 and i[1] >= 0:
                show_map[i[0], i[1]] = 1.0
        if self.display_map:
            cv2.imshow('Targets', show_map)
            cv2.imshow('Green_targets', show_green_map)
            cv2.imshow('Gold_targets', show_gold_map)
            cv2.imshow('Red_targets', show_red_map)
            cv2.waitKey(10)

    def rotate_point(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy
        
    def transform_floor_mask_angles_to_map(self, image, wall_in_view):
        map = np.ones((40, 40), dtype=float)
        map *= 0.5
        pixel_values = np.zeros((40, 40), dtype=float)
        field_of_view_ratio = .577  # this is the tan(30 degrees), since we have a +- 30 degree field of view
        # 84 columns of the image correspond to 60 degrees in the field of view
        # column_to_radian_factor = 60./(84. * 57.3)
        # for z in range(83, -1, -1): #image row 83 is at the bottom, row 0 is at the top
        
        #important to remember that opencv image rows are the first index, cols are the second index
        #For example, in the observation image, the top left pixel has index [0][0], the top center pixel index is [0][41], the top right pixel index = [0][83]
        #the bottom left pixel index = [83][0], the bottom middle index is [83][41], and the bottom right index = [83][83]
        #In the 40x40 arena maps, the top left pixel has index [0][0], the top center pixel index is [0][19], the top right pixel index = [0][39]
        # the bottom left pixel index = [39][0], the bottom middle index is [39][19], and the bottom right index = [39][39]
        # In the 80x80 arena maps, the top left pixel has index [0][0], the top center pixel index is [0][39], the top right pixel index = [0][79]
        # the bottom left pixel index = [79][0], the bottom middle index is [79][39], and the bottom right index = [79][79]
        
        for x in range(self.image_size): #horizontal index for the image (columns to the left and right of the agent)
            for z in range(self.image_size): #rows in front of the agent in the image.  When the rows are farther away than this, the corresponding map values become too uncertain to use
                #note that the corresponding image z index is 83-z because opencv images have their 0th row at the top of the image
                #and the row closest to the agent is the bottom row of the image
                num_rows = 1
                # calibrate the relationship of pixel position to distance in the arena
                if z <= 11: 
                    start_length = 0  # distance from the agent that these z values in the image correspond to
                    end_length = 1
                    wall_distance = 1
                    num_rows = 12 #the number of image rows that we will integrate to generate a single map row
                elif z <= 20:
                    start_length = 1  #when the number of image rows is > 11 and <= 20, we are between 1 and 2 meters on the map
                    end_length = 2
                    wall_distance = 2
                    num_rows = 9
                elif z <= 27:
                    start_length = 2
                    end_length = 3
                    wall_distance = 3
                    num_rows = 7
                elif z <= 32:
                    start_length = 3
                    end_length = 4
                    wall_distance = 4
                    num_rows = 5
                elif z <= 34:
                    start_length = 4
                    end_length = 5
                    wall_distance = 5
                    num_rows = 2
                elif z == 35:
                    start_length = 5
                    end_length = 6
                    wall_distance = 6
                elif z == 36:
                    start_length = 6
                    end_length = 8
                    wall_distance = 8
                elif z == 37:
                    start_length = 8
                    end_length = 10
                    wall_distance = 9
                elif z == 38:
                    start_length = 10
                    end_length = 13
                    wall_distance = 13
                elif z == 39:
                    start_length = 14
                    end_length = 22
                    wall_distance = 19
                elif z <= 41:
                    start_length = 22
                    end_length = 39
                    wall_distance = 39
                    num_rows = 2  # suppress these last two a little
                else:
                    #we only use these rows to find the max floor level
                    start_length = 38
                    end_length = 39
                    wall_distance = 39
                                  
                #determine how far we are seeing floor to the right and left
                #it equals the tangent of 30 degrees * the distance from the agent to where the floor ends
                x_offset = int((field_of_view_ratio * end_length) + 0.5)
                #print("x_offset = ", x_offset)
                #if x_offset < 1:
                #    x_offset = 1
                #decide how many values of x in the image to combine to create a value for the map
                #for example, if we are doing the any of the first 12 image rows in front of the agent,
                #then end_length = 1 and the only values we are creating for the map are the 2 points directly in front of the agent
                #so we will combine all 42 x values to the left to generate one of those two and all 42 values to the right to generate the other
                #in that case, x_offset = 1, x_threshold = 42, and num_x_thresholds = 2
                x_threshold = int((84. / (x_offset * 2)) + 0.5)
                num_x_thresholds = int((84. / x_threshold) + 0.5) 
                if num_x_thresholds > 39: #we create a map value for each threshold, so the number of thresholds cannot exceed the size of the map
                    num_x_thresholds = 39
                    x_threshold = 2.15  # this is (84/39)
                #print("x_threshold = ", x_threshold, ", num_x_thresholds = ", num_x_thresholds, ", x = ", x)
                for i in range(num_x_thresholds): #to use num_x_thresholds, we go from 0 to num_x_thresholds - 1
                    #print("threshold number = ", i, ", x = ", x, ", i*x_threshold = ", i*x_threshold, ", (i+1)*x_threshold = ", (i+1)*x_threshold)
                    if x >= i * x_threshold and x < (i + 1) * x_threshold:  #if this value for x is within the current threshold range
                        #for this value for z (height in the image), we determined the corresponding rows in the image
                        #which are start_length and end_length.  For example, if z <= 11, start_length = 0 and end_length = 1
                        #because the first 12 image rows in front of the agent take up only 1 meter of distance in front of the agent on the map
                        #now for each of the rows from start_length to end_length we add the value of the image pixel [83-z][x]
                        
                        #look for the wall, obstacle, or target that is blocking the view of the floor
                        if z > 0:
                            if image[83 - z][x] > 127 and image[83 - (z - 1)][x] > 127 and int(wall_distance + 0.5) > self.max_map_row_floor[int((19 - ((num_x_thresholds - 1) / 2)) + i + 0.5)]:
                                self.max_map_row_floor[int((19 - ((num_x_thresholds - 1) / 2)) + i + 0.5)] = int(wall_distance + 0.5)
                                #if x == 83 or x == 82:
                                    #print("new max z: threshold number = ", i, ", num_x_thresholds = ", num_x_thresholds, ", new max floor level: x = ", x, ", image z = ", z, ", 83-z = ", 83 - z, ", index = ", int((19 - ((num_x_thresholds - 1) / 2)) + i + 0.5), "previous value = ", image[83 - z][x],
                                    #", value = ", image[83 - (z - 1)][x], ", max map floor level = ", self.max_map_row_floor[int((19 - ((num_x_thresholds - 1) / 2)) + i + 0.5)])
                            #elif x == 83 or x == 82:
                            #    #print("no new max z: threshold number = ", i, ", num_x_thresholds = ", num_x_thresholds, ", x = ", x, ", current z = ", z, ", 83- currentz = ", 83 - z, ", index = ", int((19 - ((num_x_thresholds - 1) / 2)) + i + 0.5), "previous value = ", image[83 - z][x],
                            #    #      ", value = ", image[83 - (z - 1)][x], ", previous max map floor level = ", self.max_map_row_floor[int((19 - ((num_x_thresholds - 1) / 2)) + i + 0.5)])
                            #                                  
                        if z <= self.z_max: #bigger values for z have too much uncertainty in distance
                            for j in range(start_length, end_length):
                                pixel_values[39 - j][int((19 - ((num_x_thresholds - 1) / 2)) + i + 0.5)] += image[83 - z][x] / (x_threshold * num_rows)
                                if z > 500:
                                    print("threshold number = ", i, ", map row index= ", 39-j, ", map column index  = ", int((19 - ((num_x_thresholds - 1) / 2)) + i + 0.5),
                                          ", image row index = ", 83 - z, ", image column index = ", x, 
                                          ", raw image value = ", image[83 - z][x],
                                          ", integration step value = ", image[83 - z][x] / (x_threshold * num_rows),
                                          ", integration total value = ", pixel_values[39 - j][int((19 - ((num_x_thresholds - 1) / 2)) + i + 0.5)])
                    #elif z == 0:
                    #    print("x is outside threshold range, i = ", i, ", i*x_threshold = ", i*x_threshold, ", num_x_thresholds = ", num_x_thresholds, ", x = ", x)
        # note that these x and z correspond to map coordinates.  The x and z above correspond to image coordinates
        #print("self.max_map_row_floor =", self.max_map_row_floor)
        no_values_before_z_equals_one = True
        for x in range(40):
            #print(" x = ", x, ", self.max_map_row_floor[x] = ", self.max_map_row_floor[x])
            for z in range(39, -1, -1):
                if pixel_values[39 - z][x] > 127.:  # if we see floor, mark the spot as unoccupied
                    #print("unoccupied coords, x, z = ", x, z)
                    map[39 - z][x] = 0.001
                '''
                #mark where the floor ends
                if z == self.max_map_row_floor[x]:
                    #print("matched self.max_map_row_floor =", self.max_map_row_floor[x], ", with x =", x, ", z = ", z, ", 39-z = ", 39 - z)
                    #cv2.circle(map, (x, 39 - z), 1, 1.0, -1)
                    #map[39 - z][x] = 0.95
                    if z == 1 and x == 20: 
                        z_primed = True 
                        if x >= 5 and x <= 34:
                            cv2.rectangle(map, (x - 5, 20), (x + 5, 38), 1, -1)
                    if z == 0 and x == 20 and wall_in_view == 1 and no_values_before_z_equals_one:
                        cv2.rectangle(map, (0, 20), (39, 38), 1, -1)
                        no_values_before_z_equals_one = False
                '''
        if self.max_map_row_floor[20] == 1 and self.max_map_row_floor[19] == 0 and self.max_map_row_floor[21] == 0:
            if wall_in_view == 1:
                #fill in short wall
                cv2.rectangle(map, (15, 20), (25, 38), 1, -1)
            else:
                #fill in obstacle
                cv2.rectangle(map, (15, 35), (25, 38), 1, -1)
            #prepare for filling in long wall
            self.facing_perpendicular_to_wall = True
        if self.facing_perpendicular_to_wall:
            sum = 0
            for i in range(40):
                sum += self.max_map_row_floor[i]
            if sum == 0:
                if wall_in_view == 1:
                    #fill in long wall
                    cv2.rectangle(map, (0, 20), (39, 38), 1, -1)
                    self.long_wall = True
                self.facing_perpendicular_to_wall = False
        if self.max_map_row_floor[20] >= 5:
            self.facing_perpendicular_to_wall = False
                        
                    #map[39 - z][x] = 0.95
                #alternative way to mark where the floor ends
                #else:
                #    if z > 0:
                #        #we found where the floor ends, so this is either a wall, an obstacle, or a target
                #        if pixel_values[39 - (z - 1)][x] > 127.:
                #            map[39 - z][x] = 0.95
        return map

    def transform_image_pixel_to_map_floor_point(self, image_x, image_z):
        #image_x is how far the pixel is from the left of the agent image
        #image_z is how far the pixel is in front of the agent in the image
        field_of_view_ratio = .577  # this is the tan(30 degrees), since we have a +- 30 degree field of view
        map_x = 0
        if image_z <= 11:
            map_distance = 1 # distance from the agent that these z values in the image correspond to   
        elif image_z <= 20:
            map_distance = 2
        elif image_z <= 27:
            map_distance = 3
        elif image_z <= 32:
            map_distance = 4
        elif image_z <= 34:
            map_distance = 5
        elif image_z == 35:
            map_distance = 6
        elif image_z == 36:
            map_distance = 8
        elif image_z == 37:
            map_distance = 10
        elif image_z == 38:
            map_distance = 13
        elif image_z == 39:
            map_distance = 22
        elif image_z <= 41:
            map_distance = 30
        else:
            print("unable to transform_image_pixel_to_map_floor_point, image_z = ", image_z)
            return 0, 0

        #map_z is the distance from the agent to the target
        map_z = int(map_distance + 0.5)
        #print("image_z = ", image_z, ", map_distance = ", map_distance)

        x_offset = int((field_of_view_ratio * map_distance) + 0.5)
        x_threshold = int((84. / (x_offset * 2)) + 0.5)
        num_x_thresholds = int(((84. / x_threshold) - 1) + 0.5)
        #print("x_offset = ", x_offset, ", x_threshold = ", x_threshold, ", num_x_thresholds = ", num_x_thresholds)
        if num_x_thresholds > 39:
            num_x_thresholds = 39
            x_threshold = 2.15  # this is (84/78) * 2
        for i in range(num_x_thresholds + 1):
            #print("i = ", i, ", image_x = ", image_x, ", i*threshold = ", i * x_threshold, ", int(19.5 - (num_x_thresholds / 2)) + i = ", int(19.5 - (num_x_thresholds / 2)) + i)
            if image_x > i * x_threshold and image_x <= (i + 1) * x_threshold:
                map_x = int(19.5 - (num_x_thresholds / 2)) + i
                #print("found a map_x = ", map_x)
                break
        if map_x > 39:
            map_x = 39
        elif map_x < 0:
            map_x = 0
        return map_x, map_z

    def translate_map_to_center(self, map):
        translate = self.arena_size
        big_map = np.ones((self.arena_size * 2, self.arena_size * 2))
        big_map *= 0.5
        for i in range(self.arena_size):
            for j in range(self.arena_size):                
                vertical_index = int(i + 0.5 + translate - self.arena_size) #offset by full arena size because the floor space vertical origin is at the bottom of the image
                horiz_index = int(j + 0.5 + translate - (self.arena_size / 2.)) #offset by half arena size because the floor space horizontal origin is in the middle of the image
                if vertical_index < 0:
                    vertical_index = 0
                elif vertical_index > (self.arena_size * 2) - 1:
                    vertical_index = (self.arena_size * 2) - 1               
                if horiz_index < 0:
                    horiz_index = 0
                elif horiz_index > (self.arena_size * 2) - 1:
                    horiz_index = (self.arena_size * 2) - 1
                big_map[vertical_index][horiz_index] = map[i][j]
        return big_map

    def translate_map(self, map):
        big_map = np.ones((self.arena_size * 2, self.arena_size * 2))
        big_map *= 0.5
        for i in range(self.arena_size*2):
            for j in range(self.arena_size*2):                
                vertical_index = int(i + 0.5 + self.z - self.arena_size) #offset by full arena size because the floor space vertical origin is at the bottom of the image
                horiz_index = int(j + 0.5 + self.x - self.arena_size)  # offset by full arena size because the centered floor space horizontal origin is in the middle of the image
                if vertical_index < 0:
                    vertical_index = 0
                elif vertical_index > (self.arena_size * 2) - 1:
                    vertical_index = (self.arena_size * 2) - 1               
                if horiz_index < 0:
                    horiz_index = 0
                elif horiz_index > (self.arena_size * 2) - 1:
                    horiz_index = (self.arena_size * 2) - 1
                #if i == 20 and j == 20:
                    #print("vertical index = ", vertical_index, ", horiz index = ", horiz_index)
                #big_map[int(i + 0.5 + self.x - self.arena_size)][int(j + 0.5 + self.z - (self.arena_size / 2))] = map[i][j]
                big_map[vertical_index][horiz_index] = map[i][j]
        return big_map
        
    def rotate_bound(self, image, angle_radians):
        angle = angle_radians * 57.3
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    def transform_network_action_to_model_action(self, network_action):
        model_action = [0.0, 1.0]
        if network_action == 0:
            model_action = [0.0, 2.0] # left
        elif network_action == 1:
            model_action = [1.0, 2.0] # forward left
        elif network_action == 2:
            model_action = [1.0, 0.0] # forward
        elif network_action == 3:
            model_action = [1.0, 1.0] # forward right
        elif network_action == 4:
            model_action = [0.0, 1.0] # right
        elif network_action == 5:
            model_action = [2.0, 1.0] # backward right
        elif network_action == 6:
            model_action = [2.0, 0.0] # backward
        elif network_action == 7:
            model_action = [2.0, 2.0] # backward left
        elif network_action == 8:
            model_action = [0.0, 0.0] # none
        else:
            print("received an unknown network action = ", network_action, " to translate to action, returning [0.0, 1.0] (right turn)")
        return model_action

    def astar(self, start, end, map, show_search_progress=False):
        print("finding an A* path from ({0:.1f}, {1:.1f}) to ({2:.1f}, {3:.1f})".format(
            start[0], start[1], end[0], end[1]))
        if start[0] > end[0] - 5 and start[0] < end[0] + 5 and start[1] > end[1] - 5 and start[1] < end[1] + 5:
            #start and end are close enough that we do not need to compute a path
            #but we do not want to return a zero length path because that indicates a path was not found
            short_path = []
            short_path.append(start)
            short_path.append(end)
            return short_path
        #first check that the goal is within the bounds of the map and is not inside an obstacle
        # check for edge
        if start[0] > (len(map) - 1) or start[0] < 0 or start[1] > (len(map[len(map) - 1]) - 1) or start[1] < 0:
            print("A* start is outside of the map bounds")
            return None            
        if end[0] > (len(map) - 1) or end[0] < 0 or end[1] > (len(map[len(map) - 1]) - 1) or end[1] < 0:
            print("A* goal is outside of the map bounds")
            return None
        # check for obstacles
        if map[int(start[0] + 0.5)][int(start[1] + 0.5)] > 0.75:
            print("A* start is within an obstacle")
            return None
        if map[int(end[0] + 0.5)][int(end[1] + 0.5)] > 0.75:
            print("A* goal is within an obstacle")
            return None
        
        display_search_progress_image = np.zeros((80, 80, 3))
        
        if self.display_map and not self.astar_map_window_created:
            cv2.namedWindow('A*', cv2.WINDOW_NORMAL)
            self.astar_map_window_created = True
        if show_search_progress and (not self.astar_search_progress_window_created) and self.display_map:
            cv2.namedWindow('Astar_search_progress', cv2.WINDOW_NORMAL)
            self.astar_search_progress_window_created = True
        MAX_NUM_TRIES = 5000
        num_tries = 0
        # create start and end node
        start_node = Node(None, start)
        start_node.cost_from_start = start_node.estimated_cost_to_goal = start_node.full_cost = 0
        end_node = Node(None, end)
        end_node.cost_from_start = end_node.estimated_cost_to_goal = end_node.full_cost = 0
        if show_search_progress:
            display_search_progress_image[int(end_node.position[1] + 0.5)][int(end_node.position[0] + 0.5)] = (0, 0, 255)

        # initialize both open and closed list
        # the open list are the nodes that need to be examined
        open_list = []
        # the closed list are nodes that have already been expanded in every direction
        closed_list = []
        # Add the start node to the open list
        # later we will generate child nodes from all 8 possible moves from this node and add them to the open list
        open_list.append(start_node)

        # check every node on the open list to find the one with lowest total cost
        # we will remove that one from the open list and add it to the closed list
        while len(open_list) > 0:
            num_tries += 1
            if num_tries > MAX_NUM_TRIES:
                print("astar used the maximum number of tries but failed to find a path to the goal")
                return None

            # the current node is the first one on the open list
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                # check if this node has lower total cost than the current node
                if item.full_cost < current_node.full_cost:
                    # make this node the current node, but check the whole open list to find the node with lowest cost
                    current_node = item
                    current_index = index

                    # pop the current node off the open list, add it to the closed list
            # will will generate child nodes in every direction from the current node
            # and once that is done, we will no longer need to examine this node, so it belongs on the closed list
            open_list.pop(current_index)
            closed_list.append(current_node)

            if show_search_progress:
                for i in open_list:
                    # show the open list positions in green
                    display_search_progress_image[int(i.position[1] + 0.5)][int(i.position[0]+ 0.5)] = (0, 255, 0)
                for i in closed_list:
                    # show the closed list positions in gold
                    display_search_progress_image[int(i.position[1] + 0.5)][int(i.position[0] + 0.5)] = (0, 255, 128)
                if show_search_progress and self.display_map:
                    cv2.imshow("Astar_search_progress", display_search_progress_image)
                    cv2.waitKey(10)

            # check to see if we have found the goal
            if (end_node.position[0] - 1 <= current_node.position[0] and end_node.position[0] + 1 >= current_node.position[0] 
                    and end_node.position[1] - 1 <= current_node.position[1] and end_node.position[1] + 1 >= current_node.position[1]):
                path = []
                current = current_node
                while current is not None:
                    # run back along the nodes to create the path in reverse (from goal to start)
                    path.append(current.position)
                    current = current.parent
                #print("number of passes through the astar loop = ", num_tries)
                #print("A* path length = ", len(path))
                if self.display_map:
                    astar_map = self.occupancy_map.copy()
                    for points in path:
                        astar_map[int(points[1] + 0.5)][int(points[0] + 0.5)] = 1
                    cv2.imshow("A*", astar_map)
                return path[::-1]  # reverse the list and return the path from start to goal

            # generate children nodes to the current node
            children = []
            new_position_distance = []
            # check every possible move from the current node (the 8 adjacent locations)
            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # Adjacent squares
                # get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                # check for edge
                if node_position[0] > (len(map) - 1) or node_position[0] < 0 or node_position[1] > (len(map[len(map) - 1]) - 1) or node_position[1] < 0:
                    continue

                # check for obstacles
                if map[int(node_position[0] + 0.5)][int(node_position[1] + 0.5)] > 0.75:
                    continue

                # this is a viable path, create a new node
                new_node = Node(current_node, node_position)

                # append the new node to the children of the current node
                children.append(new_node)

                # record whether or not it is a diagonal move so that we get the cost correct
                if new_position[0] == 0 or new_position[1] == 0:
                    new_position_distance.append(1.0)
                else:
                    # diagonal moves cost is sqrt(2) instead of 1
                    new_position_distance.append(1.41)

            child_number = -1  # this keeps track of which child node we are evaluating, so we can get the correct new_position_distance
            # calculate costs for each child node
            for child in children:
                child_number += 1
                # check if this node is on the closed list
                # if so, we have already explored all of its possible moves, so we can just continue
                duplicate = False
                for closed_child in closed_list:
                    if child == closed_child:
                        duplicate = True
                        break
                if duplicate:
                    continue

                # calculate the costs
                child.cost_from_start = current_node.cost_from_start + new_position_distance[child_number]
                child.estimated_cost_to_goal = np.sqrt(((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2))
                child.full_cost = child.cost_from_start + child.estimated_cost_to_goal

                # check if this is a duplicate location to a node in the open list
                duplicate = False
                for open_node in open_list:
                    if child == open_node:
                        # this child node has the same location as a node on the open list
                        # If the child node has a higher cost from start than the existing node, we just move on
                        # In normal astar, if the child node has a lower cost than the existing node, 
                        # we add it to the open list.  However, this slows things down tremendously
                        # For our use, if the child node has a lower cost than the existing node,
                        # then we replace the existing node with the child node.

                        if child.cost_from_start < open_node.cost_from_start:
                            open_node.cost_from_start = child.cost_from_start
                            # open_node.estimated_cost_to_goal = child.estimated_cost_to_goal  #this value is unchanged
                            open_node.full_cost = child.cost_from_start + child.estimated_cost_to_goal
                            # we replace the open node's parent with the child node's parent so that
                            # the whole path shifts to the path that goes from start to this child node
                            open_node.parent = current_node
                        duplicate = True
                        break
                if duplicate:
                    continue

                # the child node was not on the open list or the closed list, add it to the open list
                open_list.append(child)
        print("astar ended with an empty open list and failed to find a path to the goal")
        return None



    def convert_distance_to_moves(self, distance):
        '''
        3 moves, v = 6.5, 2 backs, distance 1 meter
        6 moves, v = 11, 3 backs = 3.1
        9 moves, v = 14.1, 4 backs = 5.9
        12 moves, v = 16.3, 4 backs = 9.1
        15 moves, v = 17.8, 4 backs = 12.5
        18 moves, v = 18.8, 4 backs = 16.0
        21 moves, v = 19.6, 4 backs = 19.7
        24 moves, v = 20.1, 4 backs = 23.4
        27 moves, v = 20.4, 4 backs = 27.1
        30 moves, v = 20.6, 4 backs = 30.9
        '''
        if distance <= 1.0:
            return (3, 2)
        if distance <= 2.3:
            return(5, 3)
        if distance <= 3.1:
            return (6, 3)
        if distance <= 4.0:
            return(7, 3)
        if distance <= 5.0:
            return(8, 3)
        if distance <= 5.9:
            return(9, 3)
        if distance <= 7.0:
            return(10, 4)
        if distance < 8.1:
            return(11, 4)
        if distance <= 9.2:
            return (12, 4)
        else:
            moves = int((distance - 9.2) + 12.5)
            return (moves, 4)     

    def follow_path(self, path):
        if path is None:
            return 0, 0, 0, []
        if len(path) == 0:
            return 0, 0, 0, []
        turn_indexes = []
        #find the first turn       
        j = 1
        while j + 18 < len(path):
            dx = dz = 0
            for i in range(j, j + 10):
                dx += path[i][0] - path[i - 1][0]
                dz += path[i][1] - path[i - 1][1]
            dx1 = dz1 = 0
            for i in range(j + 10, j + 19):
                dx1 += path[i][0] - path[i - 1][0]
                dz1 += path[i][1] - path[i - 1][1]
            if (dx >= dx1 - 2 and dx <= dx1 + 2) and (dz >= dz1 - 2 and dz <= dz1 + 2):
                #no turn
                j += 1
                continue
            #found a turn, now figure out exactly where it happens
            '''
            previous_dx = path[j][0] - path[j - 1][0]
            previous_dz = path[j][1] - path[j - 1][1]
            for i in range(j + 1, j + 19):
                dx = path[i][0] - path[i - 1][0]
                dz = path[i][1] - path[i - 1][1]
                if dx == previous_dx and dx == previous_dz:
                    #no turn
                    previous_dx = dx
                    previous_dz = dz
                    continue
                turn_indexes.append(i)
                break
            '''
            turn_indexes.append(j + 10)
            j += 10

        print("A* path length = ", len(path), ", with ", len(turn_indexes), " turns.  The turns are at indices: ", turn_indexes)
        if len(turn_indexes) == 0:
            #no turns, so do the whole path
            #path_yaw = np.arctan((path[0][0] - path[len(path) - 1][0]) / (path[0][1] - path[len(path) - 1][1]))
            # y values (numerator) are reversed because images have index 0 as the top row
            # we add 1.57 because straight up is 0 degrees in our image
            path_yaw = math.atan2(path[0][1] - path[len(path) - 1][1], path[len(path) - 1][0] - path[0][0]) - 1.57
            num_turns = int(round((path_yaw - self.yaw) * 9.55)) #each turn is 6 degrees = .1047 radians.  1/.1047 = 9.55
            print("path yaw degrees = ", path_yaw * 57.3, ", current yaw degrees = ", self.yaw * 57.3, ", num_turns = ", num_turns)
            #input("astar moves")
            #360 degrees is 60 turns
            if np.abs(num_turns) > 30:
                if num_turns > 0:
                    num_turns = num_turns - 60
                else:
                    num_turns = num_turns + 60
            # make up for the initial left turn
            num_turns -= 1
            distance = np.sqrt(((path[len(path) - 1][0] - path[0][0]) ** 2) + ((path[len(path) - 1][1] - path[0][1]) ** 2))
            num_moves, num_back = self.convert_distance_to_moves(distance)
            path =[]
            #print("path_yaw degrees = ", path_yaw * 57.3, " current yaw degrees = ", self.yaw * 57.3,
            #      ", num_turns = ", num_turns, ", distance = ", distance, ", num_moves = ", num_moves)
            return num_turns, num_moves, num_back, path
        #we will just go the first line segment
        if (path[turn_indexes[0]][1] - path[0][1]) == 0:
            if (path[turn_indexes[0]][0] - path[0][0]) >= 0:
                path_yaw = -1.57
            elif (path[turn_indexes[0]][0] - path[0][0]) < 0:
                path_yaw = 1.57                   
        else:
            #path_yaw = np.arctan((path[0][0] - path[len(path) - 1][0]) / (path[0][1] - path[len(path) - 1][1]))
            # y values (numerator) are reversed because images have index 0 as the top row
            # we add 1.57 because straight up is 0 degrees in our image
            path_yaw = math.atan2(path[0][1] - path[len(path) - 1][1], path[len(path) - 1][0] - path[0][0]) - 1.57
        num_turns = int(round((path_yaw - self.yaw) * 9.55))  # each turn is 6 degrees = .1047 radians.  1/.1047 = 9.55
        print("segment path yaw degrees = ", path_yaw * 57.3, ", current yaw degrees = ", self.yaw * 57.3, ", num_turns = ", num_turns)
        # 360 degrees is 60 turns
        if np.abs(num_turns) > 30:
            if num_turns > 0:
                num_turns = num_turns - 60
            else:
                num_turns = num_turns + 60
        #make up for the initial left turn
        num_turns -= 1

        distance = np.sqrt(((path[turn_indexes[0]][0] - path[0][0]) ** 2) + ((path[turn_indexes[0]][1] - path[0][1]) ** 2))
        num_moves, num_back = self.convert_distance_to_moves(distance)
        #print("path_yaw degrees = ", path_yaw * 57.3, " current yaw degrees = ", self.yaw * 57.3,
        #      ", num_turns = ", num_turns,  ", distance = ", distance, ", num_moves = ", num_moves)
        for i in range(turn_indexes[0]):
            path.pop()
        return num_turns, num_moves, num_back, path


    def find_unexplored_area_near_current_location(self, check_in_new_direction):
        if self.display_map:
            exploration_image = self.occupancy_map.copy()
            if (not self.exploration_window_created) and self.display_map:
                cv2.namedWindow('Exploration', cv2.WINDOW_NORMAL)
                self.exploration_window_created = True
        #begin at 5 away, in the direction we are already facing
        #if not check_in_new_direction:
        i = int(self.x + 0.5 - (10 * (np.sin(self.yaw))))
        j = int(self.z + 0.5 - (10 * (np.cos(self.yaw))))
        #else:
        #    print("checking for unexplored area in new direction")
        #    i = int(self.x + 0.5 + (10 * (np.sin(self.yaw))))
        #    j = int(self.z + 0.5 + (10 * (np.cos(self.yaw))))
        print("looking for an unexplored region near ", i, j)
        unexplored_x = -1
        unexplored_z = -1
        direction = 1
        counter = 1
        i_wait_count = 0
        j_wait_count = counter

        while (i >= 5 or i <= len(self.occupancy_map * 2) - 6 or j >= 5 or j <= len(self.occupancy_map * 2) - 6) and counter < 1000:
            unexplored_region = True
            if i < 5:
                i = len(self.occupancy_map * 2) - 6
            elif i > len(self.occupancy_map * 2) - 6:
                i = 5
            if j < 5:
                j = len(self.occupancy_map * 2) - 6
            elif j > len(self.occupancy_map * 2) - 6:
                j = 5
            for m in range(-5, 6):
                for n in range(-5, 6):
                    if self.display_map:
                        exploration_image[j][i] = 1
                        if self.display_map:
                            cv2.imshow("Exploration", exploration_image)
                            cv2.waitKey(10)
                    if self.occupancy_map[j + n][i + m] < 0.4 or self.occupancy_map[j + n][i + m] > 0.6:
                        #print("this point was explored previously, i+m = ", i+m, ", j+n = ", j+n, ", pixel = ", self.occupancy_map[j + n][i + m])
                        unexplored_region = False
                        #move the search 5 away from this point
                        i = int(i + 0.5 - (5 * (np.sin(self.yaw))))
                        j = int(j + 0.5 - (5 * (np.cos(self.yaw))))
                        break
                    #else:
                    #    print("this point was unexplored, i+m = ", i + m, ", j+n = ", j + n, ", pixel = ", self.occupancy_map[j + n][i + m])
                    #input("check exploration")
                if not unexplored_region:
                    break
            if unexplored_region:
                unexplored_x = i
                unexplored_z = j
                break
            
            if i_wait_count == 0:
                i_wait_count = -1
                j_wait_count = counter
            if j_wait_count == 0:
                j_wait_count = -1
                i_wait_count = counter
                counter += 1
                direction = -direction
            if i_wait_count > 0:
                i += direction
                i_wait_count -= 1
            if j_wait_count > 0:
                j += direction
                j_wait_count -= 1   
                

                
        if unexplored_region:
            #found an unexplored region
            print("unexplored center point = ", unexplored_x, unexplored_z, ", astar counter = ", counter)
            path = self.astar((self.x, self.z), (i, j), self.occupancy_map, show_search_progress=False)
            return path
        print("unable to find an unexplored region") 
        return None
  
    def add_obstacle(self, in_front, wall):       
        #find a line perpendicular to our current yaw
        if wall == 1:  #wall in front
            length = self.arena_size - 1
            thickness = 20
        else:
            length = 10.
            thickness = 3
        
        delta_x = length * math.cos(self.previous_yaw)
        delta_z = length * math.sin(self.previous_yaw)
        start_x = int(self.x + 0.5 - delta_x)
        start_z = int(self.z + 0.5 + delta_z)
        end_x = int(self.x + 0.5 + delta_x)
        end_z = int(self.z + 0.5 - delta_z)
            
        '''
        if start_x < 0:
            start_x = thickness
        elif start_x >= len(self.occupancy_map) - thickness:
            start_x = len(self.occupancy_map) - thickness
        if start_z < 0:
            start_z = thickness
        elif start_z >= len(self.occupancy_map) - thickness:
            start_z = len(self.occupancy_map) - thickness
        
        
        if start_x < 0:
            start_x = 0
        elif start_x >= len(self.occupancy_map) - 1:
            start_x = len(self.occupancy_map) - 1
        if start_z < 0:
            start_z = 0
        elif start_z >= len(self.occupancy_map) - 1:
            start_z = len(self.occupancy_map) - 1
        if end_x < 0:
            end_x = 0
        elif end_x >= len(self.occupancy_map) - 1:
            end_x = len(self.occupancy_map) - 1
        if end_z < 0:
            end_z = 0
        elif end_z >= len(self.occupancy_map) - 1:
            end_z = len(self.occupancy_map) - 1

        print("adding obstacle, start_x = ", start_x, ", start_z = ", start_z,
              ", end_x = ", end_x, ", end_z = ", end_z, ", delta_x = ", delta_x, ", delta_z = ", delta_z)
        cv2.line(self.occupancy_map, (start_x, start_z), (end_x, end_z), 1, 1)
        '''   
        for i in range(thickness):           
            extended_delta_x = i * math.sin(self.previous_yaw)
            extended_delta_z = (i + 3) * math.cos(self.previous_yaw)
            
            if in_front:
                extended_start_x = int(start_x + 0.5 - extended_delta_x)
                extended_start_z = int(start_z + 0.5 - extended_delta_z)
                extended_end_x = int(end_x + 0.5 - extended_delta_x)
                extended_end_z = int(end_z + 0.5 - extended_delta_z)
                
            else:
                extended_start_x = int(start_x + 0.5 + extended_delta_x)
                extended_start_z = int(start_z + 0.5 + extended_delta_z)
                extended_end_x = int(end_x + 0.5 + extended_delta_x)
                extended_end_z = int(end_z + 0.5 + extended_delta_z)
            cv2.line(self.occupancy_map, (extended_start_x, extended_start_z), (extended_end_x, extended_end_z), 1, 2)
            
        print("adding obstacle, extended_start_x = ", extended_start_x, ", extended_start_z = ", extended_start_z,
              ", extended_end_x = ", extended_end_x, ", extended_end_z = ", extended_end_z,
              ", extended_delta_x = ", extended_delta_x, ", extended_delta_z = ", extended_delta_z)
        print("line angle degrees = ", math.atan2(extended_end_x - extended_start_x, extended_end_z - extended_start_z) * 57.3)
        
        if wall == 1:
            for i in range(thickness):               
                extended_delta_x = -(self.arena_size + i) * math.sin(self.previous_yaw)
                extended_delta_z = -(self.arena_size + i + 3) * math.cos(self.previous_yaw)
                
                if in_front:
                    extended_start_x = int(start_x + 0.5 - extended_delta_x)
                    extended_start_z = int(start_z + 0.5 - extended_delta_z)
                    extended_end_x = int(end_x + 0.5 - extended_delta_x)
                    extended_end_z = int(end_z + 0.5 - extended_delta_z)

                else:
                    extended_start_x = int(start_x + 0.5 + extended_delta_x)
                    extended_start_z = int(start_z + 0.5 + extended_delta_z)
                    extended_end_x = int(end_x + 0.5 + extended_delta_x)
                    extended_end_z = int(end_z + 0.5 + extended_delta_z)
                    
                cv2.line(self.occupancy_map, (extended_start_x, extended_start_z), (extended_end_x, extended_end_z), 1, 2)
            print("adding opposite wall, extended_start_x = ", extended_start_x, ", extended_start_z = ", extended_start_z,
                  ", extended_end_x = ", extended_end_x, ", extended_end_z = ", extended_end_z,
                  ", extended_delta_x = ", extended_delta_x, ", extended_delta_z = ", extended_delta_z)
            print("line angle degrees = ", math.atan2(extended_end_x - extended_start_x, extended_end_z - extended_start_z) * 57.3)
        
        #cv2.imshow('Occupancy_map', self.occupancy_map)
        #cv2.waitKey(0)
        '''
            if extended_start_x < 0:
                extended_start_x = 0
            elif extended_start_x >= len(self.occupancy_map) - 1:
                extended_start_x = len(self.occupancy_map) - 1
            if extended_start_z < 0:
                extended_start_z = 0
            elif extended_start_z >= len(self.occupancy_map) - 1:
                extended_start_z = len(self.occupancy_map) - 1
            if extended_end_x < 0:
                extended_end_x = 0
            elif extended_end_x >= len(self.occupancy_map) - 1:
                extended_end_x = len(self.occupancy_map) - 1
            if extended_end_z < 0:
                extended_end_z = 0
            elif extended_end_z >= len(self.occupancy_map) - 1:
                extended_end_z = len(self.occupancy_map) - 1
        '''

            
        '''
        if end_x - start_x != 0:
            slope = (end_z - start_z) / (end_x - start_x)
           
        if in_front:
            rectangle_start_x = int(start_x + 0.5 - (thickness * math.sin(self.yaw)))
            rectangle_start_z = int(start_z + 0.5 - (thickness * math.cos(self.yaw)))
            print(" r_x = ", rectangle_x)
        else:
            rectangle_start_x = int(start_x + 0.5 + (thickness * math.sin(self.yaw)))
            rectangle_start_z = int(start_z + 0.5 + (thickness * math.cos(self.yaw)))
            print(" r_x1 = ", rectangle_x)
            
        rectangle_end_x = rectangle_start_x + (length * slope)
        rectangle_end_z = rectangle_start_z + (length * slope)

        print("adding obstacle, start_x = ", start_x, ", start_z = ", start_z, ", delta_x = ", delta_x, ", delta_z = ", delta_z,
              ", end_x = ", end_x, ", end_z = ", end_z,
              ", rectangle_x = ", rectangle_x, ", rectangle_z = ", rectangle_z, ", yaw degrees = ", self.yaw * 57.3)
        
        #thickness = 2
        #if wall:
        #    thickness = 7
        if in_front:
            rectangle_end_x = int(end_x + 0.5 + (thickness * math.sin(self.yaw + 1.57)))
            rectangle_end_z = int(end_z + 0.5 + (thickness * math.cos(self.yaw)))
        else:
            rectangle_end_x = int(end_x + 0.5 - (thickness * math.sin(self.yaw + 1.57)))
            rectangle_end_z = int(end_z - 0.5 - (thickness * math.cos(self.yaw)))
        
            
        #if the rectangle goes past an edge, just draw a line
        draw_rectangle = True
        if rectangle_x < 0 or rectangle_x >= len(self.occupancy_map) - 1 or rectangle_z >= len(self.occupancy_map) - 1 or rectangle_z < 0:
            print("not drawing a rectangle because it extends past the borders of the map")
            draw_rectangle = False
        else:
            if rectangle_x < 0:
                rectangle_x = 0
            elif rectangle_x >= len(self.occupancy_map) - 1:
                rectangle_x = len(self.occupancy_map) - 1
            if rectangle_z < 0:
                rectangle_z = 0
            elif rectangle_z >= len(self.occupancy_map) - 1:
                rectangle_z = len(self.occupancy_map) - 1

        #cv2.rectangle(self.occupancy_map, (start_x, start_z), (rectangle_end_x, rectangle_end_z), 1, -1)
        if draw_rectangle:
            cv2.rectangle(self.occupancy_map, (end_x, end_z), (rectangle_x, rectangle_z), 1, -1)
        else:
            cv2.line(self.occupancy_map, (start_x, start_z), (end_x, end_z), 1, 2)
        #cv2.rectangle(self.occupancy_map, (start_x, start_z), (rectangle_end_x, rectangle_end_z), 1, -1)
        #cv2.line(self.occupancy_map, (start_x, start_z), (end_x, end_z), 1, thickness)
        if True: #wall:
            #put in a parallel one 40m away
            other_wall_start_x = int(start_x + 0.5 + (self.arena_size * math.sin(self.yaw)))
            other_wall_start_z = int(start_z + 0.5 + (self.arena_size * math.cos(self.yaw)))
            other_wall_end_x = int(end_x + 0.5 + (self.arena_size * math.sin(self.yaw)))
            other_wall_end_z = int(end_z + 0.5 + (self.arena_size * math.cos(self.yaw)))
            if in_front:
                rectangle_end_x = int(other_wall_end_x + 0.5 + (thickness * math.sin(self.yaw)))
                rectangle_end_z = int(other_wall_end_z + 0.5 - (thickness * math.cos(self.yaw)))
            else:
                rectangle_end_x = int(other_wall_end_x + 0.5 - (thickness * math.sin(self.yaw)))
                rectangle_end_z = int(other_wall_end_z - 0.5 + (thickness * math.cos(self.yaw)))

            if other_wall_start_x < 0:
                other_wall_start_x = 0
            elif other_wall_start_x >= len(self.occupancy_map) - 1:
                other_wall_start_x = len(self.occupancy_map) - 1
            if other_wall_start_z < 0:
                other_wall_start_z = 0
            elif other_wall_start_z >= len(self.occupancy_map) - 1:
                other_wall_start_z = len(self.occupancy_map) - 1
            if rectangle_end_x < 0:
                rectangle_end_x = 0
            elif rectangle_end_x >= len(self.occupancy_map) - 1:
                rectangle_end_x = len(self.occupancy_map) - 1
            if rectangle_end_z < 0:
                rectangle_end_z = 0
            elif rectangle_end_z >= len(self.occupancy_map) - 1:
                rectangle_end_z = len(self.occupancy_map) - 1
            print("adding obstacle, other_wall_start_x = ", other_wall_start_x, ", other_wall_start_z = ", other_wall_start_z,
                  ", other_wall_end_x = ", other_wall_end_x, ", other_wall_end_z = ", other_wall_end_z,
                  ", rectangle_end_x = ", rectangle_end_x, ", rectangle_end_z = ", rectangle_end_z, ", yaw degrees = ", self.yaw * 57.3)
            #cv2.rectangle(self.occupancy_map, (other_wall_start_x, other_wall_start_z), (rectangle_end_x, rectangle_end_z), 1, -1)
            cv2.line(self.occupancy_map, (other_wall_start_x, other_wall_start_z), (other_wall_end_x, other_wall_end_z), 1, 2)
        #cv2.rectangle(self.occupancy_map, (56, 2), (17, 17), 1, -1)
        cv2.waitKey(0)
        for i in range(thickness):
            for j in range(start_x, end_x + 1):
                for m in range(start_z, end_z + 1):
                    if wall:
                        if m+i >= 0 and m+i < len(self.occupancy_map) and j+i >= 0 and j+i < len(self.occupancy_map):
                            self.occupancy_map[m+i][j+i] = 1
                    else:
                        if m-i >= 0 and m-i < len(self.occupancy_map) and j-i >= 0 and j-i < len(self.occupancy_map):
                            self.occupancy_map[m-i][j-i] = 1
        '''    

    def find_astar_path_to_green_seen_location(self):
        print("searching for the shortest path to a green_seen location.  Number of green_seen locations = ", len(self.green_seen_locations))
        previous_green_seen_x = previous_green_seen_z = -10
        shortest_path_length = 10000
        shortest_path = []
        for i in range(len(self.green_seen_locations)):
            if (previous_green_seen_x >= self.green_seen_locations[i][0] - 1 and previous_green_seen_x <= self.green_seen_locations[i][0] + 1
                and previous_green_seen_z >= self.green_seen_locations[i][1] - 1 and previous_green_seen_z <= self.green_seen_locations[i][1]):
                #same location, different yaw, no need to check
                print("this green_seen location is the same as the previous one, except for yaw, moving on")
                continue
            path = self.astar((self.x, self.z), (self.green_seen_locations[i][0], self.green_seen_locations[i][1]), self.occupancy_map)
            previous_green_seen_x = self.green_seen_locations[i][0]
            previous_green_seen_z = self.green_seen_locations[i][1]
            if path is not None:
                print("green seen path length = ", len(path), ", i = ", i)
                if len(path) < shortest_path_length:
                    shortest_path_length = len(shortest_path)
                    shortest_path = path
                    print("green seen shortest path length = ", len(shortest_path), ", i = ", i)
        if len(shortest_path) == 0:
            print("unable to find a path to a green seen location")
            return None
        print("path to a green seen location found, path length = ", len(shortest_path))
        if self.display_map:
            astar_map = self.occupancy_map.copy()
            for points in shortest_path:
                astar_map[int(points[1] + 0.5)][int(points[0] + 0.5)] = 1
            cv2.imshow("A*", astar_map)
            cv2.waitKey(10)
        return shortest_path
        


if __name__ == "__main__":
    env = init_environment()
    arena_config_in = ArenaConfig('../configs/3-Obstacles.yaml')
    env.reset(arenas_configurations=arena_config_in)
    m = Mapper(40, 0.0606, True)
    user_input = USER_INPUT
    auto_forward = False
    for i in range(8):
        res = env.step([0.0, 0.0])
        velocities = res['Learner'].vector_observations[0]
        #print("velocities:", velocities)
    print("agent should have finished dropping onto the floor by now")

    previous_velocity = 0.0
    distance = 0.0
    num_actions = 0
    num_turns = num_moves = 0
    astar_actions = False
    path = []
    while True:
        num_actions += 1
        if user_input:
            try:
                move_action = int(input("enter movement 0-8 (left(0), left-forward(1), forward(2),"
                                        " right-forward(3), right(4), right-backward(5), backward(6), left-backward(7), none(8): "))
            except ValueError:
                print("command is out of range, try again")
                continue            
            if move_action < 0:
                break
            if move_action == 9:
                # start coordinates are in image format (vertical, horizontal) with (0,0) at the top right
                start = (m.z, m.x)
                
                target_list_green_goal_found = False
                if len(m.target_list) > 0:
                    max_times_seen = 0
                    for target in m.target_list:
                        #target list: x, z, diameter, color, number of times seen
                        if target[3] == 1:  #green target
                            if target[4] > max_times_seen:
                                goal = (target[0], target[1])
                                max_times_seen = target[4]
                                target_list_green_goal_found = True
                if not target_list_green_goal_found:
                    if len(m.gold_occupancy_list) > 0:
                        goal = m.gold_occupancy_list[0]
                    elif len(m.green_occupancy_list) > 0:
                        goal = m.green_occupancy_list[0]
                    else:
                        goal = (30, 10)  # 10 rows from the top, 30 columns from the left
                path = m.astar(start, goal, m.occupancy_map, True)
                if path is not None:
                    #print("astar path from current location to ", goal, " is: ", path)
                    num_turns, num_moves, num_back, path = m.follow_path(path)
                    user_input = False
                    astar_actions = True
                move_action = 8 # no move, just to complete this round of the loop
            if move_action == 99:
                auto_forward = True
                user_input = False
                move_action = 2
            action = m.transform_network_action_to_model_action(move_action)
        else:
            if astar_actions:
                if num_turns > 0:
                    action = [0,2] #left
                    num_turns -= 1
                elif num_turns < 0:
                    action = [0,1] #right
                    num_turns += 1
                else:
                    if num_moves > 0:
                        action = [1, 0] #forward
                        num_moves -= 1
                    elif num_moves == 0:
                        if num_back > 0:
                            action = [2, 0] #backward to help come to a stop
                            num_back -= 1
                        if num_back == 0:  #do not want elif here because we can go ahead and take the last back move below
                            if len(path) == 0:
                                print("astar move complete")
                                user_input = True
                                astar_actions = False
                            else:
                                #get the turns and moves for the next line segment
                                num_turns, num_moves, num_back, path = m.follow_path(path)
                        
            elif num_actions > 50:
                break
            elif num_actions <= 10 and (not auto_forward): #first turn 60 degrees
                action = [0,1]
            else:
                action = [1,0]

        res = env.step(action)
        done = res['Learner'].local_done[0]
        if done:
            print("episode is done, resetting")
            user_input = USER_INPUT
            auto_forward = False
            for i in range(8):
                res = env.step([0.0, 0.0])
                velocities = res['Learner'].vector_observations[0]
                # print("velocities:", velocities)
            print("agent should have finished dropping onto the floor by now")
            previous_velocity = 0.0
            distance = 0.0
            num_actions = 0
            num_turns = num_moves = 0
            astar_actions = False
            path = []
            display_search_progress_image = np.zeros((80, 80, 3))
            user_input = True
            astar_actions = False
            m.reset()
            res = env.step(action)
        velocities = res['Learner'].vector_observations[0]
        delta_distance = ((velocities[2] + previous_velocity) / 2.0) * 0.0606
        distance += delta_distance
        previous_velocity = velocities[2]
        image = res['Learner'].visual_observations[0][0]
        array = image * 255
        frame = array.astype(np.uint8)
        m.frame = frame
        m.update(velocities, action, True)
        #print("vel x = {0:.4f}, vel z = {1:.4f}, delta_distance = {2:.2f}, total distance = {3:.2f}".format(velocities[0], velocities[2], delta_distance, distance))


    #print("x = {0:.2f}, z = {1:.2f}, yaw = {2:.2f}, yaw degrees = {3:.1f}".format(m.x, m.z, m.yaw, m.yaw * 57.3))
    #print("x distance from origin = {0:.2f}, z distance from origin = {1:.2f}".format(m.x - m.origin_x, m.origin_z - m.z))
    m.show_occupancy_map()
    env.close()
    print("all done")
    exit(0)
