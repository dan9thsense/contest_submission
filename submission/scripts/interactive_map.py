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

ARENA_SIZE = 80
#DELTA_TIMESTEP = 0.0595 #this value is from the issues page
DELTA_TIMESTEP = 0.0606 #this value was determined locally
SHOW_MAP = True
GRID_IMAGE_SIZE = 840
GRID_CELL_SIZE = int(840/40)
USER_INPUT = True
DISPLAY_OLD_MAP = False

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

class Mapper:
    def __init__(self):
        self.reset()

    def __del__(self):
        if self.map_window_created:
            cv2.destroyWindow('Arena')

    def reset(self):
        self.map = np.ones((ARENA_SIZE, ARENA_SIZE))
        #map to store visited / obstacle cells
        self.grid_map = np.zeros((ARENA_SIZE, ARENA_SIZE))
        #grid image for visualization
        self.grid_image = np.ones((GRID_IMAGE_SIZE*2, GRID_IMAGE_SIZE*2, 3))*255
        self.map *= 0.5
        self.origin_x = float(ARENA_SIZE) / 2.0
        self.origin_z = float(ARENA_SIZE) / 2.0
        self.x = float(ARENA_SIZE) / 2.0
        self.z = float(ARENA_SIZE) / 2.0
        self.yaw = 0.0
        self.previous_x = float(ARENA_SIZE) / 2.0
        self.previous_z = float(ARENA_SIZE) / 2.0
        self.previous_yaw = 0.0
        self.previous_heading = 0.0
        self.previous_vel_x = 0.0
        self.previous_vel_z = 0.0
        self.wall_points = []
        self.previous_wall = False
        self.map_window_created = False
        self.create_grid_image()

    #create images with grids, each grid size is GRID_CELL_SIZE x GRID_CELL_SIZE
    def create_grid_image(self):
        for i in range(0, GRID_IMAGE_SIZE*2, GRID_CELL_SIZE):
            for j in range(0, GRID_IMAGE_SIZE*2, GRID_CELL_SIZE):
                cv2.rectangle(self.grid_image, (i, j), (i + GRID_CELL_SIZE, j + GRID_CELL_SIZE), (0, 0, 0), 3)

    def mark_cell(self, x, y, color):
        if color == 'green': #visited cell
            cv2.rectangle(self.grid_image, (x, y), (x + GRID_CELL_SIZE, y + GRID_CELL_SIZE),
                                    (0, 255, 0), cv2.FILLED)
        if color == 'blue': #revisited cell
            cv2.rectangle(self.grid_image, (x, y), (x + GRID_CELL_SIZE, y + GRID_CELL_SIZE),
                                    (255, 0, 0), cv2.FILLED)
        if color == 'red': #obstacle
            cv2.rectangle(self.grid_image, (x, y), (x + GRID_CELL_SIZE, y + GRID_CELL_SIZE),
                                    (0, 0, 255), cv2.FILLED)
        if color == 'cyan':
            cv2.rectangle(self.grid_image, (x, y), (x + GRID_CELL_SIZE, y + GRID_CELL_SIZE),
                                    (255, 255, 0), cv2.FILLED)
        if color == 'magenta': #cells between agent and target
            cv2.rectangle(self.grid_image, (x, y), (x + GRID_CELL_SIZE, y + GRID_CELL_SIZE),
                                    (255, 0, 255), cv2.FILLED)
        grid_image_copy = self.grid_image.copy()
        cv2.imshow('grid image', grid_image_copy)

    def update(self, velocities, action, move_turn_data, display_map = False):
        delta_yaw = 0.0
        if action[1] == 1:
            delta_yaw = -0.1047  # 6 degrees cw
        elif action[1] == 2:
            delta_yaw = 0.1047  # 6 degrees ccw
        self.yaw = self.previous_yaw + delta_yaw

        vel_x = velocities[0]
        vel_z = velocities[2]
        #deltaT = time.time() - self.previous_time
        deltaT = DELTA_TIMESTEP
        #self.previous_time = time.time()
        #need to convert from agent reference frame to arena reference frame
        #we use negative vel_z because opencv increases index values as you move from top to bottom
        #but the arena increases index values as you move bottom to top
        arena_vel_x = (vel_x * math.cos(self.previous_yaw)) + (-vel_z * math.sin(self.yaw))
        arena_vel_z = (-vel_z * math.cos(self.previous_yaw)) + (vel_x * math.sin(self.yaw))
        #print("vel_x = {0:.2f}, vel_z = {1:.2f}, arena vel_x = {2:.2f}, arena vel_z = {3:.2f}".format(vel_x, vel_z, arena_vel_x, arena_vel_z))

        delta_x = ((self.previous_vel_x + arena_vel_x) / 2.0) * deltaT
        delta_z = ((self.previous_vel_z + arena_vel_z) / 2.0) * deltaT
        #use converted coordinates to calculate the move
        self.x = self.previous_x + delta_x
        self.z = self.previous_z + delta_z
        delta_distance = math.sqrt((delta_x * delta_x) + (delta_z*delta_z))
        total_distance_from_origin = math.sqrt(((self.x - self.origin_x)*(self.x - self.origin_x)) + (
        (self.z - self.origin_z)*(self.z - self.origin_z)))

        #yaw_degrees = yaw * 57.3
        #print("arena vel x = {0:.4f}, arena vel z = {1:.4f}, yaw = {2:.4f}, delta_distance = {3:.4f}, total_distance from origin = {4:.4f}, x = {5:.2f}, z = {6:.2f}".format(
        #    arena_vel_x, arena_vel_z, self.yaw, delta_distance, total_distance_from_origin, self.x, self.z))

        if abs(vel_x) < 0.01 and abs(vel_z) < 0.01:
            if abs(self.previous_x) > 2.0 or abs(self.previous_z) > 2.0 or action[0] != 0:
                #we found a wall (or this is the first point)
                #the matrix is (rows, cols) while the image has x has the horizontal axis (cols)
                #so we have to swap the indicies if we want to plot this on the image directly
                self.wall_points.append((int(self.z + 0.5), int(self.x + 0.5)))

        #add 0.5 because the int() function truncates the float
        self.update_map(int(self.x + 0.5), int(self.previous_x + 0.5), int(self.z + 0.5), int(self.previous_z + 0.5), move_turn_data)
        #print("x = {0:.2f}, z = {1:.2f}, yaw degrees = {2:.1f}, deltaT = {3:.2f}".format(x, z, yaw * 57.3, deltaT))
        self.previous_x = self.x
        self.previous_z = self.z
        self.previous_yaw = self.yaw
        self.previous_vel_x = arena_vel_x
        self.previous_vel_z = arena_vel_z
        if display_map:
            if DISPLAY_OLD_MAP:
                self.show_map(10)
            else:
                self.show_grid_map(10)

    def update_map(self, x, previous_x, z, previous_z, move_turn_data):
        #find all the points between our previous position and our current position
        cv2.line(self.map, (previous_x, previous_z), (x, z), (0, 0, 0), 1)
        if self.grid_map[x, z] == 1 and move_turn_data['turn'] == 0:
            #mark the cells revisited blue
            #print('revisiting : {}, {}'.format(x, z))
            self.mark_cell(x * GRID_CELL_SIZE, z * GRID_CELL_SIZE, 'blue')
        else:
            #draw the marked cells on the grid image, newly visited cells are marked green
            self.grid_map[x, z] = 1
            self.mark_cell(x * GRID_CELL_SIZE, z * GRID_CELL_SIZE, 'green')
        #print(x, z)
        #self.map = self.rotate_bound(self.map, 1.57)

    def mark_obstacle(self, x, z, direction):
        #find all the points between our previous position and our current position
        #print('angle = {}'.format(-self.yaw*57.3))
        if direction == 'left':
            left_cell_x = x - math.cos(-self.yaw)
            left_cell_x = [np.ceil(left_cell_x) if left_cell_x%1 >= 0.5 else np.floor(left_cell_x)]

            left_cell_z = z - math.sin(-self.yaw)
            left_cell_z = [np.ceil(left_cell_z) if left_cell_z % 1 >= 0.5 else np.floor(left_cell_z)]

            left_cell_x = int(left_cell_x[0])
            left_cell_z = int(left_cell_z[0])
            self.grid_map[left_cell_x, left_cell_z] = -1
            self.mark_cell(left_cell_x * GRID_CELL_SIZE, left_cell_z * GRID_CELL_SIZE, 'red') #obstacles are marked red
            #print("left x, z = {} {}".format(left_cell_x, left_cell_z))

        if direction == 'right':
            right_cell_x = x + math.cos(-self.yaw)
            right_cell_x = [np.ceil(right_cell_x) if right_cell_x % 1 >= 0.5 else np.floor(right_cell_x)]

            right_cell_z = z + math.sin(-self.yaw)
            right_cell_z = [np.ceil(right_cell_z) if right_cell_z % 1 >= 0.5 else np.floor(right_cell_z)]

            right_cell_x = int(right_cell_x[0])
            right_cell_z = int(right_cell_z[0])
            self.grid_map[right_cell_x, right_cell_z] = -1
            self.mark_cell(right_cell_x * GRID_CELL_SIZE, right_cell_z * GRID_CELL_SIZE, 'red') #obstacles are marked red
            #print("right  x, z = {} {}".format(right_cell_x, right_cell_z))

        if direction == 'front':
            front_cell_x = x + math.sin(-self.yaw)
            front_cell_z = z - math.cos(self.yaw)
            #print("front  x, z = {} {}".format(front_cell_x, front_cell_z))
            front_cell_x = [np.ceil(front_cell_x) if front_cell_x % 1 >= 0.5 else np.floor(front_cell_x)]
            front_cell_z = [np.ceil(front_cell_z) if front_cell_z % 1 >= 0.5 else np.floor(front_cell_z)]
            front_cell_x = int(front_cell_x[0])
            front_cell_z = int(front_cell_z[0])
            self.grid_map[front_cell_x, front_cell_z] = -1
            self.mark_cell(front_cell_x * GRID_CELL_SIZE, front_cell_z * GRID_CELL_SIZE, 'red') #obstacles are marked red

        grid_image_copy = self.grid_image.copy()
        cv2.imshow('grid image', grid_image_copy)

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

    #marks the current cell if a target is visible, 2 for green, 3 for gold
    def update_target_seen_from(self, color):
        if color == 'green':
            self.grid_map[self.x, self.z] = 2
        if color == 'gold':
            self.grid_map[self.x, self.z] = 3

    #marks all the cells between the cell and target with the value 4
    #it could potentially mean that the these are free
    #distance_in_pixels - distance from the agent to the base of the ball
    def mark_all_cells_till_target(self, distance_in_pixels, angle = 0):

        if distance_in_pixels <=28:
            distance_in_meters = 2
        if distance_in_pixels > 28 and distance_in_pixels <= 35:
            distance_in_meters = 3
        if distance_in_pixels > 35:
            distance_in_meters = 8
        for i in range(1, int(distance_in_meters)):
            front_cell_x = self.x + i*math.sin(-self.yaw)
            front_cell_x = [np.ceil(front_cell_x) if front_cell_x % 1 >= 0.5 else np.floor(front_cell_x)]

            front_cell_z = self.z - i*math.cos(-self.yaw)
            # print("front  x, z = {} {}".format(front_cell_x, front_cell_z))
            front_cell_z = [np.ceil(front_cell_z) if front_cell_z % 1 >= 0.5 else np.floor(front_cell_z)]
            front_cell_x = int(front_cell_x[0])
            front_cell_z = int(front_cell_z[0])
            self.grid_map[front_cell_x, front_cell_z] = 4
            self.mark_cell(front_cell_x * GRID_CELL_SIZE, front_cell_z * GRID_CELL_SIZE, 'magenta') #cells between agent and target are magenta


    def show_map(self, wait_time = 0):
        if not self.map_window_created:
            cv2.namedWindow('Arena', cv2.WINDOW_NORMAL)
            self.map_window_created = True
        arrow_length = 10.0
        show_map = self.map.copy()
        current_location = (int(self.previous_x + 0.5), int(self.previous_z + 0.5))
        #opencv positive angle goes cw but our convention is for positive angle to go ccw
        #so we use negative angle for the display (sin only, since cos does not change with sign)
        yaw_arrow_end_x = int((self.previous_x + arrow_length * math.sin(-self.previous_yaw)) + 0.5)
        yaw_arrow_end_z = int((self.previous_z - arrow_length * math.cos(self.previous_yaw)) + 0.5)
        yaw_arrow_end = (yaw_arrow_end_x, yaw_arrow_end_z)
        show_map = cv2.arrowedLine(show_map, current_location, yaw_arrow_end, 255)
        for i in self.wall_points:
            if i[0] <= ARENA_SIZE - 1 and i[0] >= 0 and i[1] <= ARENA_SIZE - 1 and i[1] >= 0:
                show_map[i[0], i[1]] = 1.0
        cv2.imshow('Arena', show_map)
        cv2.waitKey(wait_time)

    def show_grid_map(self, wait_time = 0):
        if not self.map_window_created:
            cv2.namedWindow('Arena', cv2.WINDOW_NORMAL)
            self.map_window_created = True
        arrow_length = 5.0
        show_map = self.map.copy()
        current_location = (int(self.previous_x + 0.5), int(self.previous_z + 0.5))
        #opencv positive angle goes cw but our convention is for positive angle to go ccw
        #so we use negative angle for the display (sin only, since cos does not change with sign)
        yaw_arrow_end_x = int((self.previous_x + arrow_length * math.sin(-self.previous_yaw)) + 0.5)
        yaw_arrow_end_z = int((self.previous_z - arrow_length * math.cos(self.previous_yaw)) + 0.5)
        yaw_arrow_end = (yaw_arrow_end_x, yaw_arrow_end_z)
        show_map = cv2.arrowedLine(show_map, current_location, yaw_arrow_end, 255)
        grid_image_copy = self.grid_image.copy()
        #draw the arrow onto grid image
        cv2.arrowedLine(grid_image_copy, (current_location[0]*GRID_CELL_SIZE + 10, current_location[1]*GRID_CELL_SIZE + 10),
                        (yaw_arrow_end_x * GRID_CELL_SIZE + 10, yaw_arrow_end_z * GRID_CELL_SIZE + 10), (0, 255, 255), 5)
        for i in self.wall_points:
            if i[0] <= ARENA_SIZE - 1 and i[0] >= 0 and i[1] <= ARENA_SIZE - 1 and i[1] >= 0:
                show_map[i[0], i[1]] = 1.0
        cv2.imshow('Arena', show_map)
        cv2.imshow('grid image', grid_image_copy)
        cv2.waitKey(wait_time)

if DISPLAY_OLD_MAP == True:
    cv2.namedWindow('Arena', cv2.WINDOW_NORMAL)
else:
    cv2.namedWindow('grid image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('grid image', 600,600)

if __name__ == "__main__":
    env = init_environment()
    arena_config_in = ArenaConfig('../configs/distance_test.yaml')
    env.reset(arenas_configurations=arena_config_in)
    m = Mapper()
    user_input = USER_INPUT
    auto_forward = False
    for i in range(8):
        res = env.step([0.0, 0.0])
        velocities = res['Learner'].vector_observations[0]
        #print("velocities:", velocities)
    print("agent should have finished dropping onto the floor by now")

    previous_velocity = 0.0
    distance = 0.5
    num_actions = 0

    move_action = 0
    turn_action = 0
    while True:
        num_actions += 1
        if user_input:
            #move_action = int(input("enter forward (1), back (2), or none (0): "))
            pressed_key = cv2.waitKey(10)
            if pressed_key == ord('w'):
                move_action = 1
            if pressed_key == ord('s'):
                move_action = 2
            if pressed_key == ord('d'):
                turn_action = 1
            if pressed_key == ord('a'):
                turn_action = 2
            if pressed_key == ord('j'):
                m.mark_obstacle(int(m.x), int(m.z), 'left')
            if pressed_key == ord('i'):
                m.mark_obstacle(int(m.x), int(m.z), 'front')
            if pressed_key == ord('l'):
                m.mark_obstacle(int(m.x), int(m.z), 'right')
            if pressed_key == ord('t'):
                m.update_target_seen_from('green')
            if pressed_key == ord('m'):
                m.mark_all_cells_till_target(42, 0)
            if pressed_key == ord('q'):
                break;
            if turn_action or move_action:
                if move_action < 0:
                    break
                if move_action == 99:
                    auto_forward = True
                    user_input = False
                    move_action = 1
                    turn_action = 0
                action = [move_action, turn_action]
                res = env.step(action)
                velocities = res['Learner'].vector_observations[0]
                delta_distance = ((velocities[2] + previous_velocity) / 2.0) * DELTA_TIMESTEP
                distance += delta_distance
                previous_velocity = velocities[2]
                # print("vel z = {0:.4f}, vel x = {1:.4f}, delta_distance = {2:.2f}, total distance = {3:.2f}".format(velocities[2], velocities[0], delta_distance, distance))
                move_turn_data = {'move' : move_action, 'turn' : turn_action}
                m.update(velocities, action, move_turn_data, SHOW_MAP)
                move_action = 0
                turn_action = 0
    print("x = {0:.2f}, z = {1:.2f}, yaw = {2:.2f}, yaw degrees = {3:.1f}".format(m.x, m.z, m.yaw, m.yaw * 57.3))
    print("x distance from origin = {0:.2f}, z distance from origin = {1:.2f}".format(m.x - m.origin_x, m.origin_z - m.z))
    if SHOW_MAP:
        if DISPLAY_OLD_MAP:
            m.show_map()
        else:
            m.show_grid_map()
    env.close()
    print("all done")
    exit(0)
