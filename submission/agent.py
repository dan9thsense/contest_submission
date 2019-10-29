#the 6 inputs are
#inputs from opencv detector: radius, center
#current velocities

#ouputs are 9 possible moves

from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig
from animalai.envs import UnityEnvironment
from animalai.envs.exception import UnityEnvironmentException

import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import random
import math

import yaml
import sys
from time import sleep
import cv2
import scipy.misc
import math
#import imutils
from datetime import datetime

import sys

DOCKER = True
configuration_yaml_file = 'permanent_blackout_with_wall_and_bad_goal'
NUM_EPISODES = 30
MULTIPLE_CONFIGS = True
MULTIPLE_CONFIGS_SAVE_FILENAME = 'new_explore_moves'
PARAMETER_RANGE = False

WATCH_IMAGE = True
INFERENCE = True
USE_MAP = True
USE_OCCUPANCY_MAP = False
USE_TARGET_MAPS = False
SHOW_MAP = True

NUM_STATES = 8
NUM_ACTIONS = 9

MAZE_NUM_STATES = 7
MAZE_NUM_ACTIONS = 9

USE_MAZE_NETWORK = False

if MULTIPLE_CONFIGS or PARAMETER_RANGE:
    WATCH_IMAGE = False
    INFERENCE = False
    SHOW_MAP = False

if DOCKER:
    MULTIPLE_CONFIGS = True
    WATCH_IMAGE = False
    SHOW_MAP = False
    INFERENCE = False
    SHOW_MAP = False
    sys.path.append('/aaio/scripts')
    from process_image_working_docker import ProcessImage
    from select_action_working import SelectAction
    from map_working import Mapper
else:
    from scripts.process_image_working import ProcessImage
    from scripts.select_action_working import SelectAction
    from scripts.map_working import Mapper

TIMESTEPS = 1000 #note that this is just a default value for when t=0.  The actual timesteps are calculated from the reward
EPSILON = 0.01
NUM_INITIAL_INSPECTION_STEPS = 52

#when initial inspection is done, we will move forward this many times before turning
#with obstacles, we want long moves
MAX_NUM_NO_TARGET_MOVES_WITH_OBSTACLES = 50
MAX_NUM_NO_TARGET_MOVES_LOTS_OF_TIME = 30
MAX_NUM_NO_TARGET_MOVES = 25
MAX_NUM_NO_TARGET_MOVES_NEAR_TIMEOUT = 20
#when we cannot see a target, we will turn to look around for up to this many turns, at 6 degrees per turn
MAX_NUM_NO_TARGET_TURNS_WITH_OBSTACLES = 76 #wall follow to the right and look completely around at each stop
MAX_NUM_NO_TARGET_TURNS_LOTS_OF_TIME = 50
MAX_NUM_NO_TARGET_TURNS = 50
MAX_NUM_NO_TARGET_TURNS_NEAR_TIMEOUT = 50

#accept green at 249 - num_initial_inspection_steps because episode steps starts at 1
#and self.initial_inspection_steps_completed starts at 0
#this way we start accepting green when the initial inspection steps are complete (for self.timesteps = 250)
#and we will turn back to take one if it was seen during the initial inspection
#MAX_STEPS_TO_ACCEPT_GREEN = 249 - NUM_INITIAL_INSPECTION_STEPS
MAX_STEPS_TO_ACCEPT_GREEN = 325 #499 - NUM_INITIAL_INSPECTION_STEPS

MAX_NUM_STEPS_TO_REDUCE_TURNS = 0
MAX_OSCILLATIONS = 3
MAX_STUCK_NUMBER = 3
MIN_RED_RADIUS = 13.0  #smallest radius that we will use to avoid targets, see data in data/parameter_ranges/min_red_radius_4-Avoidance.txt
MIN_GREEN_RADIUS = 20.0  #smallest radius that we will use to avoid targets
MIN_HOTZONE_SIZE = 5.0
MAX_NUM_STEPS_TO_ENTER_HOTZONES = 249 - NUM_INITIAL_INSPECTION_STEPS
MAX_PIXEL = 84
MAX_VEL = 12.0
MAX_RADIUS = 30.0
#DELTA_TIMESTEP = 0.0595 #this value is from the issues page
DELTA_TIMESTEP = 0.0606 #this value was determined locally
ARENA_SIZE = 40


USING_OLD_GREEN_NETWORK = False
USING_OLD_RED_NETWORK = False
if DOCKER:
    #load_model_path_green = '/aaio/data/just_food/just_food-1024' #remember that this is an OLD_NETWORK
    #load_model_path_red = '/aaio/data/preference/preference.ckpt' #remember that this is an OLD_NETWORK
    if USE_MAZE_NETWORK:
        load_model_path_maze = '/aaio/trained_networks/maze/maze.ckpt'

    load_model_path_green = '/aaio/trained_networks/random_location_and_size/random_location_and_size.ckpt'
    load_model_path_red = '/aaio/trained_networks/random_location_bad_goal/random_location_bad_goal.ckpt'

else:
    arena_config_in = ArenaConfig('configs/' + configuration_yaml_file + '.yaml')
    #load_model_path_green = 'data/just_food/just_food-1024'  #remember that this is an OLD_NETWORK
    #load_model_path_red = 'data/preference/preference.ckpt'  #remember that this is an OLD_NETWORK
    load_model_path_green = 'trained_networks/random_location_and_size/random_location_and_size.ckpt'
    load_model_path_red = 'trained_networks/random_location_bad_goal/random_location_bad_goal.ckpt'
    if USE_MAZE_NETWORK:
        load_model_path_maze = 'trained_networks/maze/maze.ckpt'

ActionNames = ["l", "fl", "f", "fr", "r", "br", "b", "bl", "n"]

def init_environment():
    return UnityEnvironment(
        file_name='test_submission/env/AnimalAI',  #Path to the environment
        worker_id=random.randint(1, 100),  #Unique ID for running the environment (used for connection)
        seed=10,  #The random seed
        docker_training=DOCKER,  #Whether or not you are training inside a docker
        #no_graphics=False,          #Always set to False
        n_arenas=1,  #Number of arenas in your environment
        play=False,  #Set to False for training
        inference=INFERENCE,  #Set to true to watch your agent in action
        resolution=MAX_PIXEL  #Int: resolution of the agent's square camera (in [4,512], default 84)
    )

class ImportGraph():
    """  Importing and running isolated TF graph """

    def __init__(self, loc, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        #Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            #Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, loc)

            #There are TWO options how to get activation operation:
            #FROM SAVED COLLECTION:
            self.activation = tf.get_collection('activation')[0]
            #BY NAME:
            #self.activation = self.graph.get_operation_by_name('activation_opt').outputs[0]

    #def predict_one(self, state):
    #  return self.sess.run(self.logits, feed_dict={self.states: state.reshape(1, self.num_states)})
    def predict_one(self, state):
        return self.sess.run(self.activation, feed_dict={'input_state:0': state.reshape(1, self.num_states)})

class GameRunner:
    def __init__(self):
        self.graph_green = ImportGraph(load_model_path_green, NUM_STATES, NUM_ACTIONS)
        self.graph_red = ImportGraph(load_model_path_red, NUM_STATES, NUM_ACTIONS)
        if USE_MAZE_NETWORK:
            self.graph_maze = ImportGraph(load_model_path_maze, MAZE_NUM_STATES, MAZE_NUM_ACTIONS)
        else:
            self.graph_maze = None
        self.process_image = ProcessImage(MAX_PIXEL, MIN_GREEN_RADIUS, MIN_RED_RADIUS, WATCH_IMAGE)
        #self.select_action = SelectAction(self.graph_green, USING_OLD_GREEN_NETWORK, self.graph_red, USING_OLD_RED_NETWORK, self.graph_maze, NUM_INITIAL_INSPECTION_STEPS,
        #          EPSILON, MAX_OSCILLATIONS, MAX_STUCK_NUMBER, USE_MAZE_NETWORK, MAX_NUM_NO_TARGET_MOVES, MAX_NUM_NO_TARGET_TURNS, MIN_RED_RADIUS, MIN_HOTZONE_SIZE, MIN_GREEN_RADIUS, MAX_PIXEL, DELTA_TIMESTEP, False)
        self.select_action = SelectAction(self.graph_green, USING_OLD_GREEN_NETWORK, self.graph_red, USING_OLD_RED_NETWORK, self.graph_maze, NUM_INITIAL_INSPECTION_STEPS,
                                          EPSILON, MAX_OSCILLATIONS, MAX_STUCK_NUMBER, USE_MAZE_NETWORK, MAX_NUM_NO_TARGET_MOVES, MAX_NUM_NO_TARGET_TURNS, MIN_RED_RADIUS, MIN_HOTZONE_SIZE,
                                          MIN_GREEN_RADIUS, MAX_PIXEL, DELTA_TIMESTEP, False)

        #self.explore = Explore(DELTA_TIMESTEP)
        if USE_MAP:
            self.map = Mapper(ARENA_SIZE, DELTA_TIMESTEP, USE_OCCUPANCY_MAP, USE_TARGET_MAPS, SHOW_MAP)
            
        self.episode = 1
        self.episode_when_obstacle_seen = 0
        self.episode_when_red_seen = 0
        self.episode_when_blackout_seen = 0
        self.blackout_seen_in_previous_episode = False
        self.set_values()

    def set_values(self):
        self.steps = 1
        self.total_targets_found = 0
        self.num_runs_that_timed_out = 0
        self.target_found_step_numbers = []
        self.last_action_taken_before_reward = []
        self.rewards = []
        self.total_reward = 0.0
        self.total_reward_count = 0.0
        self.previous_reward_step = 0
        self.actions_taken_frequency = np.zeros((9,), dtype=int)
        self.time_remaining = TIMESTEPS
        self.first_call = True
        self.use_target_maps = USE_TARGET_MAPS
        #self.initialize_values()

    def initialize_values(self):
        if USE_MAP:
            self.map.reset()
        self.action = [0.0, 0.0]
        self.previous_action = [0.0, 0.0]
        self.previous_velocity = 0.0
        self.reward_received = False
        self.accept_green = False
        self.obstacle_seen = False
        self.blackouts_happening = False
        self.blackout_seen = False
        self.gold_targets_present = False
        self.process_image.obstacle_seen = False
        self.episode_total_reward = 0.0
        self.max_num_no_target_turns = MAX_NUM_NO_TARGET_TURNS
        self.max_num_no_target_moves = MAX_NUM_NO_TARGET_MOVES
        self.episode_steps = 1
        self.timesteps = TIMESTEPS
        self.max_steps_to_accept_green = MAX_STEPS_TO_ACCEPT_GREEN
        self.previous_color = 0 #used in case of blackouts and we want to know whether to moe forward or not
        self.time_remaining_calculated = False
        self.red_seen = False
        self.astar_initial_stop = 0
        self.astar_actions = False
        self.astar_num_turns = self.astar_num_moves = self.astar_num_back = 0
        self.astar_path = None
        self.no_more_unexplored_areas_exist = False
        self.astar_done = False
        self.num_failed_astar_moves = 0
        self.astar_stuck = 0
        self.astar_check_in_new_direction = False
        self.agent_network_action = 8
        self.target_list_green_goal_found = False
        self.select_action.initialize_values()
        
        #if self.episode < self.episode_when_red_seen + 30 and self.episode_when_red_seen > 0:
        #    self.red_seen = True
        #    self.select_action.red_color_seen = True
        #if (self.episode < self.episode_when_obstacle_seen + 6 or self.obstacle_seen_in_previous_episode) and self.episode_when_obstacle_seen > 0:
        #    self.obstacle_seen = True
        #    self.process_image.obstacle_seen = True
        
        #we use two methods because sometimes we find a green target before the blackout starts, so we need to remember more than just the previous episode
        #we do not want to use episodes from too far back or we will be assuming blackouts for many cases where they do not arise
        #we want to use a memory distance that divides evenly into 30
        #also, once we get a blackout episode, we do not update the 6 until all 6 are done, so we need to know if the previous episode had a blackout
        if (self.episode < self.episode_when_blackout_seen + 6 or self.blackout_seen_in_previous_episode) and self.episode_when_blackout_seen > 0:
            self.blackout_seen = True
            if self.blackout_seen_in_previous_episode:
                print("blackout seen in previous episode, episode = ", self.episode)
            if self.episode < self.episode_when_blackout_seen + 6:
                print("blackout seen within the past 6 episodes (at episode = ", self.episode_when_blackout_seen, "), current episode = ", self.episode)
        else:
            print("no blackout seen, episode = ", self.episode)
    
    def run(self, visual_observations, vector_observations, reward, done, sent_timesteps):
        self.timesteps = sent_timesteps
        if self.first_call:
            self.first_call = False
            self.time_remaining = self.timesteps
            self.initialize_values()
            print("\nstarting episode {0:2d}".format(self.episode))
            
            if self.red_seen:
                print("setting red_seen to True for this episode")
            if self.obstacle_seen:
                print("setting obstacle_seen to True for this episode")
            if self.blackout_seen:
                self.accept_green = True
                print("self.blackout_seen is True, so we set accept_green to True for this episode")
            else:
                print("self.blackout_seen is False, self.episode_when_blackout_seen = ", self.episode_when_blackout_seen, ", current episode = ", self.episode)
        #if we are running locally, we need to figure out the length of the episode
        if not self.time_remaining_calculated:
            if not DOCKER:
                # determine the timestep from the reward
                if reward <= 0.0 and reward > -0.0041:  # we did not get a target reward, just a time reward
                    if reward >= -0.0000001:
                        self.timesteps = 10000 # in this case, the yaml file t=0 and there is no time limit
                    elif reward >= -0.0011: #reward is -.001 -> 1/1000
                        self.timesteps = 1000
                    elif reward >= -.0021: #reward is -.002 -> 1/500
                        self.timesteps = 500
                    elif reward >= -0.0041: #reward is -.004 -> 1/250
                        self.timesteps = 250
                    print("reward = {0:.4f}, timesteps calculated to be = {1:2d}".format(reward, self.timesteps))
                self.time_remaining_calculated = True

            if not self.gold_targets_present:
                if self.timesteps == 250:
                    self.max_steps_to_accept_green = 249 - NUM_INITIAL_INSPECTION_STEPS
                elif self.timesteps == 500:
                    self.max_steps_to_accept_green = 499 - ((2 * NUM_INITIAL_INSPECTION_STEPS) + 15)
                else:
                    self.max_steps_to_accept_green = 999 - ((3 * NUM_INITIAL_INSPECTION_STEPS) + 15)
            else:
                self.max_steps_to_accept_green = MAX_STEPS_TO_ACCEPT_GREEN

        self.time_remaining = self.timesteps - self.episode_steps
        print("time remaining = ", self.time_remaining, ", max_steps_to_accept_green = ", self.max_steps_to_accept_green)
        if self.time_remaining <= self.max_steps_to_accept_green:
            if not self.accept_green:
                print("time is getting short, so we will now accept green targets, remaining steps = {0:2d}".format(self.time_remaining))
            self.accept_green = True

        if self.obstacle_seen:
            self.select_action.max_num_no_target_moves = MAX_NUM_NO_TARGET_MOVES_WITH_OBSTACLES
            self.select_action.max_num_no_target_turns = MAX_NUM_NO_TARGET_TURNS_WITH_OBSTACLES

        elif self.time_remaining > 400:
            self.select_action.max_num_no_target_moves = MAX_NUM_NO_TARGET_MOVES_LOTS_OF_TIME
            self.select_action.max_num_no_target_turns = MAX_NUM_NO_TARGET_TURNS_LOTS_OF_TIME
        elif self.time_remaining > MAX_NUM_STEPS_TO_REDUCE_TURNS:
            self.select_action.max_num_no_target_moves = MAX_NUM_NO_TARGET_MOVES
            self.select_action.max_num_no_target_turns = MAX_NUM_NO_TARGET_TURNS
            if self.time_remaining == 400:
                print("we no longer have lots of time, so we are switching parameters to reduce the number of moves and turns when no target is seen")
        else:
            self.select_action.max_num_no_target_moves = MAX_NUM_NO_TARGET_MOVES_NEAR_TIMEOUT
            self.select_action.max_num_no_target_turns = MAX_NUM_NO_TARGET_TURNS_NEAR_TIMEOUT
            if self.time_remaining == MAX_NUM_STEPS_TO_REDUCE_TURNS:
                print("we are near timeout, so we are reducing the number of looking around turns")

        #use the reward value to determine if we contacted a target
        #the reward magnitude corresponds to the size of the target and
        #the sign is determined by whether it is a good or bad target
        #For finite length runs of length T, there is a -1/T reward at each step that does not contact a target

        hotzone = False
        if reward < -1.0 / self.timesteps and reward > -0.0475:
            #print("standing in a hot zone")
            hotzone = True

        self.episode_total_reward += reward
        self.total_reward += reward

        if abs(reward) >= 0.05:
            self.total_targets_found += 1
            self.target_found_step_numbers.append(self.steps)
            if reward >= 0.05:
                self.total_reward_count += 1.0
            elif reward <= -0.05:
                self.total_reward_count -= 1.0
            print(
                "target found in episode {0:2d}, episode_step = {1:2d}, steps remaining = {2:2d}, reward (target diameter) = {3:.1f}, episode reward = {4:.2f}, total_reward = {5:.2f}, reward count = {6:.2f}, total targets found = {7:2d}".format(
                    self.episode, self.episode_steps, self.time_remaining, reward, self.episode_total_reward, self.total_reward,
                    self.total_reward_count, self.total_targets_found))
            if USE_MAP:
                print("Current location: x = {0:.1f}, z = {1:.1f}, yaw degrees = {2:.1f}, action: ({3:.0f}, {4:.0f})".format(
                    self.map.x, self.map.z, self.map.yaw * 57.3, self.action[0], self.action[1]))
            self.rewards.append(self.total_reward_count)
            self.previous_reward_step = self.steps
            self.reward_received = True

        if self.reward_received and (not done):
            print("note that the above reward was received without resetting the run")
            self.reward_received = False
            if USE_MAP:
                #clear this target from the gold occupancy map and gold target list
                self.map.gold_target_captured()
            #if we got at least one gold target, we are more willing to take a green and exit
            #self.max_steps_to_accept_green +=  50
            if self.time_remaining > self.max_steps_to_accept_green:
                #if we are not short on time, look all around for another gold one
                print("got a gold reward, looking for another one, time remaining = {0:2d}".format(self.time_remaining))
            else:
                print("got a gold reward, but we only have {0:2d} steps left, so we are accepting green targets now".format(self.time_remaining))
                self.accept_green = True
            #after we get a gold reward, we want to look around again for a target
            self.select_action.initial_inspection_steps_completed = 0           
            self.select_action.green_found_on_initial_inspection_radius = 0.0
            self.select_action.green_found_on_initial_inspection = 0

        if done:
            #don't reset the episode number because we want to remember across configs
            self.episode += 1
            print("episode incremented")
            #if self.episode > NUM_EPISODES:
            #    self.episode = 1
            #if USE_MAP and SHOW_MAP:
            #   self.map.show_map(1000)
            if not self.reward_received:
                print("run timed out after {0:2d} steps".format(self.episode_steps))
                self.target_found_step_numbers.append(self.steps)
                self.previous_reward_step = self.steps
                self.num_runs_that_timed_out += 1
            if self.blackouts_happening:
                self.blackout_seen_in_previous_episode = True
            else:
                self.blackout_seen_in_previous_episode = False
            self.first_call = True
            self.action = [0.0, 2.0]
            print("episode done, last action is to turn left")
            #input("key to keep going")
            return self.action

        #analyze the observation image, look for green, gold, and red areas
        #image is 84x84 color, so dimentions are 84x84x3
        image = visual_observations[0][0]
        array = image * 255
        frame = array.astype(np.uint8)

        #look for targets in the observation image
        radius, size, centerX, centerY, hotzone_radius, hotzone_center, color, obstacle_in_view, wall_in_view, red_in_view, blackout = self.process_image.find_targets(frame, self.accept_green)
        if size < 0:
            size = 0
        #size = radius
        if hotzone_radius > MIN_RED_RADIUS and self.time_remaining > MAX_NUM_STEPS_TO_ENTER_HOTZONES:
            #print("hotzone seen, radius = ", hotzone_radius)
            radius = hotzone_radius
            centerX = hotzone_center[0]
            centerY = hotzone_center[1]
            red_in_view = True
            color = 4
            #input("see hotzone")
        #else:
        #   print("hotzone radius = ", hotzone_radius)
        '''
        if red_in_view:
            if not self.red_seen:
                self.episode_when_red_seen = self.episode
            self.red_seen = True
        
        if color == 1:
            if USE_MAP:
                self.map.add_green_seen()
                if self.accept_green and self.astar_actions:
                    print("canceling astar_actions because we see a green target and are accepting green targets")
                    self.astar_actions = False
        '''   
        if color == 2:
            self.gold_targets_present = True
            #if we see a gold target, we want to take it, not continue to explore
            self.astar_actions = False

        if blackout:
            if not self.blackouts_happening:
                print("blackouts are happening, so we will accept green targets")
            #if we are seeing a new blackout, remember it
            if (not self.blackout_seen) or self.episode >= self.episode_when_blackout_seen + 6:
                self.episode_when_blackout_seen = self.episode
            self.blackouts_happening = True
            self.accept_green = True

        
        if obstacle_in_view or self.select_action.headbang_obstacle: # or self.obstacle_seen:
            if not self.obstacle_seen:
                print("obstacles are present, so we will accept green targets")
                self.episode_when_obstacle_seen = self.episode
            if obstacle_in_view or self.select_action.headbang_obstacle:
                self.obstacle_seen_in_previous_episode = True
            self.obstacle_seen = True 
            self.accept_green = True
            self.max_steps_to_accept_green = 2000
            #input("self obstacle seen")
        
        if USE_MAZE_NETWORK and color == 0:
            merged_observations = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            if wall_in_view == 0:  #wall to the left
                merged_observations[0] = 1.0
            elif wall_in_view == 1:  #wall in front
                merged_observations[1] = 1.0
            elif wall_in_view == 2:  #wall on the right
                merged_observations[2] = 1.0
            if self.select_action.headbang > 0:
                merged_observations[3] = 1.0
            if self.select_action.stuck > 0:
                merged_observations[4] = 1.0
            merged_observations[5] = vector_observations[0] / MAX_VEL
            merged_observations[6] = vector_observations[2] / MAX_VEL

        else:
            merged_observations = np.array([radius / MAX_RADIUS,
                                        centerX / MAX_PIXEL,
                                        centerY / MAX_PIXEL,
                                        0., 0., 0.,
                                        vector_observations[0] / MAX_VEL,
                                        vector_observations[2] / MAX_VEL])

        #pull out the velocity observations: vector_observations[0] has the x,y,z velocity.  The other 3 arrays are all [0,0,0]
        #the Unity convention is that +X is to the right and +Z is forward.  We are in the agent reference frame
        velocity = math.sqrt((vector_observations[0] * vector_observations[0]) + (vector_observations[2] * vector_observations[2]))
        z_velocity = vector_observations[2]
        print("velocity = {0:.1f}, z_velocity = {1:.1f}".format(velocity, z_velocity))
        if self.astar_stuck > 0:
            #we got stuck during an astar path, so we are pausing use of astar for a bit
            if not self.select_action.stuck:
                self.astar_stuck -= 1
            else:
                self.astar_stuck = 30
            if self.astar_stuck == 0:
                print("astar completed its timeout from getting stuck")

        '''
        initial_network_action = self.explore.wall_follow(self.previous_action, velocity, wall_in_view, 10000.)
        initial_action = self.select_action.transform_network_action_to_model_action(initial_network_action)
        self.previous_action = initial_action
        print("initial network action = ", initial_network_action)
        input("see action")
        self.steps += 1
        self.episode_steps += 1
        return initial_action
        '''
               
        #pick the action
        if not self.astar_actions:
            target_list_length = 0
            if USE_MAP:
                for target in self.map.target_list:
                    # target list: x, z, diameter, color, number of times seen
                    if target[3] == 1:  # green target
                        target_list_length += 1
            self.agent_network_action = self.select_action.choose_action(self.action, merged_observations, radius, size, color, wall_in_view, self.process_image.floor_size, velocity, z_velocity,
                                                                         self.obstacle_seen, red_in_view, hotzone, self.no_more_unexplored_areas_exist, self.astar_stuck, self.astar_done, target_list_length, self.accept_green,
                                                                         self.episode_steps, self.time_remaining)
            #if self.astar_initial_stop == 1:
            #    self.agent_network_action = 98
            if self.agent_network_action != 99 and self.agent_network_action != 98:  # when it == 99, we are going to explore with maps, when it = 98 we are going to find a green target with astar
                agent_action = self.select_action.transform_network_action_to_model_action(self.agent_network_action)
                reward_only_used_in_training, self.action = self.select_action.modify_action(agent_action, self.previous_action, velocity, z_velocity, blackout, wall_in_view, red_in_view, color,
                                                                                             self.previous_color, self.obstacle_seen, hotzone, self.episode_steps, self.time_remaining)
                modified_network_action = self.select_action.transform_model_action_to_network_action(self.action)
                if modified_network_action != self.agent_network_action:
                    print("agent network action was modified from ", self.agent_network_action, " to ", modified_network_action)
            else:
                print("select action deferred to astar, self.agent_network_action = ", self.agent_network_action, ", color = ", color)
            
        if USE_MAP:
            self.map.update(vector_observations, self.action, self.process_image.floor_mask, blackout, wall_in_view)
            #if self.select_action.headbang:
            #    print("filling in obstacle after we got a headbang")
            #    self.map.add_obstacle(True, wall_in_view)
            #elif self.select_action.tailbang:
            #    print("filling in obstacle after we got a tailbang")
            #    self.map.add_obstacle(False, wall_in_view)
            if not blackout:
                if self.process_image.green_radius > 1e-05:
                    # print("image green center = ", self.process_image.green_center[0], " pixels from the left and ", self.process_image.green_center[1], " rows from the top. radius = ", self.process_image.green_radius)
                    if self.use_target_maps:
                        self.map.update_target_map(self.process_image.green_center[0], self.process_image.green_center[1], self.process_image.green_radius, 1)
                    else:
                        self.map.update_target_list(self.process_image.green_center[0], self.process_image.green_center[1], self.process_image.green_radius, 1)
                if self.process_image.gold_radius > 1e-05:
                    if self.use_target_maps:
                        self.map.update_target_map(self.process_image.gold_center[0], self.process_image.gold_center[1], self.process_image.gold_radius, 2)
                    else:
                        self.map.update_target_list(self.process_image.gold_center[0], self.process_image.gold_center[1], self.process_image.gold_radius, 2)
                if self.process_image.red_radius > 1e-05:
                    if self.use_target_maps:
                        self.map.update_target_map(self.process_image.red_center[0], self.process_image.red_center[1], self.process_image.red_radius, 3)
                    else:
                        self.map.update_target_list(self.process_image.red_center[0], self.process_image.red_center[1], self.process_image.red_radius, 3)
                    
            #check to see if we are moving using an astar path
            if self.astar_actions:
                if self.astar_num_turns > 0:
                    print("astar turning left, number of astar turns remaining = ", self.astar_num_turns)
                    self.action = [0, 2]  # left
                    self.astar_num_turns -= 1
                elif self.astar_num_turns < 0:
                    print("astar turning right, number of astar turns remaining = ", -self.astar_num_turns)
                    self.action = [0, 1]  # right
                    self.astar_num_turns += 1
                else:
                    if self.astar_num_moves > 0:
                        print("astar moving forward, number of astar moves remaining = ", self.astar_num_moves)
                        self.action = [1, 0]  # forward
                        self.astar_num_moves -= 1
                    elif self.astar_num_moves == 0:
                        if self.astar_num_back > 0:
                            print("astar moving backward, number of astar backward moves remaining = ", self.astar_num_back)
                            self.action = [2, 0]  # backward to help come to a stop
                            self.astar_num_back -= 1
                        if self.astar_num_back == 0:  # do not want elif here because we can go ahead and take the last back move below
                            if len(self.astar_path) == 0:
                                print("astar move complete")
                                #input("look at whether we arrived at new unexplored location")
                                self.astar_actions = False
                                
                                
                                self.astar_done = True #we will only do one astar search no matter what
                                
                                
                                #if we went to a green target, that is the last astar search we do
                                if self.target_list_green_goal_found:
                                    print("green goal path was found by astar")
                                    #self.astar_done = True
                                else:
                                    print("green goal path was not found by astar")
                                self.select_action.initial_inspection_steps_completed = 0
                                self.select_action.green_found_on_initial_inspection_radius = 0.0
                                self.select_action.green_found_on_initial_inspection = 0
                                
                            else:
                                # get the turns and moves for the next line segment
                                self.astar_num_turns, self.astar_num_moves, self.astar_num_back, self.astar_path = self.map.follow_path(self.astar_path)
                                if self.agent_network_action == 99:
                                    print("since we are exploring new areas, we want to headbang into walls and obstacles to mark them, so we set self.astar_num_back = 0")
                                    self.astar_num_back = 0
                if self.select_action.check_for_headbang(self.action, velocity, self.previous_velocity, wall_in_view, blackout, self.episode_steps):
                    #print("we got a headbang or tailbang while doing astar actions")
                    if self.select_action.headbang and USE_OCCUPANCY_MAP:
                        #print("headbang = ", self.select_action.headbang, ", tailbang = ", self.select_action.tailbang)
                        print("we got a headbang while doing astar actions, adding obstacle to map")
                        self.map.add_obstacle(True, wall_in_view)
                    if self.select_action.tailbang and USE_OCCUPANCY_MAP:
                        print("we got a tailbang while doing astar actions, adding obstacle to map")
                        self.map.add_obstacle(True, wall_in_view)
                    self.astar_actions = False
                    if self.num_failed_astar_moves > 10:
                        self.no_more_unexplored_areas_exist = True
                    self.num_failed_astar_moves += 1
                    if self.astar_check_in_new_direction:
                        self.astar_check_in_new_direction = False
                    else:
                        self.astar_check_in_new_direction = True
                if self.select_action.check_for_stuck(velocity, self.previous_velocity, self.action):
                    print("we got stuck while doing astar actions: velocity, self.previous_velocity, stuck_number = ", velocity, self.previous_velocity, self.select_action.stuck_number)
                    self.astar_actions = False
                    self.astar_stuck = 30
                    if self.num_failed_astar_moves > 10:
                        self.no_more_unexplored_areas_exist = True
                    self.num_failed_astar_moves += 1
                    if self.astar_check_in_new_direction:
                        self.astar_check_in_new_direction = False
                    else:
                        self.astar_check_in_new_direction = True
                #else:
                #    print("we are not stuck while doing astar actions: velocity, self.previous_velocity, stuck_number = ", velocity, self.previous_velocity, self.select_action.stuck_number)

                #input("look at action") 
                print("astar action = ", self.action)
                #input("see astar path")
            #check to see if we should find a path to a green_seen location    
            elif self.agent_network_action == 98:
                #stop first, then calculate the path
                if self.astar_initial_stop == 0:
                    #this gets us to skip calculating network actions, setting it to = 98
                    self.astar_initial_stop = 1
                if self.astar_initial_stop == 1:
                    print("slowing down in preparation for astar, current z_velocity = {0:.1f}".format(z_velocity))
                    if z_velocity > 3.: 
                        self.action = [2, 0]
                        return
                    elif z_velocity < -3.:
                        self.action = [1, 0]
                        return
                    self.astar_initial_stop = 2
                        
                # get a green target with astar
                if not blackout:
                    self.astar_path = self.map.find_astar_path_to_green_seen_location()
                if blackout or (self.astar_path is None):
                    #try finding a path directly to a green target               
                    self.target_list_green_goal_found = False
                    if len(self.map.target_list) > 0:
                        max_times_seen = 0
                        #pick the green target that was seen the most number of times
                        for target in self.map.target_list:
                            # target list: x, z, diameter, color, number of times seen
                            if target[3] == 1:  # green target
                                if target[4] > max_times_seen:
                                    goal = (target[0], target[1])
                                    max_times_seen = target[4]
                                    self.target_list_green_goal_found = True
                                    self.astar_done = True
                        if self.target_list_green_goal_found:
                            self.astar_path = self.map.astar((self.map.x, self.map.z), (goal[0], goal[1]), self.map.occupancy_map)
                            if self.astar_path is None:
                                self.target_list_green_goal_found = False
            
                if self.astar_path is not None:
                    self.astar_actions = True
                    if len(self.astar_path) > 0:
                        if blackout:
                            print("astar path found from current location to green target location around ", self.astar_path[len(self.astar_path) - 1])
                        else:    
                            print("astar path found from current location to green_seen location around ", self.astar_path[len(self.astar_path) - 1])
                    #print("path: ", self.astar_path)
                    self.astar_num_turns, self.astar_num_moves, self.astar_num_back, self.astar_path = self.map.follow_path(self.astar_path)
                    print("num_astar_turns = ", self.astar_num_turns, ", num_astar_moves = ", self.astar_num_moves)
                else:
                    print("no path to a green_seen or a green target was found")
                    self.astar_done = True
                    self.astar_actions = False
                self.action = [0, 0]
                #input("see astar path")
            #check to see if we should find a new unexplored area
            elif self.agent_network_action == 99:
                #find a new unexplored area
                self.astar_actions = True
                #start = self.map.get_current_location()
                self.astar_path = self.map.find_unexplored_area_near_current_location(self.astar_check_in_new_direction)
                if self.astar_path is not None:
                    print("astar path found from current location to unexplored region around ", self.astar_path[-1])
                    #print("path: ", self.astar_path)
                    self.astar_num_turns, self.astar_num_moves, self.astar_num_back, self.astar_path = self.map.follow_path(self.astar_path)
                    print("num_astar_turns = ", self.astar_num_turns, ", num_astar_moves = ", self.astar_num_moves)
                    print("since we are exploring new areas, we want to headbang into walls and obstacles to mark them, so we set self.astar_back = 0")
                    self.astar_num_back = 0
                else:
                    print("no path to an open area was found")
                    self.no_more_unexplored_areas_exist = True
                    self.astar_actions = False
                self.action = [0, 0]
                #input("look at new unexplored location")
                
        if not blackout:
            self.previous_color = color  # only update this when we are not in blackout
            
        #if we have done 300 steps without seeing a green or gold target, change the direction of the exploration loop
        '''
        if self.steps == 300:
            if USE_MAP:
                if len(self.map.target_list) == 0:
                    self.select_action.loop_direction = -self.select_action.loop_direction
                else:
                    found_at_least_one_target = False
                    for target in self.map.target_list:
                        if target[3] == 1 or target[3] == 2:
                            #we have seen at least one green or one gold target
                            found_at_least_one_target = True
                            break
                    if not found_at_least_one_target:
                        self.select_action.loop_direction = -self.select_action.loop_direction
        '''           
        self.previous_velocity = velocity
        self.previous_action = self.action
        self.actions_taken_frequency[self.select_action.transform_model_action_to_network_action(self.action)] += 1
        self.steps += 1
        self.episode_steps += 1
        if self.agent_network_action == 99 or self.agent_network_action == 98 or self.astar_actions:
            print("astar action = ", self.action, ", color = ", color, ", radius = ", radius, ", centerX = ", centerX, ", centerY = ", centerY)
        else:
            print("self.agent_network_action = ", self.agent_network_action, ", action = ", self.action, ", color = ", color, ", radius = ", radius, ", size = ", size, ", loop direction = ", self.select_action.loop_direction)
        if wall_in_view >= 0:
            if wall_in_view == 0:
                print("wall in view on the left, wall_in_view = ", wall_in_view)
            elif wall_in_view == 1:
                print("wall in view in front, wall_in_view = ", wall_in_view)
            elif wall_in_view == 2:
                print("wall in view on the right, wall_in_view = ", wall_in_view)

        if self.episode == 100000 and (not DOCKER): # or self.select_action.stuck: # or self.agent_network_action == 99 or self.agent_network_action == 98 or self.select_action.stuck:
            try:
                value = input("hit any key to continue")
            except ValueError:
                print(" ")
            except EOFError:
                print(" ")
        
        return self.action

class Agent(object):
    def __init__(self):
        self.resolution = MAX_PIXEL
        self.gr = GameRunner()
        self.num_timesteps = 250

    def reset(self, t=250):
        if t == 0:
            t = 10000
        self.num_timesteps = t
        print("number of timesteps for this run = {0:2d}".format(t))
        self.gr.first_call = True

    def step(self, obs, reward, done, info):
        brainInfo = info['brain_info']
        visual_obs = brainInfo.visual_observations
        vector_obs = brainInfo.vector_observations[0]
        reward = brainInfo.rewards[0]
        done = brainInfo.local_done[0]
        return self.gr.run(visual_obs, vector_obs, reward, done, self.num_timesteps)

if __name__ == "__main__":
    env = init_environment()

    if WATCH_IMAGE:
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        #cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)

    gr = GameRunner()
    num_episodes = NUM_EPISODES
    timesteps = TIMESTEPS
    total_time = datetime.now()
    config_time = datetime.now()
    episode_time = datetime.now()
    config_t = []
    if MULTIPLE_CONFIGS:
        configs = []
        configs.append('1-Food')
        configs.append('2-Preferences')
        configs.append('3-Obstacles')
        configs.append('4-Avoidance')
        configs.append('5-SpatialReasoning')
        configs.append('6-Generalization')
        configs.append('7-InternalMemory')
        configs.append('temporary_blackout')
        configs.append('permanent_blackout')
        configs.append('permanent_blackout_with_wall_and_bad_goal')
        configs.append('hot_zone')
        configs.append('movingFood')
        configs.append('forcedChoice')
        configs.append('objectManipulation')
        configs.append('allObjectsRandom')

        total_reward = []
        total_reward_count = []
        num_runs_that_timed_out = []
        average_number_of_steps_until_reward = []
        cumulative_total_reward = 0.0
        cumulative_num_runs_that_timed_out = 0
        cumulative_total_reward_count = 0
        cumulative_average_number_of_steps_until_reward = 0
        for config in configs:
            print("\n\nstarting a new config: ", config)
            arena_config_in = ArenaConfig("configs/" + config + ".yaml")
            env.reset(arenas_configurations=arena_config_in, train_mode=True)
            gr.set_values()
            action = [0, 0]           
            steps = 0
            cnt = 1
            episode_t = []
            while cnt <= num_episodes:
                steps += 1
                info = env.step(action)
                brainInfo = info['Learner']
                visual_obs = brainInfo.visual_observations
                vector_obs = brainInfo.vector_observations[0]
                reward = brainInfo.rewards[0]
                done = brainInfo.local_done[0]

                if steps >= timesteps:
                    done = True

                action = gr.run(visual_obs, vector_obs, reward, done, timesteps)
                timesteps = gr.timesteps

                if done:
                    print("\ncompleted config is {0:s}, episode number = {1:2d} of {2:2d}, time taken = {3:.1f}".format(config, cnt, num_episodes, (datetime.now() - episode_time).total_seconds()))
                    cnt += 1
                    #if cnt % 10 == 0:
                    #   print("\n\nnew config is: ", config)
                    #   print('..................................................................Episode {} of {}'.format(cnt, num_episodes))
                    deltatime = datetime.now() - episode_time
                    episode_t.append(deltatime.total_seconds())
                    episode_time = datetime.now()
                    steps = 0
                    env.reset(arenas_configurations=arena_config_in, train_mode=True)

            print("total reward for all runs = {0:.2f}, average reward per run = {1:.2f}, total reward count = {2:.2f}".format(gr.total_reward, gr.total_reward / num_episodes, gr.total_reward_count))
            print("number of runs with positive reward = {0:2d}, number of runs that timed out = {1:2d}".format(num_episodes - gr.num_runs_that_timed_out, gr.num_runs_that_timed_out))
            avg = 0.0
            sum = 0.0
            num_points = len(gr.target_found_step_numbers)
            if num_points >= 2:
                for i in range(1, num_points):
                    sum += gr.target_found_step_numbers[i] - gr.target_found_step_numbers[i - 1]
                avg = sum / float(num_points)
                print("average number of steps until reward = {0:.2f}".format(sum / float(num_points)))

            total_reward.append(gr.total_reward)
            num_runs_that_timed_out.append(gr.num_runs_that_timed_out)
            total_reward_count.append(gr.total_reward_count)
            average_number_of_steps_until_reward.append(avg)

            cumulative_total_reward += gr.total_reward
            cumulative_num_runs_that_timed_out += gr.num_runs_that_timed_out
            cumulative_total_reward_count += gr.total_reward_count
            cumulative_average_number_of_steps_until_reward += avg
            deltatime = datetime.now() - config_time
            config_t.append(deltatime.total_seconds())
            print("time taken for this config = {0:.1f}, average time per episode = {1:.1f}".format(deltatime.total_seconds(), deltatime.total_seconds() / num_episodes))
            config_time = datetime.now()
        save_data_path = 'data/multiple_configs/'
        deltatime = datetime.now() - total_time
        print("total time taken = {0:.1f}, average time per config = {1:.1f}, avg time per episode = {2:.1f} ".format(deltatime.total_seconds(),
              deltatime.total_seconds() / len(config_t), deltatime.total_seconds() / (len(config_t) * num_episodes)))
        with open(save_data_path + MULTIPLE_CONFIGS_SAVE_FILENAME + '.txt', 'w') as f:
            print("\nnumber of episodes = {0:2d}".format(NUM_EPISODES))
            f.write("number of episodes = {0:2d}".format(NUM_EPISODES))
            parameters_string = "total_reward, total_reward_count, num_runs_that_timed_out, average_number_of_steps_until_reward, time taken:"
            print(parameters_string)
            f.write("\n" + parameters_string)
            for i in range(len(configs)):
                parameters_string = "{0:6.2f}, {1:3.0f}, {2:2.0f}, {3:3.0f}, {4:.1f}, {5:s}".format(total_reward[i], total_reward_count[i], num_runs_that_timed_out[i], average_number_of_steps_until_reward[i], config_t[i], configs[i])
                print(parameters_string)
                f.write("\n" + parameters_string)

            cumulative_total_reward /= len(configs)
            cumulative_num_runs_that_timed_out /= len(configs)
            cumulative_total_reward_count /= len(configs)
            cumulative_average_number_of_steps_until_reward /= len(configs)
            average_time = 0
            for value in config_t:
                average_time += value
            average_time /= len(configs)
            # print("\noverall average total_reward, overall average total_reward_count, overall average num_runs_that_timed_out, overall average, average_number_of_steps_until_reward: {0:0.2f}, {1:.1f}, {2:.2f}, {3:0.2f}".format(
            #    cumulative_total_reward, cumulative_total_reward_count, cumulative_num_runs_that_timed_out, cumulative_average_number_of_steps_until_reward))

            parameters_string = "{0:6.2f}, {1:3.0f}, {2:2.0f}, {3:3.0f}, {4:0.1f}, overall averages".format(cumulative_total_reward, cumulative_total_reward_count, cumulative_num_runs_that_timed_out, cumulative_average_number_of_steps_until_reward, average_time)
            print(parameters_string)
            f.write("\n" + parameters_string)

    elif PARAMETER_RANGE:
        min_parameter = 1.0
        max_parameter = 30.0
        delta_parameter = 2.0

        parameters = []
        total_reward = []
        total_reward_count = []
        num_runs_that_timed_out = []
        average_number_of_steps_until_reward = []
        cumulative_total_reward = 0.0
        cumulative_num_runs_that_timed_out = 0
        cumulative_total_reward_count = 0
        cumulative_average_number_of_steps_until_reward = 0



        parameter_name = "min_red_radius"



        for parameter in np.arange(min_parameter, max_parameter + delta_parameter, delta_parameter):
            parameters.append(parameter)



            gr.select_action.min_red_radius = parameter
            gr.process_image.min_red_radius = parameter



            print("\nparameter value = : ", parameter)
            env.reset(arenas_configurations=arena_config_in, train_mode=True)
            gr.set_values()
            action = [0, 0]
            cnt = 1
            steps = 0
            while cnt <= num_episodes:
                steps += 1
                info = env.step(action)
                brainInfo = info['Learner']
                visual_obs = brainInfo.visual_observations
                vector_obs = brainInfo.vector_observations[0]
                reward = brainInfo.rewards[0]
                done = brainInfo.local_done[0]

                if steps >= timesteps:
                    done = True

                action = gr.run(visual_obs, vector_obs, reward, done, timesteps)
                timesteps = gr.timesteps

                if done:
                    cnt += 1
                    if cnt % 10 == 0:
                        print('..................................................................Episode {} of {}'.format(cnt, num_episodes))
                    steps = 0
                    env.reset(arenas_configurations=arena_config_in, train_mode=True)

            print("total reward for all runs = {0:.2f}, average reward per run = {1:.2f}, total reward count = {2:.2f}".format(gr.total_reward, gr.total_reward / num_episodes, gr.total_reward_count))
            print("number of runs with positive reward = {0:2d}, number of runs that timed out = {1:2d}".format(num_episodes - gr.num_runs_that_timed_out, gr.num_runs_that_timed_out))
            avg = 0.0
            sum = 0.0
            num_points = len(gr.target_found_step_numbers)
            if num_points >= 2:
                for i in range(1, num_points):
                    sum += gr.target_found_step_numbers[i] - gr.target_found_step_numbers[i - 1]
                avg = sum / float(num_points)
                print("average number of steps until reward = {0:.2f}".format(sum / float(num_points)))

            total_reward.append(gr.total_reward)
            num_runs_that_timed_out.append(gr.num_runs_that_timed_out)
            total_reward_count.append(gr.total_reward_count)
            average_number_of_steps_until_reward.append(avg)

            cumulative_total_reward += gr.total_reward
            cumulative_num_runs_that_timed_out += gr.num_runs_that_timed_out
            cumulative_total_reward_count += gr.total_reward_count
            cumulative_average_number_of_steps_until_reward += avg

        save_data_path = 'data/parameter_ranges/'
        with open(save_data_path + parameter_name + '_' + configuration_yaml_file + '.txt', 'w') as f:
            parameters_string = "parameter being varied is: {0:s}, range goes from {1:.2f} to {2:.2f}, number of episodes = {3:2d}, config: {4:s}".format(
                parameter_name, min_parameter, parameters[-1], NUM_EPISODES, configuration_yaml_file)
            print("\n" + parameters_string)
            f.write(parameters_string)
            parameters_string = "total_reward, total_reward_count, num_runs_that_timed_out, average_number_of_steps_until_reward, " + parameter_name
            print(parameters_string)
            f.write("\n" + parameters_string)
            for i in range(len(parameters)):
                #parameters_string = "\nconfig: {0:s}, total_reward = {1:0.2f}, total_reward_count = {2:.1f}, num_runs_that_timed_out = {3:.2f}, average_number_of_steps_until_reward = {4:0.2f}".format(
                #configs[i], total_reward[i], total_reward_count[i], num_runs_that_timed_out[i], average_number_of_steps_until_reward[i])
                #parameters_string = "\nconfig, total_reward, total_reward_count, num_runs_that_timed_out, average_number_of_steps_until_reward: {0:s}, {1:0.2f}, {2:.1f}, {3:.2f}, {4:0.2f}".format(
                #   configs[i], total_reward[i], total_reward_count[i], num_runs_that_timed_out[i],
                #   average_number_of_steps_until_reward[i])
                parameters_string = "{0:6.2f}, {1:3.0f}, {2:2.0f}, {3:3.0f}, {4:3.2f}".format(
                    total_reward[i], total_reward_count[i], num_runs_that_timed_out[i], average_number_of_steps_until_reward[i], parameters[i])
                print(parameters_string)
                f.write("\n" + parameters_string)

            cumulative_total_reward /= len(parameters)
            cumulative_num_runs_that_timed_out /= len(parameters)
            cumulative_total_reward_count /= len(parameters)
            cumulative_average_number_of_steps_until_reward /= len(parameters)
            average_parameter = (max_parameter + min_parameter) / 2.0

            parameters_string = "{0:6.2f}, {1:3.0f}, {2:2.0f}, {3:3.0f}, {4:3.2f}, overall averages".format(
                cumulative_total_reward, cumulative_total_reward_count, cumulative_num_runs_that_timed_out, cumulative_average_number_of_steps_until_reward, average_parameter)
            print(parameters_string)
            #parameters_string = "\n{0:0.2f}, {1:.1f}, {2:.2f}, {3:0.2f}, {4:.2f}, overall averages".format(
            #   cumulative_total_reward, cumulative_total_reward_count, cumulative_num_runs_that_timed_out, cumulative_average_number_of_steps_until_reward, average_parameter)
            f.write("\n" + parameters_string)

        plt.scatter(parameters, average_number_of_steps_until_reward)
        plt.show()

    else:
        env.reset(arenas_configurations=arena_config_in, train_mode=True)
        action = [0, 0]
        cnt = 1
        steps = 0
        total_time = datetime.now()
        episode_time = datetime.now()
        episode_t = []
        while cnt <= num_episodes:
            steps += 1
            info = env.step(action)
            brainInfo = info['Learner']
            visual_obs = brainInfo.visual_observations
            vector_obs = brainInfo.vector_observations[0]
            reward = brainInfo.rewards[0]
            done = brainInfo.local_done[0]

            if steps >= timesteps:
                done = True
                '''
                try:
                    value = input("observe timeout, hit any key to continue")
                except ValueError:
                    print(" ")
                except EOFError:
                    print(" ")
                '''
            action = gr.run(visual_obs, vector_obs, reward, done, timesteps)
            timesteps = gr.timesteps

            if done:
                cnt += 1
                if cnt % 10 == 0:
                    print('..................................................................Episode {} of {}'.format(cnt, num_episodes))
                    
                steps = 0
                deltatime = datetime.now() - episode_time
                episode_t.append(deltatime.total_seconds())
                print("time taken for this episode = {0:.1f}".format(deltatime.total_seconds()))
                episode_time = datetime.now()
                env.reset(arenas_configurations=arena_config_in, train_mode=True)

        print("\ntotal reward for all runs = {0:.2f}, average reward per run = {1:.2f}, total reward count = {2:.2f}".format(
            gr.total_reward, gr.total_reward / num_episodes, gr.total_reward_count))
        print("number of runs with positive reward = {0:2d}, number of runs that timed out = {1:2d}".format(
            num_episodes - gr.num_runs_that_timed_out, gr.num_runs_that_timed_out))
        sum = 0
        for value in episode_t:
            sum += value
        print("average time taken by episodes = {0:.1f}: ".format(sum / num_episodes))

        num_points = len(gr.target_found_step_numbers)
        if num_points >= 2:
            sum = 0.0
            for i in range(1, num_points):
                sum += gr.target_found_step_numbers[i] - gr.target_found_step_numbers[i - 1]
            avg = sum / float(num_points)
            print("average number of steps until reward = {0:.2f}".format(sum / float(num_points)))

        plt.plot(gr.target_found_step_numbers)
        plt.show()

        plt.plot(gr.rewards)
        plt.show()

        forward_moves = 0
        neutral_moves = 0
        backward_moves = 0
        for i in range(9):
            if i == 1 or i == 2 or i == 3:
                forward_moves += gr.actions_taken_frequency[i]
            elif i == 0 or i == 4 or i == 8:
                neutral_moves += gr.actions_taken_frequency[i]
            else:
                backward_moves += gr.actions_taken_frequency[i]
        print("number of forward moves = {0:2d}, neutral moves = {1:2d}, backward moves = {2:2d}".format(forward_moves,
                                                                                                         neutral_moves,
                                                                                                         backward_moves))
        x = np.arange(9)
        plt.bar(x, gr.actions_taken_frequency)
        plt.show()
    env.close()
