import numpy as np
import random

class SelectAction:
    def __init__(self, graph_green, using_old_green_network, graph_red, using_old_red_network, graph_maze, number_of_initial_inspection_steps_to_use,
                 epsilon, max_oscillations, max_stuck_number, use_maze_network, max_num_no_target_moves, max_num_no_target_turns,
                 min_red_radius, min_hotzone_size, min_green_radius, max_pixel, delta_timestep, training):
        #self.explore = Explore(delta_timestep)
        self.graph_green = graph_green
        self.using_old_green_network = using_old_green_network
        self.using_old_red_network = using_old_red_network
        self.graph_red = graph_red
        self.graph_maze = graph_maze
        self.number_of_initial_inspection_steps_to_use = number_of_initial_inspection_steps_to_use
        self.epsilon = epsilon
        self.max_oscillations = max_oscillations
        self.max_stuck = max_stuck_number
        self.use_maze_network = use_maze_network
        self.max_num_no_target_moves = max_num_no_target_moves
        self.max_num_no_target_turns = max_num_no_target_turns
        self.min_green_radius = min_green_radius
        self.min_red_radius = min_red_radius
        self.min_hotzone_size = min_hotzone_size
        self.max_pixel = max_pixel
        self.training = training
        self.initialize_values()

    def initialize_values(self):
        self.x = self.z = self.yaw = 0.0
        #self.explore.first_bang = True
        #self.explore.number_of_corners_found = 0
        self.previous_action = [0., 0.]
        self.previous_velocity = 0.0
        self.previous_z_velocity = 0.0
        self.oscillation_number = 0
        self.stuck_number = 0
        self.previous_linear_actions = []
        self.previous_velocities = []
        self.stuck = False
        self.loop_direction = 1
        self.headbang = False
        self.headbang_number = 0
        self.tailbang = False
        self.tailbang_number = 0
        self.number_of_tailbangs = 0
        self.headbang_obstacle = False
        self.extra_obstacle_turns = 0
        self.turning_to_face_new_direction = 0
        self.initial_inspection_steps_completed = 0
        #self.number_of_look_arounds = 0
        self.green_found_on_initial_inspection = 0
        self.initial_inspection_steps_completed_when_green_target_seen = 0
        self.steps_to_go = 0
        self.green_found_on_initial_inspection_radius = 0.0
        self.num_no_target_moves = 0
        self.num_no_target_turns = 0
        self.wall_moves = 0
        self.wall_turns = 0
        self.blackout = False
        self.temporary_blackouts_happening = False
        self.permanent_blackouts_happening = False
        self.blackout_start_steps = 0
        self.permanent_blackout = False
        self.wandering = 0
        self.wandering_count = 0
        self.wandering_moves = 0
        self.wandering_turns = 0
        self.wandering_cycles = 0
        self.astar_wandering = False
        self.stopped_for_blackout = False
        self.turning_toward_previous_target_during_blackout = False
        self.astar_done = False
        self.astar_finished = False
        self.red_color_seen = False
        self.previous_move_was_headbang_recovery = False
        self.previous_move_was_tailbang_recovery = False

    def transform_model_action_to_network_action(self, model_action):
        #the model takes actions with 2 entries per arena.  each entry can take 3 values ((nothing, forward, backward) and (nothing, right, left)
        #the values are floats:  0.0 for nothing, 1.0 for forward or right, 2.0 for backward or left
        #each increment of turning is 6 degrees
        #4 arenas
        #action = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #1 arena
        #action = [1.0, 2.0]
        #However, for the neural net, we need an entry for each distict action, so
        #network actions = [left, forward left, forward, forward right, right, backward right, backward, backward left, none]
        #model actions:  model_action[0]: 0.0 = none, 1.0 = forward, 2.0 = backward; model_action[1]: 0.0 = none, 1.0 = right, 2.0 = left
        network_action = 8  #none
        if model_action[0] == 0:
            if model_action[1] == 0:
                network_action = 8  #none
                #print("received a model action of [0,0] to translate to a network action, returning 8")
            elif model_action[1] == 1:
                network_action = 4  #right
            elif model_action[1] == 2:
                network_action = 0  #left
        elif model_action[0] == 1:
            if model_action[1] == 0:
                network_action = 2  #forward
            elif model_action[1] == 1:
                network_action = 3  #forward right
            elif model_action[1] == 2:
                network_action = 1  #forward left
        elif model_action[0] == 2:
            if model_action[1] == 0:
                network_action = 6  #backward
            elif model_action[1] == 1:
                network_action = 5  #backward right
            elif model_action[1] == 2:
                network_action = 7  #backward left
        else:
            print("received an unknown model action of ", model_action, " to translate to network action, returning 8")
        return network_action

    def transform_network_action_to_model_action(self, network_action):
        model_action = [0.0, 1.0]
        if network_action == 0:
            model_action = [0.0, 2.0]  #left
        elif network_action == 1:
            model_action = [1.0, 2.0]  #forward left
        elif network_action == 2:
            model_action = [1.0, 0.0]  #forward
        elif network_action == 3:
            model_action = [1.0, 1.0]  #forward right
        elif network_action == 4:
            model_action = [0.0, 1.0]  #right
        elif network_action == 5:
            model_action = [2.0, 1.0]  #backward right
        elif network_action == 6:
            model_action = [2.0, 0.0]  #backward
        elif network_action == 7:
            model_action = [2.0, 2.0]  #backward left
        elif network_action == 8:
            model_action = [0.0, 0.0]  #none
        else:
            print("received an unknown network action = ", network_action, " to translate to action, returning [0.0, 1.0] (right turn)")
        return model_action

    def transform_old_network_action_to_new_network_action(self, old_network_action):
        if old_network_action == 0:
            new_network_action = 8  #none
        elif old_network_action == 1:
            new_network_action = 2  #forward
        elif old_network_action == 2:
            new_network_action = 1  #forward left
        elif old_network_action == 3:
            new_network_action = 0  #left
        elif old_network_action == 4:
            new_network_action = 7  #backward left
        elif old_network_action == 5:
            new_network_action = 6  #backward
        elif old_network_action == 6:
            new_network_action = 5  #backward right
        elif old_network_action == 7:
            new_network_action = 4  #right
        elif old_network_action == 8:
            new_network_action = 3  #forward right
        else:
            print("received an unknown old network action = ", old_network_action, " to translate to new network action, returning 4 (right turn)")
            new_network_action = 4
        return new_network_action

    def choose_action(self, previous_action, state, radius, size, color, wall_in_view, floor_size, velocity, z_velocity, obstacle_seen, red_in_view, hotzone, no_more_unexplored_areas_exist, astar_stuck, astar_done,
                      target_list_length, accept_green, episode_step, time_remaining):
        # first we look all around to find targets
        # print("self.initial_inspection_steps_completed = ", self.initial_inspection_steps_completed, "accept_green = ", accept_green)
        # print(color, episode_step, self.initial_inspection_steps_completed)
        # print(self.stopped_for_blackout, self.temporary_blackouts_happening, self.green_found_on_initial_inspection_radius )
        self.astar_done = astar_done
        #print("astar_done = ", self.astar_done)
        
        if self.wandering > 0 and color == 0:
            # we are in permanent blackout so we want to use the map and astar
            # try to find a green target
            if accept_green and (not self.astar_done) and astar_stuck == 0 and target_list_length > 0:
                return 98
            else:
                #this value does not matter-- wandering action is done in modified_actions
                return 2
            # wander around the neighborhood of the green target for a while
            # then go searching for new areas to explore
            #if (not no_more_unexplored_areas_exist) and self.wandering_count > 75 and astar_stuck == 0:
            #    return 99
            # we are in permanent blackout and modify_action will determine our action, so it does not matter what we return here
        
        if red_in_view:
            if not self.red_color_seen:
                print("red target seen, there are bad goals in this run")
            self.red_color_seen = True

        if wall_in_view > 0 and color == 0 and (not obstacle_seen) and (not self.red_color_seen) and self.green_found_on_initial_inspection_radius < 0.01:
            if wall_in_view != 1:  #if it is in front, we will not take an action, since we do not know which way to turn
                print("starting wall action sequence")
                self.wall_moves = 10
                #we will do a full look around when we get away from the wall
                self.initial_inspection_steps_completed = 0
            #we are next to a wall, so, if it is on the right or left, do fixed pattern to move away
            if wall_in_view == 0:
                #wall is on the left
                #if self.red_color_seen:
                self.wall_turns = -20  #turn right 120 degrees
                #print("wall is on the left, moving forward, so we will be turning right")
                #else:
                #   self.wall_turns = 5 #we are moving backwards, turn 30 degrees left
                #   print("wall is on the left, moving backwards, so we will be turning left")
            elif wall_in_view == 2:
                #if self.red_color_seen:
                self.wall_turns = 15  #turn left 90 degrees
                #print("wall is on the right, moving forwards, so we will be turning left")
                #else:
                #   self.wall_turns = -5 #we are moving backwards, turn 30 degrees right
                #   print("wall is on the right, moving backwards, so we will be turning right")

        if self.blackout:
            if self.green_found_on_initial_inspection_radius >= 0.01:
                #we saw a green target while looking around, but it was not in the last image before blackout.
                #we can turn to it during blackout
                if self.steps_to_go == 0:
                    print("turning during blackout to face a previously seen green target")
                    #if self.green_found_on_initial_inspection > 0:
                    #   green_target_location =  self.green_found_on_initial_inspection + self.number_of_initial_inspection_steps_to_use - 60
                    #else:
                    #   green_target_location = self.green_found_on_initial_inspection + self.number_of_initial_inspection_steps_to_use
                    #print("self.initial_inspection_steps_completed_when_green_target_seen = ", self.initial_inspection_steps_completed_when_green_target_seen, ", green_target_location = ", green_target_location)
                    left_steps_to_go = self.initial_inspection_steps_completed_when_green_target_seen + (60 - self.initial_inspection_steps_completed)
                    right_steps_to_go = self.initial_inspection_steps_completed - self.initial_inspection_steps_completed_when_green_target_seen
                    #note that when this is done, if we are off a little and no target is seen
                    #then we will do a full look-around.  Since that uses left turns, it is ok to be
                    #a bit too far to the right.  But we do not want to be a bit too far to the left
                    #or we will have to turn nearly 360 degrees to see the target
                    #For that reason, we add 6 degrees to the steps if we are turning to the right
                    #we should probably subtract 12 degrees to the steps if we are turning left, but
                    #will wait to experiment on that.
                    if right_steps_to_go <= left_steps_to_go:
                        self.steps_to_go = -(right_steps_to_go + 1)  #adding 6 degrees gets the target more centered
                    else:
                        self.steps_to_go = left_steps_to_go
                    print("original green found on initial inspection = {0:2d}, updated = {1:2d}".format(self.green_found_on_initial_inspection, self.steps_to_go))
                    self.green_found_on_initial_inspection = self.steps_to_go
                    if self.initial_inspection_steps_completed < self.number_of_initial_inspection_steps_to_use:
                        self.initial_inspection_steps_completed = self.number_of_initial_inspection_steps_to_use
                self.turning_toward_previous_target_during_blackout = True
            else:
                self.turning_toward_previous_target_during_blackout = False
                #return self.transform_model_action_to_network_action(previous_action)
        else:
            self.turning_toward_previous_target_during_blackout = False

        if self.wall_moves > 0 and (not self.turning_toward_previous_target_during_blackout) and (not self.red_color_seen):
            #input("see wall moves")
            if color != 2 and (not (color == 1 and accept_green)) \
                    and (not (color == 3 and radius >= self.min_red_radius)) and (not (color == 4 and radius >= self.min_hotzone_size)):
                #print("self.wall_turns = ", self.wall_turns)
                if self.wall_turns < 0:
                    self.wall_turns += 1
                    #print("turning right to get away from the wall")
                    return 4  #turn right
                elif self.wall_turns > 0:
                    self.wall_turns -= 1
                    #print("turning left to get away from the wall")
                    return 0  #turn left
                self.wall_moves -= 1
                print("moving forwards to get away from the wall")
                return 2  #forward
            elif color == 2 or (color == 1 and accept_green) \
                    or (color == 3 and radius >= self.min_red_radius) or (color == 4 and radius >= self.min_hotzone_size):
                print("target seen while doing wall action sequence, stopping wall action")
                #we see a target, so we stop getting away from the wall
                self.wall_moves = 0

        if self.initial_inspection_steps_completed < self.number_of_initial_inspection_steps_to_use:
            print("doing initial inspection step ", self.initial_inspection_steps_completed,
                  " there are ", self.number_of_initial_inspection_steps_to_use - self.initial_inspection_steps_completed, " remaining")
            if color == 2:
                #found gold on initial inspection, so we will stop looking around and go get it
                print("found a gold target during initial look around")
                self.initial_inspection_steps_completed = self.number_of_initial_inspection_steps_to_use
                #self.green_found_on_initial_inspection = 0
                #self.green_found_on_initial_inspection_radius = 0.0

            elif color == 1:
                distance = float(state[2]) / (self.max_pixel / 2)
                #print("green radius = {0:.2f}, green height = {1:.2f}, ratio = {2:.2f}", radius, distance, radius*distance)
                if accept_green:  #and self.number_of_look_arounds > 1:
                    print("found a green target during initial look around and we are accepting green targets")
                    self.initial_inspection_steps_completed = self.number_of_initial_inspection_steps_to_use
                elif size > self.green_found_on_initial_inspection_radius:
                    #found a green target, but we are not ready to take green ones yet
                    #remember the location of the one with the biggest radius
                    self.green_found_on_initial_inspection_radius = size
                    self.initial_inspection_steps_completed_when_green_target_seen = self.initial_inspection_steps_completed
                    if self.number_of_initial_inspection_steps_to_use - self.initial_inspection_steps_completed < self.initial_inspection_steps_completed + (
                            60 - self.number_of_initial_inspection_steps_to_use):
                        #shorter to turn back to the green target
                        self.green_found_on_initial_inspection = self.initial_inspection_steps_completed - self.number_of_initial_inspection_steps_to_use
                        if self.green_found_on_initial_inspection == 0:
                            self.green_found_on_initial_inspection = -1
                    else:
                        #shorter to move forward through the full circle.  We prefer this because we see more of the arena
                        self.green_found_on_initial_inspection = self.initial_inspection_steps_completed + (60 - self.number_of_initial_inspection_steps_to_use)
                    print("while looking for a gold target, we found a green one. If we do not find a gold one and time gets short, we will turn {0:2d} steps to get this green".format(
                        self.green_found_on_initial_inspection))

            #if after the two sections above, self.initial_inspection_steps_completed is still < self.number_of_initial_inspection_steps_to_use
            #then we just want to continue the initial look around.
            #if we are in blackout, we want to pause the initial look around-- this is checked for in
            #modify_actions in the blackout section, but in case that changes, we check for it here too
            if self.initial_inspection_steps_completed < self.number_of_initial_inspection_steps_to_use:
                if self.blackout:
                    #we might be moving during blackout if we had a target in sight
                    #or we might be stopped.  Either way, we return no movement from here
                    #since that is controlled by the blackout routine in modified action
                    #if self.stopped_for_blackout:
                    #   print("waiting for blackout to end, then we will resume initial lookaround")
                    return 8  #no movement
                else:
                    self.initial_inspection_steps_completed += 1
                    #we will initially look all the way around for targets until we see a gold one
                    self.previous_color = color
                    return 0  #left turn

        if self.initial_inspection_steps_completed == self.number_of_initial_inspection_steps_to_use:
            if (color == 1 and accept_green and (self.green_found_on_initial_inspection == 0 or size >= self.green_found_on_initial_inspection_radius - .1)) or color == 2:
                self.initial_inspection_steps_completed = self.number_of_initial_inspection_steps_to_use + 1
                print("found a target while looking around, current color = {0:2d}".format(color))
                #input("look at target")
            elif self.green_found_on_initial_inspection != 0 and accept_green:
                print("turning to face the green target we found while looking for a gold one, number and direction of turns to go = {0:2d}, episode step = {1:2d}, time remaining = {2:2d}".format(
                    self.green_found_on_initial_inspection, episode_step, time_remaining))
                if self.green_found_on_initial_inspection < 0:
                    #shorter to go back
                    self.green_found_on_initial_inspection += 1
                    self.previous_color = color
                    return 4  #right turn
                else:
                    #shorter to keep on going through
                    self.green_found_on_initial_inspection -= 1
                    self.previous_color = color
                    return 0  #left turn
            else:
                if self.green_found_on_initial_inspection_radius < 0.01:
                    if accept_green:
                        print("did not see a green or gold target during initial inspection, episode step = {0:2d}".format(episode_step))
                    else:
                        print("did not see a gold target during initial inspection, episode step = {0:2d}".format(episode_step))
                else:
                    if accept_green:
                        if self.turning_toward_previous_target_during_blackout:
                            print("we have turned to the target during blackout, but we will not try to move to it")
                            #do not reset self.steps_to_go to 0 or we will recalculate steps_to_go
                            self.initial_inspection_steps_completed = self.number_of_initial_inspection_steps_to_use
                            self.green_found_on_initial_inspection_radius = 0.0
                            self.stopped_for_blackout = True
                            self.turning_toward_previous_target_during_blackout = False
                            return 8
                        else:
                            print("after trying to face the green target, we did not see it, episode step = {0:2d}".format(episode_step))
                    else:
                        print("we saw a green target, but it is too early to take it, episode step = {0:2d}, remaining_steps = {1:2d}".format(episode_step, time_remaining))
                self.initial_inspection_steps_completed = self.number_of_initial_inspection_steps_to_use + 1
                print("no longer looking around")

        #once we start exploring, we want to forget any green targets seen during the initial lookaround
        #otherwise, during blackouts we will try to turn to those targets, but we will have moved since seeing them
        self.steps_to_go = 0
        self.green_found_on_initial_inspection = 0
        self.green_found_on_initial_inspection_radius = 0.0
        #self.number_of_look_arounds += 1

        if color == 1 and (not accept_green) and radius < self.min_green_radius:
            #print("green target seen while exploring, too small to avoid, too early to take")
            color = 0
        if color == 3:
            if radius < self.min_red_radius:
                #print("red or hotzone target seen while exploring, too small to avoid")
                color = 0
        if color == 4:
            if radius < self.min_hotzone_size:
                #print("red or hotzone target seen while exploring, too small to avoid")
                color = 0
            #else:
            #   print("hotzone seen and needs to be avoided")

        if color == 0:  #if we do not see a target, look around for one

            if self.previous_color == 1 or self.previous_color == 2:
                #if we lost sight of a target that was there on the last image, we want to stop and look around
                self.initial_inspection_steps_completed = 0
                self.previous_color = color
                return 4  #right

            self.previous_color = color
            if self.use_maze_network:
                return np.argmax(self.graph_maze.predict_one(state))
            
            if obstacle_seen: #move backwards when there are obstacles present.
                if (self.headbang or self.tailbang or self.stuck) and self.turning_to_face_new_direction == 0:
                    if self.extra_obstacle_turns > 3:
                        self.extra_obstacle_turns = 0
                        self.turning_to_face_new_direction = 23
                        print("had a headbang or tailbang, setting up to turn to face new direction, turns = ", self.turning_to_face_new_direction)                      
                    else:
                        self.extra_obstacle_turns += 1
                        self.turning_to_face_new_direction = 15
                        print("had a headbang or tailbang, setting up to turn to face new direction, turns = ", self.turning_to_face_new_direction)                      
                if self.turning_to_face_new_direction != 0:
                    self.turning_to_face_new_direction -= 1
                    print("turning right to face new direction.  number of turns remaining = ", self.turning_to_face_new_direction)
                    return 4  #turn right for 90 degrees
                else:
                    print("an obstacle was seen during this episode, doing standard backwards move")
                    return 6  #backwards
            
            #do not use astar with obstacles present-- it just runs into the obstacles
            if accept_green and (not self.astar_done) and astar_stuck == 0 and target_list_length > 0 and (not self.red_color_seen) and (not obstacle_seen):
                print("returning 98")
                return 98

            #vel_z = state[7]
            if self.num_no_target_moves < self.max_num_no_target_moves:
                if self.num_no_target_moves == 0 and self.max_num_no_target_moves == 76:
                    print("starting long obstacle move")
                elif self.num_no_target_moves == 75 and self.max_num_no_target_moves == 76:
                    print("finishing long obstacle move")
                self.num_no_target_moves += 1
                print("moving forward, looking for a target, number of moves to go before turning = ", self.max_num_no_target_moves - self.num_no_target_moves)
                return 2  #forward

            elif self.num_no_target_turns < self.max_num_no_target_turns:  #each turn is 6 degrees, so we will look around 200 degrees and then move forward
                self.num_no_target_turns += 1
                #print("vel_z = ", vel_z, "previous_action = ", previous_action)
                #if vel_z > 1: #slow down to look around, move backward and turn the same direction as previously
                if previous_action[1] == 0 or previous_action[1] == 1:
                    #if we were already turning right (or not turning), keep turning right
                    print("turning right, looking for a target, number of turns to go = ", self.max_num_no_target_turns - self.num_no_target_turns)
                    return 4  #right
                else:
                    print("turning left, looking for a target, number of turns to go = ", self.max_num_no_target_turns - self.num_no_target_turns)
                    return 0  #left
            else:
                self.num_no_target_turns = 0
                self.num_no_target_moves = 0
                print("finished a set of no target turns and moves, resetting the values back to 0")

            return 0  #left

        else:
            self.previous_color = color
            self.num_no_target_moves = 0
            self.num_no_target_turns = 0
            if (color == 1 and accept_green) or color == 2:
                if color == 1:
                    print("moving toward green target")
                else:
                    print("moving toward gold target")
                if self.training:
                    return -1
                old_network_action = np.argmax(self.graph_green.predict_one(state))
                if self.using_old_green_network:
                    network_action = self.transform_old_network_action_to_new_network_action(old_network_action)
                else:
                    network_action = old_network_action

                #if not self.red_color_seen:
                if network_action == 5:
                    #for some reason, we sometimes get oscillations with backward right moves
                    #so we change those selections to forward right
                #    print("changed network action from 5 (backward right) to 3 (forward right)")
                    network_action = 3
                #if network_action == 6:
                #   print("changed network action from 6 (backward) to 2 (forward)")
                #   network_action = 2
                #elif network_action == 7:
                #   #for some reason, we sometimes get oscillations with backward moves
                #   #so we change those selections to forward
                #    print("changed network action from 7 (backward left) to 0 (left)")
                #    network_action = 0
                if network_action == 8:
                    print("changed network action from 8 (no move) to 2 (forward")
                    #we do not want to allow nothing actions, so we will move forward instead
                    network_action = 2  #forward

                return network_action
            elif color == 3 or (color == 1 and (not accept_green)) or color == 4:
                if self.training:
                    return -1
                #self.max_num_no_target_turns = 3
                network_action = np.argmax(self.graph_red.predict_one(state))
                if self.using_old_red_network:
                    network_action = self.transform_old_network_action_to_new_network_action(network_action)
                print("avoiding area or target with color = {0:2d}".format(color))
                return network_action

            print("unknown color!  color reported = ", color, ", accept_green = ", accept_green, ", size = ", size, ", radius = ", radius, ", min green radius = ", self.min_green_radius)
            return random.randint(0, 4)

    def modify_action(self, action, previous_action, velocity, z_velocity, current_blackout, wall_in_view, red_in_view, color, previous_color, obstacle_seen, hotzone, episode_step, time_remaining):
        modified_action = action
        reward = 0.0
        '''
        if self.turning_toward_previous_target_during_blackout:
            #we check blackout so that the permanent and temp blackout flags get set, but we do not use the returned action
            check_blackout(self, action, current_blackout, color, episode_step, time_remaining)
            #if current_blackout:
            #it is much better to continue this even if we are not in current blackout, since we get lined up with
            #green before going back to looking around
            print("we are turning toward previous target during blackout.  current blackout = ", current_blackout)
            #input("see whats up with moving in blackout")
            return reward, modified_action
            #else:
            #   print("blackout is gone, so we stop moving toward previous target")
            #   self.turning_toward_previous_target_during_blackout = False
            #   #input("see what that is about")
        #if random.random() < self.epsilon and (not self.training):
        #   #when training, random actions and epislon are determined when choosing the agent action
        #   print("modified action to be a random_action")
        #   return reward, [float(random.randint(0,1)), float(random.randint(1, 2))]
        '''

        #send it the previous action to know if we headbanged
        #but base the reward on the current action that the agent selected
        if self.check_for_headbang(previous_action, velocity, self.previous_velocity, wall_in_view, current_blackout, episode_step):
            if (self.headbang and action[0] == 1) or (self.tailbang and action[0] == 2):
                reward = -3.0

        '''
        if self.check_for_headbang(previous_action, velocity, self.previous_velocity, wall_in_view, current_blackout, episode_step):
            if (self.headbang and action[0] == 1) or (self.tailbang and action[0] == 2):
                reward = -3.0
        '''
        modified_action = self.check_blackout(modified_action, current_blackout, previous_color, episode_step, time_remaining)
        #if current_blackout:
        #   input("see data in blackout")

        #if not current_blackout:
        #   reward, modified_action = self.check_for_wall(wall, action)

        if not self.stuck:
            if self.check_for_stuck(velocity, self.previous_velocity, action):
                #just became stuck
                reward = -3.0
                #do front left turn when stuck
                modified_action = [1.0, 2.0]
        else:
            if self.check_for_stuck(velocity, self.previous_velocity, action):
                #still stuck, negative reward for not turning when stuck
                reward = -1.0
                #do front left turn when stuck
                modified_action = [1.0, 2.0]
            else:
                #just became unstuck, positive reward
                #we may have become unstuck because of previously modified actions, so we only want to
                #give a positive reward if the agent choice was to move forward on this step.
                if action[0] != 0:
                    reward = 1.0

        if not self.stopped_for_blackout:
            modified_action = self.check_for_oscillation(modified_action)
            #if modified_action[0] != action[0] or modified_action[1] != action[1]:
            #    print("action modified in oscillation check, modified_action = ", modified_action, ", action = ", action)

        if hotzone:
            if (color == 0 or color == 3 or color == 4) and modified_action[0] == 0:
                print("in a hotzone, modifying action to move out")
                if obstacle_seen:
                    modified_action[0] = 2  #backward
                else:
                    modified_action[0] = 1  #forward
                    
        elif red_in_view and z_velocity > 12.0 and modified_action[0] != 2 and (not hotzone):
            print("throttling forward speed because red is in this episode")
            modified_action[0] = 2  # backward
            
        self.previous_action = modified_action
        self.previous_velocity = velocity
        self.previous_z_velocity = z_velocity
        return reward, modified_action

    def check_for_headbang(self, action, velocity, previous_velocity, wall_in_view, blackout, episode_step):
        if velocity < 0.05 and previous_velocity > 2.0 and episode_step != 1:
            #print("headbang at step = {0:2d}".format(episode_step))
            if wall_in_view < 0 and action[0] == 1 and (not blackout):
                print("headbang with no wall in view, must have obstacles present")
                self.headbang_obstacle = True

            if action[0] == 1:
                print("headbang")
                self.headbang = True
            elif action[0] == 2:
                print("tailbang")
                self.tailbang = True
            else:
                print("none_bang")
            return True
        self.headbang = False
        self.tailbang = False
        return False

    def check_for_stuck(self, velocity, previous_velocity, action):
        if velocity < 0.05 and previous_velocity < 0.05 and action[1] == 0 and (not self.stopped_for_blackout):
            if self.stuck_number > self.max_stuck:
                if not self.stuck:
                    print("stuck")
                self.stuck = True
                return True
            else:
                self.stuck_number += 1
                self.stuck = False
                return False
        else:
            if self.stuck:
                print("unstuck")
            self.stuck = False
            self.stuck_number = 0
            return False

    def check_for_oscillation(self, action):
        if action[0] == 0. and action[1] == 0.:
            #print("check for oscillation was sent a non movement action, moving forward left instead")
            return [1.0, 2.0]

        if (action[0] != 0 and action[0] == self.previous_action[0]) or action[1] == self.previous_action[1]:
            #print("oscillation number reset, previously at {0:2d}".format(self.oscillation_number))
            self.oscillation_number = 0
            return action

        if action[0] == 0.0:
            if action[1] == 1.0:
                if self.previous_action[1] == 2.0:
                    if self.oscillation_number > self.max_oscillations:
                        print("left oscillation inhibited")
                        return [1.0, 2.0]
                    else:
                        self.oscillation_number += 1
                        return action
            elif action[1] == 2.0:
                if self.previous_action[1] == 1.0:
                    if self.oscillation_number > self.max_oscillations:
                        print("right oscillation inhibited")
                        return [1.0, 1.0]
                    else:
                        self.oscillation_number += 1
                        return action
            #else:
            #   print("check for oscillation received an unknown value for action[1] = ", action[1])
            self.oscillation_number = 0
            return action

        elif action[0] == 2.0:
            #print("check for oscillation received a backwards move value for action[0] = ", action[0], ", changing to forward")
            #if self.previous_action[0] == 1.0:
            #if self.oscillation_number > self.max_oscillations:
            #       print("backwards oscillation inhibited")
            #       return [1.0, action[1]]
            #else:
            #   self.oscillation_number += 1
            #   return action
            return [1.0, action[1]]

        elif action[0] == 1.0:
            if self.previous_action[0] == 2.0:
                #print("check for oscillation received a previous backwards move value for action[0] = ", action[0])
                if self.oscillation_number > self.max_oscillations:
                    #print("backwards oscillation inhibited")
                    return [2.0, action[1]]
                else:
                    self.oscillation_number += 1
                    return action
        #else:
        #   print("check for oscillation received an unknown value for action[0] = ", action[0])
        self.oscillation_number = 0
        return action

    def check_blackout(self, action, current_blackout, previous_color, episode_step, time_remaining):
        if current_blackout:
            if not self.blackout:
                print("blackout started at episode step = {0:2d}, there are {1:2d} steps remaining, previous color = {2:2d}".format(episode_step, time_remaining, previous_color))
                self.blackout_start_steps = episode_step
                self.blackout = True

                if episode_step == 25 or episode_step == 50:
                    self.permanent_blackouts_happening = True
                    print("Since the lights went out at episode step = {0:2d}, the lights will eventually go out completely".format(episode_step))
                elif not self.permanent_blackouts_happening:
                    if not self.temporary_blackouts_happening:
                        print("Since the lights went out at episode step = {0:2d},  which is not 25 or 50, the blackouts are temporary".format(episode_step))
                    self.temporary_blackouts_happening = True

            if self.permanent_blackouts_happening and (not self.permanent_blackout) and episode_step - self.blackout_start_steps > 5:
                print("we have entered a permanent blackout")
                self.permanent_blackout = True
                #we might be wandering, not stopped, but this prevents other modifiers from altering the command
                #for example, check for oscillation would change backwards moves to forward ones
                self.stopped_for_blackout = True
                if not self.red_color_seen:
                    print("no red was seen, so we will wander around")
                else:
                    print("there are red targets nearby, so we will stop rather than wander")

            #elif self.permanent_blackouts_happening:
            #   print("episode_step - self.blackout_start_steps", episode_step - self.blackout_start_steps)

            if (not self.permanent_blackout) and (previous_color == 1 or previous_color == 2):
                if episode_step - self.blackout_start_steps < 12:
                    #print("moving forward during temporary blackout because the last image had a good target")
                    #when we finish this move, we want to look around for a target
                    self.initial_inspection_steps_completed = 0
                    self.green_found_on_initial_inspection_radius = 0.0
                    self.green_found_on_initial_inspection = 0
                    return [1.0, 0.0]
                else:
                    #print("finished moving forward during temporary blackout, waiting for blackout to end")
                    self.stopped_for_blackout = True
                    return [0.0, 0.0]

            if not self.permanent_blackout and self.green_found_on_initial_inspection_radius < 0.01:
                if not self.stopped_for_blackout:
                    print("stopped for blackout because we have not seen a good target")
                self.stopped_for_blackout = True
                return [0.0, 0.0]

            if (not self.permanent_blackout) and self.green_found_on_initial_inspection_radius >= 0.01:
                #we are turning toward a previously seen green target
                return action
            #print("self.wandering = ", self.wandering)
            #input("wandering")
            if self.astar_done and (not self.astar_finished):
                if not self.astar_wandering:
                    self.wandering = 0
                    self.wandering_moves = 2
                    self.wandering_turns = 5
                    self.astar_wandering = True
                    print("wandering in a spiral after astar")    
                if self.wandering == 0 and self.astar_done:
                    self.wandering = 1
                    self.wandering_cycles = 6
                    #print(self.wandering, self.wandering_cycles, self.wandering_moves, self.wandering_turns)
                    #input("look")
                
                '''
                if self.headbang:
                    self.wandering_cycles = 1
                    self.wandering_moves = 0
                    self.wandering_turns = 15
                    self.astar_finished = True
                    self.wandering = 0
                    print("wandering in reverse toward the center after a headbang, moves remaining = ", -self.wandering_moves)
                    return [2.0, 0.0]
    
                if self.tailbang:
                    self.wandering_cycles = 1
                    self.wandering_moves = 0 #positive wandering moves with 0 turns means move directly forward
                    self.wandering_turns = -15
                    self.astar_finished = True
                    self.wandering = 0
                    print("wandering forward toward the center after a tailbang, moves remaining = ", self.wandering_moves)
                    return [1.0, 0.0]
                '''
                
                if self.wandering_cycles > 0:
                    if self.wandering_moves != 0:
                        if self.wandering_moves > 0:
                            self.wandering_moves -= 1
                            print("wandering forward, moves remaining = ", self.wandering_moves)
                            return [1.0, 0.0]
                        else:
                            self.wandering_moves += 1 # move backward
                            print("wandering backward, moves remaining = ", -self.wandering_moves)
                            return[2.0, 0.0]
    
                    if self.wandering_turns != 0:
                        if self.wandering_turns > 0:
                            self.wandering_turns -= 1
                            print("wandering left, turns remaining = ", self.wandering_turns)
                            return[0.0, 2.0]
                        else:
                            self.wandering_turns += 1
                            print("wandering right, turns remaining = ", -self.wandering_turns)
                            return[0.0, 1.0]
                    #we get to here when the cycle of moves and turns is done
                    self.wandering_cycles -= 1
                    
                    #note that wandering_moves and wandering_turns both = 0 right now
                    self.wandering_moves = self.wandering + 2
                    self.wandering_turns = 5
                    print("wandering left, starting another cycle, wandering cycles remaining = ", self.wandering_cycles, ", self.wandering_moves = ", self.wandering_moves)
                    return [0.0, 2.0]
                
                self.wandering += 1
                self.wandering_cycles = 6
                # note that wandering_moves and wandering_turns both = 0 right now
                self.wandering_moves = self.wandering + 2
                self.wandering_turns = 5
                if self.wandering > 3: # and time_remaining < 200:
                    self.astar_finished = True
                    self.wandering = 0
                    print("wandering forward left, astar wandering finished")
                    #input("astar wandering finished")
                return [1.0, 2.0]
                

            '''
            if not self.red_color_seen:
                #if there is no red out there, then we are free to explore in the dark
                if self.use_maze_network:
                    return self.transform_network_action_to_model_action(np.argmax(self.graph_maze.predict_one(state)))
                else:
                    #print("wandering during permanent blackout because no red targets were seen earlier")
                    #return reward, [float(random.randint(0,1)), float(random.randint(1, 2))]
            '''
            
            if self.wandering == 0:
                self.wandering = 1
                self.wandering_moves = 37
                self.wandering_turns = 19
            if self.headbang:
                self.wandering_moves = 0
                self.wandering_turns = 15
            if self.tailbang:
                self.wandering_moves = 0
                self.wandering_turns = -15
            if self.wandering_moves > 0:
                self.wandering_moves -= 1
                return [1.0, 0.0]
            elif self.wandering_moves < 0:
                self.wandering_moves += 1
                return [2.0, 0.0]
            if self.wandering_turns > 0:
                self.wandering_turns -= 1
                return [0.0, 2.0]
            elif self.wandering_turns < 0:
                self.wandering_turns += 1
                return [0.0, 1.0]

            self.wandering_moves = 37
            self.wandering_turns = 19
            return [1.0, 2.0]
                    
            '''
            else:
                #if there is red out there, maybe better to not move around
                if not self.stopped_for_blackout:
                    print("stopped for blackout because a red target was seen in the last image")
                self.stopped_for_blackout = True
                return [0.0, 0.0]
            '''
        elif self.blackout:
            self.blackout = False
            self.stopped_for_blackout = False
            print("blackout ended at episode step = {0:2d}, there are {1:2d} steps remaining".format(episode_step, time_remaining))
            if episode_step - self.blackout_start_steps < 10:
                print("since the blackout lasted {0:2d} steps, we expect that the lights will eventually go out completely".format(episode_step - self.blackout_start_steps))
                self.permanent_blackouts_happening = True
            else:
                print("since the blackout lasted {0:2d} steps, we expect that the lights will go out periodically".format(episode_step - self.blackout_start_steps))

            if episode_step - self.blackout_start_steps >= 10 and (not self.permanent_blackout):
                self.temporary_blackouts_happening = True
            return [0.0, 2.0]
        return action

