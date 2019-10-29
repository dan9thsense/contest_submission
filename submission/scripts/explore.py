
class Explore:
    def __init__(self, delta_timestep):
        self.delta_timestep = delta_timestep
        self.number_of_corners_found = 0
        self.first_bang = True
        self.reset_initial_values()

    def reset_initial_values(self):
        self.moves = -50
        self.turns = -30
        self.direction = -1
        self.corner_turns = 0
        self.wall_turns = 0
        self.stuck_turns = 0
        self.corner_is_nearby = 0
        self.previous_velocity = 0.0
        self.distance = 0.0
        self.bang = False
        self.bang_min_vel_reached = False
        self.floor_not_seen = False
        self.stuck = 0
        self.num_moves = 0
        self.low_vel_moves = 0




    def wall_follow(self, previous_action, wall_in_view, floor_size, velocity):
        print(previous_action, wall_in_view, floor_size, velocity, ", moves = ", self.moves, ", turns = ", self.turns, self.stuck_turns, self.wall_turns)
        if velocity > 2.0 and (not self.bang_min_vel_reached):
            print("bang min velocity reached")
            self.bang_min_vel_reached = True
        if velocity < 0.5 and self.bang_min_vel_reached:
            self.bang = True
            self.bang_min_vel_reached = False
            if self.first_bang:
                print("first bang, we will turn left")
                self.first_bang = False
                self.moves = 0
                self.turns = 29
                return 0
            else:
                if self.direction > 0:
                    print("when bang occured, we were going forward, so we will turn right 270 degrees")
                    self.moves = 0
                    self.turns = -39
                    return 0
                else:
                    print("when bang occurred, we were going backwards, so we will turn right 90 degrees")
                    self.moves = 0
                    self.turns = -13
                    return 4

        elif velocity < 0.3 and previous_action[1] == 0:
            if self.stuck > 4:
                self.reset_initial_values()
                print("got stuck, turning now")
                self.stuck_turns = 2
            else:
                self.stuck += 1

        delta_distance = ((velocity + self.previous_velocity) / 2.0) * self.delta_timestep
        self.distance += delta_distance
        self.previous_velocity = velocity
        print("velocity = {0:.2f}, distance = {1:.2f}".format(velocity, self.distance))

        # velocity checks.  If the velocity is not increasing properly, we may be dragging along the wall
        if (self.num_moves == 6 and velocity < 8.0) or (self.num_moves == 15 and delta_distance < 1.0):
            if self.num_moves == 6:
                print("at move 6, velocity = {0:.2f}, which is too slow, must be dragging along, need to turn a bit".format(velocity))
            else:
                print("at move 15, delta_distance = {0:.2f}, and velocity = {1:.2f}. This is too short, need to turn a bit".format(delta_distance, velocity))

            # if self.direction > 0:
            # moving forwards, so we need to turn left
            self.low_vel_moves = 2
            self.num_moves = 0
            return 0
            # else:
            #    #moving backwards, so we need to turn right
            #    self.low_vel_moves = 2
            #    self.num_moves = 0
            #    return 4    

        if self.wall_turns < 0:
            print("there was a wall to the left or front, turn to the right, wall turns =", self.wall_turns)
            if wall_in_view == 0 or wall_in_view == 1:
                self.wall_turns = -3
            else:
                self.wall_turns += 1
            return 4

        if self.wall_turns > 0:
            print("there was a wall to the right, turn to the left, wall_turns = ", self.wall_turns)
            if wall_in_view == 2:
                self.wall_turns = 3
            else:
                self.wall_turns -= 1
            return 0

        if self.floor_not_seen:
            print("there is an obstacle in front")
            # turn until we see enough floor to move and then move forward into it
            if floor_size > 2000:
                self.floor_not_seen = False
                self.moves = 50
                self.direction = 1
                return 2
            return 0

        if self.stuck_turns > 0:
            print("turning left to get unstuck")
            self.stuck_turns -= 1
            return 0

        if self.low_vel_moves > 0:
            self.low_vel_moves -= 1
            print("turning left due to low velocity")
            return 0
        if self.low_vel_moves < 0:
            print("turning right due to low velocity")
            self.low_vel_moves += 1
            return 4

        print("self.moves = ", self.moves, ", self.direction = ", self.direction)

        # go backwards until we hit something
        if self.moves < 0:
            self.moves += 1
            self.num_moves += 1
            return 6
        # go forwards until we hit something
        if self.moves > 0:
            self.moves -= 1
            self.num_moves += 1
            return 2
        '''
        if self.distance > 19. and self.distance < 21.:
            if previous_action[0] == 1:
                #we were going forwards, so we will turn right 180 degrees
                self.moves = 0
                self.turns = -30
            if previous_action[0] == 2:
                #we were moving backwards, so we will turn left 180 degrees
                self.moves = 0
                self.turns = 30
        '''

        if self.distance > 34. and self.distance < 42. and self.corner_turns == 0:
            self.number_of_corners_found += 1
            print("found a corner,  number of corners found = {0:2d}".format(self.number_of_corners_found))
            if self.direction > 0:
                # we were going forwards, so we will turn left 270 degrees 
                self.reset_initial_values()
                self.moves = 0
                self.corner_turns = 40
                # and then move backwards
                self.corner_is_nearby = -1
            else:
                # we were moving backwards, so we will turn right 90 degrees
                self.reset_initial_values()
                self.moves = 0
                self.corner_turns = -13
                # and then move forwards
                self.corner_is_nearby = 1
            # we are going to turn left 45 times, but we set the counter at 44 so that we can use it to tell later when we finished the turns
            # self.corner_turns = 44

        if self.corner_turns > 0:
            print("turning left in a corner, number of corner turns = ", self.corner_turns)
            self.corner_turns -= 1
            return 0

        if self.corner_turns < 0:
            print("turning right in a corner, number of corner turns = ", self.corner_turns)
            self.corner_turns += 1
            return 4

        if self.corner_is_nearby != 0:
            if self.corner_is_nearby == 1:
                print("finished the corner turn, now moving forward")
                self.corner_is_nearby = 0
                self.moves = 50
                self.direction = 1
                self.turns = -45
                return 2
            else:
                print("finished the corner turn, now moving backwards")
                self.corner_is_nearby = 0
                self.moves = -50
                self.direction = -1
                self.turns = 15
                return 6

        # finished turning, but we did not go a distance equal to the arena wall length
        # so we may not be in a corner

        # first check for walls
        if wall_in_view >= 0:
            self.reset_initial_values()
            # note that we do the turns, we watch for when the wall is gone and the do a couple more turns
            if wall_in_view == 0:
                print("wall is seen on the left, we turn right a bit and then will try moving backwards to get to a corner")
                self.wall_turns = -3
                return 4
            elif wall_in_view == 1:
                print("wall is seen in front, we turn right and then will try moving backwards to get to a corner")
                self.wall_turns = -3
                return 4
            elif wall_in_view == 2:
                print("wall is seen on the right, we turn left a bit and then will try moving forwards to get to a corner")
                self.wall_turns = 3
                return 0

        # next check if we facing an obstacle
        if floor_size <= 1500:
            print("we do not see much of a floor and we have not seen a wall, must be facing an obstacle, we will trn left and reset to initial values")
            # so we must be facing an obstacle
            # we will try moving along that obstacle
            self.reset_initial_values()
            self.floor_not_seen = True
            return 0

        if self.turns > 0:
            print("turning left")
            self.turns -= 1
            return 0

        if self.turns < 0:
            print("turning right")
            self.turns += 1
            return 4

        print("finished turning, might be in a corner, not sure, time to move again")
        self.reset_initial_values()
        if previous_action[1] == 2:
            print("we were turning left, so we want to move backwards")
            self.moves = -50
            self.direction = -1
            return 6
        if previous_action[1] == 1:
            print("we were turning right, so we want to move forwards")
            self.moves = 50
            self.direction = 1
            return 2

        print("reached the end of wall following. previous action = ", previous_action)
        self.reset_initial_values()
        return 6