import numpy as np

class Explore:
    def __init__(self, delta_timestep):
        self.delta_timestep = delta_timestep
        self.reset_initial_values()

    def reset_initial_values(self):
        self.x = 0.0
        self.z = 0.0
        self.yaw = 0.0

    def update_current_location(self, x, z, yaw):
        self.x = x
        self.z = z
        self.yaw = yaw
        
    def get_directions_to_move_toward(self, x, z):
        angle = 0.0
        if abs(x - self.x) < 0.02:
            if x - self.x >= 0:
                angle = 1.57
            else:
                angle = -1.57
        else:
            slope = (z - self.z) / (x - self.x)
            angle = np.arctan(slope)
        distance = np.sqrt(((x - self.x) * (x - self.x)) + ((z - self.z) * (z - self.z)))
        return distance, angle

    def transform_floor_mask_to_map(self, image, map):
        pixel_values = np.zeros((40,40), dtype=float)
        #for z in range(83, -1, -1): #image row 83 is at the bottom, row 0 is at the top
        for z in range(31):
            for x in range(84):
                if z <= 11:
                    if x < 42:
                        pixel_values[39][19] += image[x][83-z]
                    else:
                        pixel_values[39][20] += image[x][83-z]
                elif z <= 20:
                    if z == 12:
                        pixel_values[39][19] /= 42.
                        pixel_values[39][20] /= 42.
                    if x < 13:
                        pixel_values[38][19] += image[x][83-z]
                    elif x < 27:
                        pixel_values[38][20] += image[x][83-z]
                    else:
                        pixel_values[38][21] += image[x][83-z]
                elif z <= 25:
                    if z == 21:
                        pixel_values[38][19] /= 42.
                        pixel_values[38][20] /= 42.
                        pixel_values[38][21] /= 42.
                    if x < 8:
                        pixel_values[37][18] += image[x][83-z]
                    elif x < 16:
                        pixel_values[37][19] += image[x][83-z]
                    elif x < 24:
                        pixel_values[37][20] += image[x][83-z]
                    elif x < 32:
                        pixel_values[37][21] += image[x][83-z]
                    else:
                        pixel_values[37][22] += image[x][83-z]
                elif z <= 29:
                    if z == 26:
                        pixel_values[37][18] /= 42.
                        pixel_values[37][19] /= 42.
                        pixel_values[37][20] /= 42.
                        pixel_values[37][21] /= 42.
                        pixel_values[37][22] /= 42.
                    if x < 5:
                        pixel_values[36][16] += image[x][83-z]
                    elif x < 10:
                        pixel_values[36][17] += image[x][83-z]
                    elif x < 15:
                        pixel_values[36][18] += image[x][83-z]
                    elif x < 20:
                        pixel_values[36][19] += image[x][83-z]
                    elif x < 25:
                        pixel_values[36][20] += image[x][83-z]
                    elif x < 30:
                        pixel_values[36][21] += image[x][83-z]
                    elif x < 35:
                        pixel_values[36][22] += image[x][83-z]
                    else:
                        pixel_values[37][23] += image[x][83-z]
                if z == 30:
                    pixel_values[36][16] /= 42.
                    pixel_values[36][17] /= 42.
                    pixel_values[36][18] /= 42.
                    pixel_values[36][19] /= 42.
                    pixel_values[36][20] /= 42.
                    pixel_values[36][21] /= 42.
                    pixel_values[36][22] /= 42.
                    pixel_values[36][23] /= 42.
            
            for z in range(16,24):
                for x in range(36,40):
                    if pixel_values[x][z] > 0.5: #if we see floor, mark the spot as unoccupied
                        map[x][z] = 0
            return map

