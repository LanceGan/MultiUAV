import numpy as np
import math


class User(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class UAV(object):
    def __init__(self, x, y, h):
        self.x = x
        self.y = y
        self.h = h

    def move_inside(self, phi, dist,distmax):
        phi = np.clip(phi,-np.pi,np.pi)
        dist = np.clip(dist,0,distmax)  
        self.x = self.x + dist * np.cos(phi)
        self.y = self.y + dist * np.sin(phi)
   



