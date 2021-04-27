# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math
import time



# # postion coordinates
# # for OAK-D: Switch Z and Y coordinates
# z = [0, 1]
# x = [0, 2]
# y = [0, 3]
# # time array
# t = [0, 1]

def vectors(x,y,z,t):
    # theta = math.degrees(math.atan((z[1] - z[0])/(y[1] - y[0])))
    # print("theta: " + str(theta))
    v0x = ((x[1] - x[0])/(t[1] - t[0]))
    v0y = ((y[1] - y[0])/(t[1] - t[0]))
    v0z = ((z[1] - z[0])/(t[1] - t[0]))
    print("vx: " + str(v0x) + " vy: " + str(v0y) + " vz: " + str(v0z))
    return v0x, v0y, v0z

    # v0 = math.sqrt(math.pow(v0x, 2) + math.pow(v0y, 2))
    # print("v0: " + str(v0))



def trajectories(x,y,z,t):
    theta = math.degrees(math.atan((z[1] - z[0])/(y[1] - y[0])))
    print("theta: " + str(theta))
    v0x = ((x[1] - x[0])/(t[1] - t[0]))
    v0y = ((y[1] - y[0])/(t[1] - t[0]))
    v0z = ((z[1] - z[0])/(t[1] - t[0]))
    #print("vx: " + str(v0x) + " vy: " + str(v0y) + " vz: " + str(v0z))

    v0 = math.sqrt(math.pow(v0x, 2) + math.pow(v0y, 2))
    # print("v0: " + str(v0))

    distance = (math.pow(v0, 2) * math.sin(math.radians(2*theta)))/9.807
    print("distance: " + str(distance))



    ## PLOT TRAJECTORY WITH MATPLOTLIB
    # xx = np.linspace(0, distance, 100)
    # #yy = math.tan(math.radians(theta))*xx - 9.8*math.pow(xx, 2)/(2*math.pow(v0,2)*math.pow((math.cos(math.radians(theta))),2))
    # # yy = math.tan(math.radians(theta))*xx - 9.8*xx**2/(2*v0**2*math.pow((math.cos(math.radians(theta))),2))
    # yy = (math.tan(math.radians(theta))*xx - 9.8*xx**2/(2*v0**2*math.cos(math.radians(theta))**2))
    # zz = 0
    # fig = plt.figure(figsize=(10, 5))
    # # Create the plot
    #
    # ax = plt.axes(projection="3d")
    #
    # # Creating plot
    # ax.scatter3D(xx, yy, zz, color="green")
    # ax.plot(xx,yy,zz, color='r')
    # plt.title("simple 3D scatter plot")
    # ax.set_xlabel('X-axis', fontweight ='bold')
    # ax.set_ylabel('Y-axis', fontweight ='bold')
    # ax.set_zlabel('Z-axis', fontweight ='bold')
    # # plt.plot(xx, yy, zz)
    #
    # # Show the plot
    # plt.show()

# trajectories(x,y,z,t)