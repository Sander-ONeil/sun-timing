import matplotlib.pyplot as plt
import numpy as np

def vec(a,b):
    return np.array([a,b],dtype=np.float64)
def vec3(a,b,c):
    return np.array([a,b,c],dtype=np.float64)
def normalize(a):
    return a/np.linalg.norm(a)

points_per_day = 24
timescale = 365.25*points_per_day#per year
time = int(points_per_day*365.25)

# Calculate the distance of the fc = np.sqrt(a**2 - b**2)
f1 = vec(-783.79/2,0)
f2 = vec(783.79/2,0)

x = np.zeros((time,2))
x_analema = x+0
c_analema = np.zeros((time))
time_array =  np.zeros((time))
plast = vec(23455,0)

#totaldist =  46909
d_sun = f1 - plast
d_2 = f2 - plast
l1 = np.linalg.norm(d_sun)
l2 = np.linalg.norm(d_2)
totaldist = l1 + l2

a = totaldist/2
c = abs(f1[0] - f2[0])/2
b = np.sqrt(a*a-c*c)
print('abc',a,b,c)

#specific energy = v^2/2 + mu/r
initial_speed =1.4484e5+.999e2
orbitalenergy = 1.087188e10
mu = 5.1e14
orbitalenergy = initial_speed*initial_speed/2 - mu/(np.linalg.norm(d_sun))
print('orbital energy',orbitalenergy)
last_angle = 0
last_m = 0
total_angle = 0
x[0] = plast
m_change_expected = 2*np.pi / 365.25

def transform_planet(p,t):
    
    angle = t*np.pi*2
    day_matrix = np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])

    angle_degrees = 23.5
    earth_tilt = np.deg2rad(angle_degrees)
    # Define the rotation matrix for rotation about the Y-axis
    tilt_matrix = np.array([
        [np.cos(earth_tilt), 0, np.sin(earth_tilt)],
        [0, 1, 0],
        [-np.sin(earth_tilt), 0, np.cos(earth_tilt)]
    ])
    # Perform the rotation
    rotated_point = np.dot(tilt_matrix, np.dot(day_matrix, p))
    return rotated_point

def rev_transform_planet(p,t):
    
    angle = -t*np.pi*2/points_per_day*364.25/365.25
    day_matrix = np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])

    angle_degrees = 23.5
    earth_tilt = np.deg2rad(angle_degrees)
    # Define the rotation matrix for rotation about the Y-axis
    tilt_matrix = np.array([
        [np.cos(earth_tilt), 0, np.sin(earth_tilt)],
        [0, 1, 0],
        [-np.sin(earth_tilt), 0, np.cos(earth_tilt)]
    ])


    angle_tilt_to_elipse = 0.22363
    day_tilt_to_elipse = np.array([
        [np.cos(angle_tilt_to_elipse), np.sin(angle_tilt_to_elipse), 0],
        [-np.sin(angle_tilt_to_elipse), np.cos(angle_tilt_to_elipse), 0],
        [0, 0, 1],
    ])
    # Perform the rotation
    p = vec3(p[0],p[1],0)
    rotated_point = np.dot(day_matrix, np.dot(tilt_matrix, np.dot(day_tilt_to_elipse,p)))
    rotated_point = normalize(rotated_point)
    angle = -t*np.pi*2/points_per_day#*(364.25/365.25)**2
    return (vec(np.arctan2(rotated_point[1],rotated_point[0]),rotated_point[2]),angle%(2*np.pi))
    #return rotated_point

for n in range(1,time):

    d_sun = f1 - plast
    d_2 = f2 - plast

    ## SPEED calc
    r = np.linalg.norm(d_sun)
    v = np.sqrt(2*(orbitalenergy + mu/r))


    #print(v)
    normal = normalize(normalize(d_sun) + normalize(d_2))
    forward = vec(normal[1],-normal[0])
    plast += forward*v/timescale

    analema =rev_transform_planet( f1 - plast,n)
    x_analema[n] = analema[0]
    c_analema[n] = analema[1]
    p_centered = plast




    angle = np.arctan2(plast[1]/b,plast[0]/a)

    ##########Correction
    M = angle - 0.0167086*np.sin(angle-np.pi)
    dif = M-last_m
    goal_dif = 2*np.pi / timescale
    goal_to_actual_ratio = goal_dif/dif

    angle_change = angle - last_angle
    if angle_change > 0:
        goal_angle_change = angle_change*goal_to_actual_ratio
        angle = last_angle + goal_angle_change
    ################################################################
    plast = vec(np.cos(angle)*a,np.sin(angle)*b)

    
    # print(M,M-last_m)  
    last_m = M+0

    total_angle += max(angle - last_angle,0)
    last_angle = angle+0
    

    
    time_array[n] = n/timescale
    x[n] = plast

print(plast)



# # Generate points for the ellipse
# # theta = np.linspace(0, 2 * np.pi, 100)
# # x = a * np.cos(theta)
# # y = b * np.sin(theta)

# # Plotting the ellipse and the foci
# # 
# # fig = plt.figure()
# # ax = fig.add_subplot(projection='3d')
# # ax.scatter(x[:,0],x[:,1],n_array)
# #plt.figure(figsize=(8, 6))
# #plt.plot(x[:,0],x[:,1], 'bo', label='Ellipse')
# fig = plt.figure()
# plt.scatter(x_analema[:,0],x_analema[:,1], label='Ellipse', c=c_analema/(2*np.pi) * 24, cmap = 'viridis')
# plt.colorbar(label="hour")
# #plt.scatter(f1[0],f1[1], color='red', label='Foci')
# #plt.scatter( f2[0],f2[1], color='red', label='Foci')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Plot of an Ellipse with Foci')
# plt.grid(True)
# plt.legend()
# plt.axis('equal')  # Equal scaling for both axes

# ax = fig.add_subplot(projection='3d')
# n_array = np.arange(0,time)
# ax.scatter(x[:,0],x[:,1],n_array)

# plt.show()
fig, ax = plt.subplots(2, 2, figsize=(12, 6))

ax1 = ax[0,0]
ax2 = ax[1,0]
ax3 = ax[0,1]
ax4 = ax[1,1]
# First subplot: 2D scatter plot
sc = ax1.scatter(x_analema[:,0], x_analema[:,1], label='Ellipse', c=c_analema/(2*np.pi) * 24, cmap='twilight_shifted')
cbar = plt.colorbar(sc, ax=ax1, label="hour")
#ax1.scatter(f1[0], f1[1], color='red', label='Foci')
#ax1.scatter(f2[0], f2[1], color='red', label='Foci')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('Y-axis')
ax1.set_title('Anelema')
ax1.grid(True)
#ax1.legend()
ax1.axis('equal')
ax2.axis('equal')  # Equal scaling for both axes
ax3.axis('equal')
# Second subplot: 3D scatter plot


day_array = (time_array*365.25 + 182 )% 365.25


sc3 = ax3.scatter(x_analema[:,0], x_analema[:,1], label='Ellipse', c=day_array, cmap='Set3')
cbar = plt.colorbar(sc3, ax=ax3, label="day")

n_array = np.arange(0, time)
elipse = ax2.scatter(x[:,0], x[:,1], c = day_array,cmap = 'hsv', label = 'earth')
cbar = plt.colorbar(elipse, ax=ax2, label="Day",)
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')
ax2.scatter(f1[0], f1[1], color='red', label='Sun Foci 1')
ax2.scatter(f2[0], f2[1], color='red', label='Foci 2')
ax2.set_title('orbit plot')
ax2.legend()
ax2.grid()
plt.tight_layout()
plt.show()
