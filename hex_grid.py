import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

# Definitions
radius = 2
hexagon_twist_angle = 60
grid_twist_angle = 60
gap_factor = 0.999

# Convert angles to radians
hexagon_twist_angle_rad = np.deg2rad(hexagon_twist_angle)
grid_twist_angle_rad = np.deg2rad(grid_twist_angle)

# Generate honeycomb centers
center_x = []
center_y = []

range_val = 10
for i in range(-range_val, range_val + 1):
    for j in range(-range_val, range_val + 1):
        cx = (3 * radius * j) - (j) * 1.5 * radius
        cy = np.sqrt(3) * radius * i + (j % 2) * np.sqrt(3) * radius / 2

        center_x.append(cx)
        center_y.append(cy)

# Convert lists to NumPy arrays for element-wise operations
center_x = np.array(center_x)
center_y = np.array(center_y)

# Perform rotation transformations
rotated_center_x = center_x * np.cos(hexagon_twist_angle_rad) - center_y * np.sin(hexagon_twist_angle_rad)
rotated_center_y = center_x * np.sin(hexagon_twist_angle_rad) + center_y * np.cos(hexagon_twist_angle_rad)

grid_center_x = rotated_center_x * np.cos(grid_twist_angle_rad) - rotated_center_y * np.sin(grid_twist_angle_rad)
grid_center_y = rotated_center_x * np.sin(grid_twist_angle_rad) + rotated_center_y * np.cos(grid_twist_angle_rad)

# Generate twisted honeycomb grid
x = []
y = []
waypoints = []

for i in range(len(grid_center_x)):
    t = np.linspace(0, 2 * np.pi, 7) + hexagon_twist_angle_rad
    x_cell = grid_center_x[i] + gap_factor * radius * np.cos(t + grid_twist_angle_rad)
    y_cell = grid_center_y[i] + gap_factor * radius * np.sin(t + grid_twist_angle_rad)

    x.extend(x_cell)
    y.extend(y_cell)

    cx = grid_center_x[i]
    cy = grid_center_y[i]

    waypoints.append([cx, cy])

# Generate a circular fence
num_no_go_circles = 2
no_go_circle_radius = 5
no_go_circle_centers = np.array([[-5, 5], [5, -5]])

fence_radius = 15
center_fence_x = 0
center_fence_y = 0
theta = np.linspace(0, 2 * np.pi, 100)
fence_x = center_fence_x + fence_radius * np.cos(theta)
fence_y = center_fence_y + fence_radius * np.sin(theta)

# Generate waypoints inside and outside the fence, excluding no-go zones
waypoints_inside_fence = []
waypoints_outside_fence = []

for waypoint in waypoints:
    x, y = waypoint
    distance_to_fence_center = np.sqrt((x - center_fence_x)**2 + (y - center_fence_y)**2)

    # Check if the waypoint is inside the fence but outside of the no-go zones
    inside_fence = distance_to_fence_center <= fence_radius
    inside_no_go_zones = any(
        np.sqrt((x - no_go_circle_centers[:, 0])**2 + (y - no_go_circle_centers[:, 1])**2) <= no_go_circle_radius
    )
    if inside_fence and not inside_no_go_zones:
        waypoints_inside_fence.append(waypoint)
    else:
        waypoints_outside_fence.append(waypoint)

# Calculate the distance matrix for waypoints inside the fence
distance_matrix = distance_matrix(waypoints_inside_fence, waypoints_inside_fence)

# Solve the Traveling Salesman Problem (TSP) using Linear Programming
row_ind, col_ind = linear_sum_assignment(distance_matrix)

# Reorder the waypoints based on the TSP solution
tsp_order = np.array(waypoints_inside_fence)[col_ind]

# Plot the results
plt.figure(figsize=(12, 6))

# Plot the hexagon grid with blue centers
for x, y in zip(grid_center_x, grid_center_y):
    hexagon_vertices_x = x + radius * np.cos(np.linspace(0, 2 * np.pi, 7))
    hexagon_vertices_y = y + radius * np.sin(np.linspace(0, 2 * np.pi, 7))
    plt.plot(hexagon_vertices_x, hexagon_vertices_y, 'k')
    plt.plot(x, y, 'b.', markersize=4)

# Plot the circular fence
plt.plot(fence_x, fence_y, 'r-', linewidth=1.5)

# Plot the no-go circles
for j in range(num_no_go_circles):
    no_go_circle_x = no_go_circle_centers[j, 0] + no_go_circle_radius * np.cos(theta)
    no_go_circle_y = no_go_circle_centers[j, 1] + no_go_circle_radius * np.sin(theta)
    plt.plot(no_go_circle_x, no_go_circle_y, 'm--', linewidth=1)

plt.title('Hexagon Grid with Circular Fence and No-Go Circles')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(['Hexagon Grid', 'Hexagon Centers', 'Fence', 'No-Go Circles'])
plt.axis('equal')

# Plot an interactive hexagon grid
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'k')
plt.plot(grid_center_x, grid_center_y, 'b.', markersize=4)
plt.plot(fence_x, fence_y, 'r-', linewidth=1.5)
for j in range(num_no_go_circles):
    no_go_circle_x = no_go_circle_centers[j, 0] + no_go_circle_radius * np.cos(theta)
    no_go_circle_y = no_go_circle_centers[j, 1] + no_go_circle_radius * np.sin(theta)
    plt.plot(no_go_circle_x, no_go_circle_y, 'm--', linewidth=1)
plt.title('Interactive Hexagon Grid')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(['Hexagon Grid', 'Hexagon Centers', 'Fence', 'No-Go Circles'])
plt.axis('equal')

# Plot the TSP solution inside the fence, avoiding the no-go zones
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'k')
plt.plot(grid_center_x, grid_center_y, 'b.', markersize=4)
plt.plot(fence_x, fence_y, 'r-', linewidth=1.5)
for j in range(num_no_go_circles):
    no_go_circle_x = no_go_circle_centers[j, 0] + no_go_circle_radius * np.cos(theta)
    no_go_circle_y = no_go_circle_centers[j, 1] + no_go_circle_radius * np.sin(theta)
    plt.plot(no_go_circle_x, no_go_circle_y, 'm--', linewidth=1)
plt.plot(tsp_order[:, 0], tsp_order[:, 1], 'g-', linewidth=1.5)
plt.title('TSP Tour Inside the Fence (Avoiding No-Go Zones)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(['Hexagon Grid', 'Hexagon Centers', 'Fence', 'No-Go Circles', 'TSP Tour'])
plt.axis('equal')

# Count waypoints inside the fence and inside the no-go zones
num_waypoints_inside_fence = 0
num_waypoints_inside_no_go_zones = 0

for waypoint in waypoints:
    # Check if the waypoint is inside the fence
    distance_to_fence_center = np.sqrt((waypoint[0] - center_fence_x)**2 + (waypoint[1] - center_fence_y)**2)
    inside_fence = distance_to_fence_center <= fence_radius
    
    # Check if the waypoint is inside any of the no-go zones
    inside_no_go_zones = False
    for j in range(num_no_go_circles):
        no_go_circle_center = no_go_circle_centers[j]
        distance_to_no_go_center = np.sqrt((waypoint[0] - no_go_circle_center[0])**2 + (waypoint[1] - no_go_circle_center[1])**2)
        if distance_to_no_go_center <= no_go_circle_radius:
            inside_no_go_zones = True
            break
    
    if inside_fence:
        num_waypoints_inside_fence += 1
    if inside_no_go_zones:
        num_waypoints_inside_no_go_zones += 1

num_waypoints_inside_fence_excluding_no_go_zones = num_waypoints_inside_fence - num_waypoints_inside_no_go_zones

print(f'Number of waypoints inside the fence: {num_waypoints_inside_fence}')
print(f'Number of waypoints inside no-go zones: {num_waypoints_inside_no_go_zones}')
print(f'Number of waypoints inside the fence (excluding no-go zones): {num_waypoints_inside_fence_excluding_no_go_zones}')
'''
# Create a plot
fig, ax = plt.subplots(figsize=(8, 6))

# Function to update the grid's position when panning
def on_pan(event):
    if event.inaxes == ax:
        # Calculate the translation based on mouse movement
        translation_x = event.xdata - event.xdata_prev
        translation_y = event.ydata - event.ydata_prev

        # Apply the translation to the grid's centers
        grid_center_x += translation_x
        grid_center_y += translation_y

        # Update the plot
        ax.clear()

        # Plot the hexagon grid with blue centers
        for x, y in zip(grid_center_x, grid_center_y):
            hexagon_vertices_x = x + radius * np.cos(np.linspace(0, 2 * np.pi, 7))
            hexagon_vertices_y = y + radius * np.sin(np.linspace(0, 2 * np.pi, 7))
            ax.plot(hexagon_vertices_x, hexagon_vertices_y, 'k')
            ax.plot(x, y, 'b.', markersize=4)

        # Plot the circular fence
        ax.plot(fence_x, fence_y, 'r-', linewidth=1.5)

        # Plot the no-go circles
        for j in range(num_no_go_circles):
            no_go_circle_x = no_go_circle_centers[j, 0] + no_go_circle_radius * np.cos(theta)
            no_go_circle_y = no_go_circle_centers[j, 1] + no_go_circle_radius * np.sin(theta)
            ax.plot(no_go_circle_x, no_go_circle_y, 'm--', linewidth=1)

        ax.set_title('Interactive Hexagon Grid')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(['Hexagon Grid', 'Hexagon Centers', 'Fence', 'No-Go Circles'])
        ax.axis('equal')

        plt.draw()

# Connect the panning event to the plot
fig.canvas.mpl_connect('motion_notify_event', on_pan)
'''
plt.show()