import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def fig_to_np_array(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()

    renderer = canvas.get_renderer()
    raw_data = renderer.buffer_rgba()
    
    width, height = canvas.get_width_height()
    img = np.frombuffer(raw_data, dtype=np.uint8)
    img = img.reshape(height, width, 4)
    img = img[:, :, :3]
    return img

def visualize_scene(camera_position, screen_center, screen_normal, screen_width, screen_height, 
                    ray_origin, ray_direction, intersection_point, view_el, view_az, show_fig=True):
    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the screen plane
    screen_u = np.cross(screen_normal, np.array([0, 1, 0]))
    if np.linalg.norm(screen_u) < 1e-6:
        screen_u = np.cross(screen_normal, np.array([1, 0, 0]))
    screen_u = screen_u / np.linalg.norm(screen_u)
    screen_v = np.cross(screen_normal, screen_u)

    corners = [
        screen_center - screen_u * screen_width / 2 - screen_v * screen_height / 2,
        screen_center - screen_u * screen_width / 2 + screen_v * screen_height / 2,
        screen_center + screen_u * screen_width / 2 + screen_v * screen_height / 2,
        screen_center + screen_u * screen_width / 2 - screen_v * screen_height / 2
    ]
    corners = np.array(corners + [corners[0]])  # Close the loop
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color='blue', label='Screen Plane')

    # Plot the screen normal
    ax.quiver(screen_center[0], screen_center[1], screen_center[2],
              screen_normal[0], screen_normal[1], screen_normal[2],
              color='cyan', length=100, label='Screen Normal')

    # Plot the ray
    ray_length = np.linalg.norm(intersection_point - ray_origin)
    t = np.linspace(0, ray_length, 100)
    ray_line = ray_origin + np.outer(t, ray_direction)
    ax.plot(ray_line[:, 0], ray_line[:, 1], ray_line[:, 2], color='red', label='Ray')

    # Plot the ray origin
    ax.scatter(*ray_origin, color='blue', label='Ray Origin', s=50)

    # Plot the intersection point
    if intersection_point is not None:
        ax.scatter(*intersection_point, color='black', label='Intersection Point', s=50)

    # Plot the camera position
    ax.scatter(*camera_position, color='red', label='Camera Position', s=50)
    ax.scatter(*screen_center, color='orange', label='Screen Center', s=50)

    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d((-400, 400))
    ax.set_ylim3d((-150, 450))
    ax.set_zlim3d((-1000, 200))
    ax.set_aspect('equal')
    ax.invert_zaxis()
    ax.legend()
    title = '3D Visualization of Screen, Ray, and Intersection' + '_' + str(view_el) + '_' + str(view_az)
    ax.set_title(title)
    ax.view_init(elev=view_el, azim=view_az)

    if show_fig: plt.show()
    return fig


def visualization(frame, face_normalized, is_blinking, camera_position, screen_center, 
                  screen_normal, screen_width, screen_height, ray_origin, ray_direction, intersection_point, font):
    if (is_blinking):
        cv2.putText(frame, "BLINKING", (50, 150), font, 5, (255, 0, 0))

    # cv2.imshow("frame", frame)
    cv2.imshow("normalized face", face_normalized)
    # show scene image
    fig_sence = visualize_scene(camera_position, screen_center, screen_normal, screen_width, screen_height, ray_origin,
                    ray_direction, intersection_point, 100, 270, show_fig=False)
    img_scene = fig_to_np_array(fig_sence)
    img_scene_bgr = cv2.cvtColor(img_scene, cv2.COLOR_RGB2BGR)
    cv2.imshow('Matplotlib Plot', img_scene_bgr)

def visualization_v2(frame, is_blinking, camera_position, screen_center, 
                  screen_normal, screen_width, screen_height, ray_origin, ray_direction, intersection_point, font):
    if (is_blinking):
        cv2.putText(frame, "BLINKING", (50, 150), font, 5, (255, 0, 0))

    cv2.imshow("frame", frame)
    # show scene image
    fig_sence = visualize_scene(camera_position, screen_center, screen_normal, screen_width, screen_height, ray_origin,
                    ray_direction, intersection_point, 100, 270, show_fig=False)
    img_scene = fig_to_np_array(fig_sence)
    img_scene_bgr = cv2.cvtColor(img_scene, cv2.COLOR_RGB2BGR)
    cv2.imshow('Matplotlib Plot', img_scene_bgr)