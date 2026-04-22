import numpy as np

# calculate the intersection point of a ray and a plane
def ray_plane_intersection(ray_origin, ray_direction, screen_center, screen_normal):
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    denominator = np.dot(ray_direction, screen_normal)
    if np.abs(denominator) < 1e-6:  # Ray is parallel to the plane
        return None
    t = np.dot(screen_center - ray_origin, screen_normal) / denominator
    if t < 0:  # Intersection is behind the ray origin
        return None
    intersection_point = ray_origin + t * ray_direction
    return intersection_point

def point_to_screen_coordinate(intersection_point, screen_center, screen_normal, pixel_scale, w_screen, h_screen):
    screen_u = np.cross(screen_normal, np.array([0, 1, 0]))  # Screen's horizontal axis
    if np.linalg.norm(screen_u) < 1e-6:
        screen_u = np.cross(screen_normal, np.array([1, 0, 0]))
    screen_u = screen_u / np.linalg.norm(screen_u)
    screen_v = np.cross(screen_normal, screen_u)  # Screen's vertical axis

    relative_point = intersection_point - screen_center
    u_distance = np.dot(relative_point, screen_u)
    v_distance = np.dot(relative_point, screen_v)
    u_pixel = u_distance/pixel_scale
    v_pixel = v_distance/pixel_scale
    PoG_x_pred = u_pixel + w_screen/2
    PoG_y_pred = v_pixel + h_screen/2
    return PoG_x_pred, PoG_y_pred