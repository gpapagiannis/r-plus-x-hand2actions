from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import open3d as o3d
import numpy as np

def point_cloud_to_depth_image(point_cloud, fx, fy, cx, cy, width=1280, height=720):
    """
    Convert a 3D point cloud into a depth image.

    Parameters:
    - point_cloud: A 2D numpy array of shape (N, 3), containing the (X, Y, Z) coordinates of each point.
    - fx, fy: The focal lengths of the camera in pixels.
    - cx, cy: The optical center (principal point) of the camera, typically the image center.
    - width, height: The dimensions of the resulting depth image.

    Returns:
    - A 2D numpy array of shape (height, width), representing the depth image, where each pixel contains
      the depth value of the corresponding point in the point cloud. Pixels with no corresponding point
      are set to 0.
    """
    # Initialize the depth image with zeros (no depth)
    depth_image = np.zeros((height, width), dtype=np.float32)

    # Extract X, Y, Z coordinates
    X, Y, Z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]

    # Exclude points with no depth or behind the camera
    # valid = Z > 0
    # X = X[valid]
    # Y = Y[valid]
    # Z = Z[valid]

    # Project points onto the image plane
    u = (X * fx / Z) + cx
    v = (Y * fy / Z) + cy

    # Convert coordinates to integer pixel indices
    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    # Filter out points that fall outside the image
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid]
    v = v[valid]
    Z = Z[valid]

    # Update the depth image
    depth_image[v, u] = Z

    return depth_image


def depth_to_point_cloud(depth_im, fx, fy, cx, cy):
    rows, cols = depth_im.shape
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    xx, yy = np.meshgrid(x, y)
    X = (xx - cx) * depth_im / fx
    Y = (yy - cy) * depth_im / fy
    Z = depth_im
    return np.stack([X, Y, Z], axis=-1)


def find_scaled_transformation(p, q, use_scale=True):
    # p, q are Nx3 arrays of corresponding points
    # Calculate centroids
    Cp = np.mean(p, axis=0)
    Cq = np.mean(q, axis=0)
    # Center points around the centroids
    p_centered = p - Cp
    q_centered = q - Cq
    # Compute scale
    # Using the ratio of sums of distances from the centroids
    scale = np.sqrt((q_centered ** 2).sum() / (p_centered ** 2).sum())

    if not use_scale:
        scale = 1
    # Compute covariance matrix H
    H = np.dot(p_centered.T, q_centered)
    # SVD on covariance matrix
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # Ensure a proper rotation matrix (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # Compute translation
    t = Cq.T - scale * np.dot(R, Cp.T)
    # Construct the transformation matrix
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T


def apply_transformation(p, T):
    """
    Apply a 4x4 transformation matrix to a set of 3D points.

    Parameters:
    - p: A 2D numpy array of shape (N, 3), containing the (X, Y, Z) coordinates of each point.
    - T: The 4x4 transformation matrix.

    Returns:
    - A 2D numpy array of shape (N, 3), containing the transformed (X, Y, Z) coordinates.
    """
    # Homogenize the points in p
    p_homogeneous = np.hstack((p, np.ones((p.shape[0], 1))))
    # Apply the transformation matrix to each point
    p_transformed_homogeneous = np.dot(T, p_homogeneous.T).T
    # Dehomogenize the points
    p_transformed = p_transformed_homogeneous[:, :3] / p_transformed_homogeneous[:, 3, np.newaxis]
    return p_transformed


def compute_principal_axis(point_cloud, switch_principal_axis=False, return_eigenstuff=False):
    # Extract principal axis of the point cloud
    # Compute the covariance matrix
    cov = np.cov(point_cloud.T)
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    # Sort the eigenvectors by decreasing eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Extract the principal axis
    principal_axis = eigenvectors[:, 0]
    second_axis = eigenvectors[:, 1]
    # Make the vector longer by keep its direction
    principal_axis = principal_axis * 10
    second_axis = second_axis * 10
    if return_eigenstuff:
        return eigenvalues, eigenvectors
    if switch_principal_axis:
        # return second_axis, principal_axis
        return second_axis, eigenvectors[:, 2] * 10
    return principal_axis, second_axis


def scale_gripper_match_hand(gripper_pcd, hand_pcd, vizualize=False):
    """Scaled the gripper to match the hand 1-to-1
    This is approximate and may not need to be engineered; but that's okay.
    """
    gripper_bb = gripper_pcd.get_axis_aligned_bounding_box()
    hand_bb = hand_pcd.get_axis_aligned_bounding_box()
    # get bounding box volume
    volume_ratio = hand_bb.volume() / gripper_bb.volume()
    gripper = gripper_pcd.points
    hand = hand_pcd.points
    scale_x = (hand_bb.get_max_bound()[0] - hand_bb.get_min_bound()[0]) / (
                gripper_bb.get_max_bound()[0] - gripper_bb.get_min_bound()[0])
    scale_y = (hand_bb.get_max_bound()[1] - hand_bb.get_min_bound()[1]) / (
                gripper_bb.get_max_bound()[1] - gripper_bb.get_min_bound()[1])
    scale_z = (hand_bb.get_max_bound()[2] - hand_bb.get_min_bound()[2]) / (
                gripper_bb.get_max_bound()[2] - gripper_bb.get_min_bound()[2])
    center = gripper_pcd.get_center()
    mean_scaling = (scale_x + scale_y + scale_z) / 3
    # pcd_scaled = (gripper - center) * np.array([scale_x, scale_y, scale_z]) + center
    pcd_scaled = (gripper - center) * np.array([mean_scaling, mean_scaling, mean_scaling]) + center
    pcd_scaled_o3d = o3d.geometry.PointCloud()
    pcd_scaled_o3d.points = o3d.utility.Vector3dVector(pcd_scaled)

    if vizualize:
        o3d.visualization.draw_geometries([pcd_scaled_o3d, hand_pcd])

    return pcd_scaled_o3d


def np_to_o3d(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def interpolate_rotation(r1, r2, num_steps):
    r1 = R.from_matrix(r1)
    r2 = R.from_matrix(r2)
    # r1 and r2 in one Rotation instance
    key_rots = R.from_matrix([r1.as_matrix(), r2.as_matrix()])
    # interpolate with Slerp
    slerp = Slerp([0, 1], key_rots)
    times = np.linspace(0, 1, num_steps)
    qs = slerp(times)

    return qs


def interpolate_translation(t1, t2, num_steps):
    ts = np.zeros((num_steps, 3))
    for i in range(num_steps):
        ts[i] = t1 + (t2 - t1) * i / (num_steps - 1)
    return ts


def interpolate_pose(pose1, pose2, num_steps):
    r1 = pose1[:3, :3]
    t1 = pose1[:3, 3]
    r2 = pose2[:3, :3]
    t2 = pose2[:3, 3]
    rs = interpolate_rotation(r1, r2, num_steps)
    ts = interpolate_translation(t1, t2, num_steps)
    poses = np.zeros((num_steps, 4, 4))
    for i in range(num_steps):
        poses[i, :3, :3] = rs[i].as_matrix()  # R.from_quat(rs[i]).as_matrix()
        poses[i, :3, 3] = ts[i]
        poses[i, 3, 3] = 1
    return poses


def interpolate_pose_sequence(poses, num_steps):
    interpolated_poses = []
    for i in range(len(poses) - 1):
        interpolated_poses.append(interpolate_pose(poses[i], poses[i + 1], num_steps))
    return np.vstack(interpolated_poses)

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
