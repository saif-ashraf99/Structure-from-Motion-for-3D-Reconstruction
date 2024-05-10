import cv2
import numpy as np
import plotly.graph_objects as go

def extract_features_and_keypoints(image):
    """ Use SIFT to extract features and keypoints. """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None:
        descriptors = []
    return keypoints, descriptors

def match_features(desc1, desc2):
    """ Match features between two sets of descriptors using FLANN matcher. """
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def find_camera_parameters(matches, kp1, kp2, camera_matrix):
    """ Compute the essential matrix and retrieve the relative camera positions """
    if not matches:
        return None, None  # Handle no matches case
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(points1, points2, camera_matrix, cv2.RANSAC, 0.999, 1.0)
    if E is None:
        return None, None  # Handle failed essential matrix calculation
    _, R, t, mask = cv2.recoverPose(E, points1, points2, camera_matrix)
    return R, t

def triangulate_points(matches, kp1, kp2, R, t, camera_matrix):
    """ Triangulate points from the matched keypoints """
    if not matches or R is None or t is None:
        return np.array([])  # Handle no matches or failed pose recovery
    proj_matrix1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    proj_matrix2 = np.hstack((R, t))
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    points_4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, points1, points2)
    points_3d = points_4d[:3, :] / np.tile(points_4d[3, :], (3, 1))
    return points_3d.T

def run_sfm(image_paths):
    """ Main SfM function to process images and extract 3D points. """
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
    if any(img is None for img in images):
        return "Error: One or more images failed to load, check the file paths."

    camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])  # Placeholder camera matrix
    all_3d_points = []
    for i in range(len(images) - 1):
        kp1, desc1 = extract_features_and_keypoints(images[i])
        kp2, desc2 = extract_features_and_keypoints(images[i+1])
        matches = match_features(desc1, desc2)
        R, t = find_camera_parameters(matches, kp1, kp2, camera_matrix)
        if R is not None and t is not None:
            points_3d = triangulate_points(matches, kp1, kp2, R, t, camera_matrix)
            all_3d_points.extend(points_3d)

    if not all_3d_points:
        return "No 3D points were reconstructed, possibly due to poor image quality or lack of matches."

    plot_3d_points(np.array(all_3d_points))  # Visualize the points
    return all_3d_points

def plot_3d_points(points_3d):
    """ Visualize 3D points using Plotly """
    fig = go.Figure(data=[go.Scatter3d(
        x=points_3d[:, 0],
        y=points_3d[:, 1],
        z=points_3d[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=points_3d[:, 2],  # Color by z-axis values
            colorscale='Viridis',  # Color scale
            opacity=0.8
        )
    )])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
