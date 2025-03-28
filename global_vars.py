import numpy as np
import os

task_folder = "assets/grasp_fanta" # folder with the task
VIZ = False # visualize the action extraction/hand tracking process
IS_PRESS_TASK = False # extract robot action for press task
LOAD_SCENE_DATA_FOR_PROCESSING=False # load scene data after extracting hand poses to create videos etc. for vizualization
INTRINSICS_REAL_CAMERA = np.load("assets/utils/intrinsics_rgb_d455.npy") # intrinsics for camera used to example_task_1 (grasp fanta)
# INTRINSICS_REAL_CAMERA = np.load("assets/utils/head_cam_intrinsic_matrix_aligned_depth.npy") # intrinsics for camera used to example_task_2 (pick up phone)
VIZ_DEMO = False # vizualize video of the demonstration before processing
START = 0 # which part of the video to use start:end
END = 1000000
FRAME_STEP = 1 # use every nth frame
hands_rgb = np.load(f"{task_folder}/rgbs.npy")[START:END]
hands_depth = np.load(f"{task_folder}/depths.npy")[START:END]
MODEL_MANO_PATH = 'hamer/_DATA/_DATA/data/mano' # path to mano model
SCENE_FILES_FOLDER = f"{task_folder}/scene_files"
os.makedirs(SCENE_FILES_FOLDER, exist_ok=True) # make dir if not exists
INTRINSICS_HAMER_RENDERER = np.eye(4)
INTRINSICS_HAMER_RENDERER[0 ,0] = 2295.0
INTRINSICS_HAMER_RENDERER[1, 1] = 2295.0
INTRINSICS_HAMER_RENDERER[0, 2] = 320.0
INTRINSICS_HAMER_RENDERER[1, 2] = 240.0
# probably do not need to change
T_OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
HUMAN_HAND_COLOR=(0.999, 0.6745, 0.4117)
MANO_HAND_IDS = {"wrist": 0,        "index_mcp": 1,     "index_pip": 2, 
                 "index_dip": 3,    "middle_mcp": 4,    "middle_pip": 5, 
                 "middle_dip": 6,   "pinkie_mcp": 7,    "pinkie_pip": 8, 
                 "pinkie_dip": 9,   "ring_mcp": 10,     "ring_pip": 11, 
                 "ring_dip": 12,    "thumb_mcp": 13,    "thumb_pip": 14, 
                 "thumb_dip": 15,   "thumb_tip": 16,    "index_tip": 17, 
                 "middle_tip": 18,  "ring_tip": 19,     "pinky_tip": 20}
DISTANCE_BETWEEN_GRIPPERS_FINGERS = 0.08507

