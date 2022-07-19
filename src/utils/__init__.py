from .config import load_default_config
from .flow_warp import flow_warp
from .general import normalize_features, load_image, get_anno, evaluate, setup_logging, set_seed
from .mesh import load_off, save_off, forward_interpolate, pre_process_mesh_pascal, vertex_memory_to_face_memory, campos_to_R_T, center_crop_fun, get_n_list


CATEGORIES = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']
ON3D_CATEGORIES = ['bed', 'bookshelf', 'calculator', 'cellphone', 'computer', 'cabinet', 'guitar', 'iron', 'knife', 'microwave', 'pen', 'pot', 'rifle', 'slipper', 'stove', 'toilet', 'tub', 'wheelchair']
PIX3D_CATEGORIES = ['bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']

KP_LIST_DICT = {'aeroplane': ['left_wing', 'right_wing', 'rudder_upper', 'noselanding', 'left_elevator', 'rudder_lower', 'right_elevator', 'tail'],
                'bicycle': ['seat_front', 'right_back_wheel', 'right_pedal_center', 'right_front_wheel', 'left_front_wheel', 'left_handle', 'seat_back', 'head_center', 'left_back_wheel', 'left_pedal_center', 'right_handle'],
                'boat': ['head_down', 'head', 'tail_right', 'tail_left', 'head_right', 'tail', 'head_left'],
                'bottle': ['body', 'bottom_left', 'bottom', 'mouth', 'body_right', 'body_left', 'bottom_right'],
                'bus': ['body_front_left_lower', 'body_front_right_upper', 'body_back_right_lower', 'right_back_wheel', 'body_back_left_upper', 'right_front_wheel', 'left_front_wheel', 'body_front_left_upper', 'body_back_left_lower', 'body_back_right_upper', 'body_front_right_lower', 'left_back_wheel'],
                'car': ['left_front_wheel', 'left_back_wheel', 'right_front_wheel', 'right_back_wheel', 'upper_left_windshield', 'upper_right_windshield', 'upper_left_rearwindow', 'upper_right_rearwindow', 'left_front_light', 'right_front_light', 'left_back_trunk', 'right_back_trunk'],
                'chair': ['seat_upper_right', 'back_upper_left', 'seat_lower_right', 'leg_lower_left', 'back_upper_right', 'leg_upper_right', 'seat_lower_left', 'leg_upper_left', 'seat_upper_left', 'leg_lower_right'],
                'diningtable': ['top_lower_left', 'top_up', 'top_lower_right', 'leg_lower_left', 'leg_upper_right', 'top_right', 'top_left', 'leg_upper_left', 'top_upper_left', 'top_upper_right', 'top_down', 'leg_lower_right'],
                'motorbike': ['front_seat', 'right_back_wheel', 'back_seat', 'right_front_wheel', 'left_front_wheel', 'headlight_center', 'right_handle_center', 'left_handle_center', 'head_center', 'left_back_wheel'],
                'sofa': ['top_right_corner', 'seat_bottom_right', 'left_bottom_back', 'seat_bottom_left', 'front_bottom_right', 'top_left_corner', 'right_bottom_back', 'seat_top_left', 'front_bottom_left', 'seat_top_right'],
                'train': ['head_top', 'mid1_left_bottom', 'head_left_top', 'mid1_left_top', 'mid2_right_bottom', 'head_right_bottom', 'mid1_right_bottom', 'head_left_bottom', 'mid2_left_top', 'mid2_left_bottom', 'head_right_top', 'tail_right_top', 'tail_left_top', 'tail_right_bottom', 'tail_left_bottom', 'mid2_right_top', 'mid1_right_top'],
                'tvmonitor': ['back_top_left', 'back_bottom_right', 'front_bottom_right', 'front_top_right', 'front_top_left', 'back_bottom_left', 'back_top_right', 'front_bottom_left']}

MESH_FACE_BREAKS_1000 = {
    'aeroplane': [306, 612, 714, 816, 924, 1032],
    'bicycle': [138, 276, 354, 432, 731, 1030],
    'boat': [184, 368, 464, 560, 836, 1112],
    'bottle': [72, 144, 360, 576, 819, 1062],
    'bus': [232, 464, 536, 608, 869, 1130],
    'car': [240, 480, 560, 640, 832, 1024],
    'chair': [132, 264, 451, 638, 842, 1046],
    'diningtable': [234, 468, 648, 828, 958, 1088],
    'motorbike': [184, 368, 464, 560, 836, 1112],
    'sofa': [209, 418, 627, 836, 957, 1078],
    'train': [210, 420, 468, 516, 796, 1076],
    'tvmonitor': [187, 374, 578, 782, 914, 1046],
    'microwave': [256, 512, 656, 800, 944, 1088],
    'toilet': [180, 360, 516, 672, 867, 1062],
    'tub': [266, 532, 684, 836, 948, 1060],
    'wheelchair': [171, 342, 459, 576, 823, 1070]
}
