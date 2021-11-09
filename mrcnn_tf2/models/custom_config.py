import os
import sys
from mrcnn_tf2.models.config import Config

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

########################
# Paths
########################
COCO_WEIGHTS_PATH = os.path.abspath('../data/models/mask_rcnn_coco.h5')
DEFAULT_LOGS_DIR = os.path.abspath('../data/logs')
DATASET_DIR = os.path.abspath('../data/dataset')
TRAINED_MODELS = os.path.abspath('../data/models')


class TrainingSnagsConfig(Config):
    """
    Custom class for tuning settings of mask-rcnn network
    """
    NAME = 'snag'
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 140
    VALIDATION_STEPS = 5
    RPN_ANCHOR_SCALES = (32, 52, 62, 92, 128)
    RPN_TRAIN_ANCHORS_PER_IMAGE = 320
    RPN_ANCHOR_STRIDE = 2
    MAX_GT_INSTANCES = 250
    TRAIN_ROIS_PER_IMAGE = 512
    DETECTION_MAX_INSTANCES = 250
    DETECTION_MIN_CONFIDENCE = 0.75
    DETECTION_NMS_THRESHOLD = 0.5
    RPN_NMS_THRESHOLD = 0.7
    POST_NMS_ROIS_TRAINING = 2048
    POST_NMS_ROIS_INFERENCE = 2048
    USE_MINI_MASK = True


class InferenceSnagsConfig(Config):
    """
    Custom class for tuning settings of mask-rcnn network
    """
    NAME = 'snag'
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 25
    DETECTION_MAX_INSTANCES = 250
    DETECTION_MIN_CONFIDENCE = 0.5
    RPN_ANCHOR_SCALES = (32, 52, 62, 92, 128)
    RPN_ANCHOR_STRIDE = 2