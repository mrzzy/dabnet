#
# src/pose/api.py
# dabnet posenet
#

import os
# API constants
SERVER_HOST = os.environ.get("POSENET_HOST", "localhost")
SERVER_PORT = 8088
FEATURES_ROUTE = "/api/features"
ANNOTATION_ROUTE = "/api/annotate"

# Feature keys
POSE_SCORE_FEATURE = "pose.pose_score"
KEYPOINT_SCORE_FEATURE = "pose.keypoint.scores"
KEYPOINT_POINTS_FEATURE = "pose.keypoint.points"
