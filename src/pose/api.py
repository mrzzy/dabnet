#
# src/pose/api.py
# dabnet posenet
#

# API constants
SERVER_PORT = 8088
FEATURES_ROUTE = "/api/features"
ANNOTATION_ROUTE = "/api/annotate"

# Feature keys
POSE_SCORE_FEATURE = "pose.pose_score"
KEYPOINT_SCORE_FEATURE = "pose.keypoint.scores"
KEYPOINT_POINTS_FEATURE = "pose.keypoint.points"
