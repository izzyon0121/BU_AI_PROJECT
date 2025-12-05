# config/settings.py
# image and preprocess defaults
DEFAULT_IMG_SIZE = 640
CONF_THRES = 0.25

# Adaptive preprocess defaults (middle values requested)
DEFAULT_BRIGHT_LOW = 85.0       # mean luminance threshold
DEFAULT_CONTRAST_LOW = 30.0     # stddev luminance threshold
DEFAULT_NOISE_HIGH = 22.0       # laplacian std threshold
DEFAULT_SHARPEN_ALPHA = 0.65    # sharpen strength
