import cv2
import numpy as np
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray

def blur_score(frame):
    gray = preprocess(frame)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def blood_ratio(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask = (
        cv2.inRange(hsv, lower_red1, upper_red1)
        + cv2.inRange(hsv, lower_red2, upper_red2)
    )

    return cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])


def smoke_score(frame):
    gray = preprocess(frame)
    return np.std(gray)

def normalize(value, min_val, max_val):
    value = max(min_val, min(value, max_val))
    return (value - min_val) / (max_val - min_val)


def blur_confidence(blur):
    return normalize(blur, 20, 120)


def blood_confidence(blood):
    return 1 - normalize(blood, 0.0, 0.4)


def smoke_confidence(smoke):
    return normalize(smoke, 10, 40)


def visibility_confidence(blur, blood, smoke):
    conf = (
        0.4 * blur_confidence(blur) +
        0.3 * blood_confidence(blood) +
        0.3 * smoke_confidence(smoke)
    )
    return int(conf * 100)

def assess_visibility(frame, profile):
    """
    profile expects:
    profile["blur_th"]
    profile["blood_th"]
    profile["smoke_th"]
    """

    blur = blur_score(frame)
    blood = blood_ratio(frame)
    smoke = smoke_score(frame)

    reasons = []

    if blur < profile["blur_th"]:
        reasons.append("BLUR")
    if blood > profile["blood_th"]:
        reasons.append("BLOOD")
    if smoke < profile["smoke_th"]:
        reasons.append("SMOKE")
    if (
        blur < profile["blur_th"] * 0.7 or
        blood > profile["blood_th"] * 1.5 or
        smoke < profile["smoke_th"] * 0.7
    ):
        visibility = "POOR"

    elif (
        blur < profile["blur_th"] or
        blood > profile["blood_th"] or
        smoke < profile["smoke_th"]
    ):
        visibility = "MODERATE"

    else:
        visibility = "GOOD"

    confidence = visibility_confidence(blur, blood, smoke)

    return visibility, reasons, confidence
