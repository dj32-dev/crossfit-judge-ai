import numpy as np

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points.
    a = First point (e.g., Hip)
    b = Vertex point (e.g., Knee)
    c = End point (e.g., Ankle)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle