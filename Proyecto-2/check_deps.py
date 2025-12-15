import sys

required = ['tensorflow', 'cv2', 'sklearn', 'matplotlib', 'skimage']
missing = []

for lib in required:
    try:
        if lib == 'cv2':
            import cv2
        elif lib == 'sklearn':
            import sklearn
        elif lib == 'skimage':
            import skimage
        else:
            __import__(lib)
        print(f"{lib}: OK")
    except ImportError as e:
        print(f"{lib}: MISSING ({e})")
        missing.append(lib)

if missing:
    sys.exit(1)
else:
    sys.exit(0)
