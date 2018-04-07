import os

def load_classes(filepath):
    filepath = os.path.expanduser(filepath)
    with open(filepath, 'r') as f:
        class_names = f.readlines()
        class_names = [class_name.strip() for class_name in class_names]
    return class_names

def load_anchors(filepath):
    filepath = os.path.expanduser(filepath)
    with open(filepath, 'r') as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors, dtype=np.float32).reshape(-1,2)
    return anchors
