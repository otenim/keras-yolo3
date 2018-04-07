import utils
from keras.models import load_model

class YOLOv3(object):

    def __init__(self,
        model_path, anchors_path, classes_path):
        self.__classes = utils.load_classes(classes_path)
        self.__anchors = utils.load_anchors(anchors_path)
        self.__keras_model = load_model(model_path)

    @property
    def classes(self):
        return self.__classes

    @property
    def anchors(self):
        return self.__anchors
