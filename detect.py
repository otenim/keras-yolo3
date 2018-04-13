import os
import argparse
from yolo import YOLO
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('model', help='Path to a serialized keras model file.')
parser.add_argument('anchors', help='Path to an anchors file.')
parser.add_argument('classes', help='Path to a class file.')
parser.add_argument('input_img', help='Path to an input image.')
parser.add_argument('output_img', help='Path to the output image.')

def main(args):

    # instantiate YOLOv3 model
    yolo = YOLO(
        model_path=os.path.expanduser(args.model),
        anchors_path=os.path.expanduser(args.anchors),
        classes_path=os.path.expanduser(args.classes),
    )

    # load PIL image
    input_img = Image.open(os.path.expanduser(args.input_img))

    # detect bounding boxes
    output_img = yolo.detect_image(input_img)

    # save output PIL image
    output_img.save(os.path.expanduser(args.output_img))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
