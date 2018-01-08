#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on server."""
# This code is based on https://github.com/allanzelener/YAD2K/blob/master/test_yolo.py
from typing import Dict, Any, List
import argparse
import os
import io
import json
import sys
import datetime
import numpy as np
from keras import backend as kback
from keras.models import load_model
from PIL import Image
from http.server import BaseHTTPRequestHandler, HTTPServer

from YAD2K.yad2k.models.keras_yolo import yolo_eval, yolo_head

parser = argparse.ArgumentParser(
    description='Run a YOLO_v2 style detection model on server')
parser.add_argument(
    'model_path',
    help='path to h5 model file containing body'
    'of a YOLO_v2 model')
parser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default='YAD2K/model_data/yolo_anchors.txt')
parser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to coco_classes.txt',
    default='YAD2K/model_data/coco_classes.txt')
parser.add_argument(
    '-s',
    '--score_threshold',
    type=float,
    help='threshold for bounding box scores, default .3',
    default=.3)
parser.add_argument(
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .5',
    default=.5)
parser.add_argument(
    '-p',
    '--port',
    type=int,
    help='http port, default 8000',
    default=8000)


def _main() -> None:
    model_path = os.path.expanduser(args.model_path)
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)

    sess = kback.get_session()  # TODO: Remove dependence on Tensorflow session.

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    with open(anchors_path) as f:
        anchors_s = f.readline()
        anchors = [float(x) for x in anchors_s.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    yolo_model = load_model(model_path)

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # TODO: Assumes dim ordering is channel last
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    print('{} model, anchors, and classes loaded.'.format(model_path))

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)

    _start_server({
        'sess': sess,
        'class_names': class_names,
        'anchors': anchors,
        'yolo_model': yolo_model,
        'model_image_size': model_image_size,
        'is_fixed_size': is_fixed_size,
    })


def _detect(setting: Dict[Any, Any], image) -> List[Dict[Any, Any]]:
    start = datetime.datetime.now()
    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(setting['yolo_model'].output, setting['anchors'], len(setting['class_names']))
    input_image_shape = kback.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold)
    print('spent time(yolo_eval)', datetime.datetime.now() - start)

    if setting['is_fixed_size']:  # TODO: When resizing we can use minibatch input.
        resized_image = image.resize(
            tuple(reversed(setting['model_image_size'])), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
    else:
        # Due to skip connection + max pooling in YOLO_v2, inputs must have
        # width and height as multiples of 32.
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        resized_image = image.resize(new_image_size, Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
        print(image_data.shape)

    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    print('spent time(before run)', datetime.datetime.now() - start)
    out_boxes, out_scores, out_classes = setting['sess'].run(
        [boxes, scores, classes],
        feed_dict={
            setting['yolo_model'].input: image_data,
            input_image_shape: [image.size[1], image.size[0]],
            kback.learning_phase(): 0
        })
    print('spent time(after run)', datetime.datetime.now() - start)

    detected_items = []
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = setting['class_names'][c]
        box = out_boxes[i]
        score = out_scores[i]
        detected_items.append({
            'class': predicted_class,
            'box': box.tolist(),
            'score': score.tolist(),
        })
    print('detected_items', detected_items)
    print('spent time(_detect)', datetime.datetime.now() - start)
    return detected_items


def make_handler_class(setting: Dict[Any, Any]) -> Any:
    class Handler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self._setting = setting
            super(Handler, self).__init__(*args, **kwargs)

        def set_cors(self):
            self.send_header('access-control-allow-origin', '*')
            allow_headers = self.headers.get('access-control-request-headers', '*')
            self.send_header('access-control-allow-headers', allow_headers)
            self.send_header('access-control-allow-methods', 'POST, OPTIONS')

        def do_OPTIONS(self):  # noqa: N802
            self.send_response(200)
            self.set_cors()

        def do_POST(self):  # noqa: N802
            print('post')
            data_string = self.rfile.read(int(self.headers['Content-Length']))
            buff = io.BytesIO(data_string)
            buff.seek(0)
            img = Image.open(buff).convert('RGB')
            detected_items = _detect(self._setting, img)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.set_cors()
            self.end_headers()
            self.wfile.write(json.dumps(detected_items).encode())
            return
    return Handler


def _start_server(setting: Dict[Any, Any]) -> None:
    server_address = ('', args.port)
    handler_cls = make_handler_class(setting)
    httpd = HTTPServer(server_address, handler_cls)
    httpd.serve_forever()
    print('httpd running...')
    sys.stdout.flush()


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    print(args)
    sys.stdout.flush()
    _main()
