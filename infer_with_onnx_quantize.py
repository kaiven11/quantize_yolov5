import base64
import cv2
import onnxruntime
import numpy as np

LABEL_NAME = 'class'


class PyModel(object):
    def __init__(self):
        self.classname = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
        self.load()

    def load(self):
        """
        模型加载
        """
        self.img_size = 640
        self.model_path = "yolov5s_ort_quant.u8s8.exclude.bigscale.onnx"
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.model = onnxruntime.InferenceSession(self.model_path, options, providers=['CPUExecutionProvider'])
        self.input_name = self.model.get_inputs()[0].name

    def xywh2xyxy(self, x):
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def xyxy2xyxy(self, x):
        x[:, 0] = x[:, 0] / self.r_w
        x[:, 1] = x[:, 1] / self.r_h
        x[:, 2] = x[:, 2] / self.r_w
        x[:, 3] = x[:, 3] / self.r_h
        return x

    def nms_cpu(self, boxes, confs, classes, nms_thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = confs.flatten().argsort()[::-1]
        keep = []
        while order.size > 0:
            idx_self = order[0]
            keep.append(idx_self)
            xx1 = np.maximum(x1[idx_self], x1[order[1:]])
            yy1 = np.maximum(y1[idx_self], y1[order[1:]])
            xx2 = np.minimum(x2[idx_self], x2[order[1:]])
            yy2 = np.minimum(y2[idx_self], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            over = inter / (areas[idx_self] + areas[order[1:]] - inter)
            inds = np.where(over <= nms_thresh)[0]
            order = order[inds + 1]
        boxes = np.concatenate((boxes[keep], confs[keep]), axis=1)
        classes = classes[keep].reshape((-1, 1))
        _boxes = np.concatenate((boxes, classes), axis=1)
        _boxes = self.xyxy2xyxy(_boxes)
        return _boxes

    def preprocess_image(self, image_raw):
        h, w, c = image_raw.shape
        self.r_w = self.img_size / w
        self.r_h = self.img_size / h
        img = cv2.resize(image_raw, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        img /= 255.0
        img = np.ascontiguousarray(img)
        return img

    def post_processing(self, conf_thresh, nms_thresh, outputs):
        xc = outputs[..., 4] > conf_thresh
        for xi, x in enumerate(outputs):
            x = x[xc[xi]]
            boxes = self.xywh2xyxy(x[:, :4])
            confs = np.expand_dims(x[:, 4], axis=1)
            classes = np.argmax(x[:, 5:], axis=-1)
            return self.nms_cpu(boxes, confs, classes, nms_thresh)

    def transform(self, data):
        return_image = data.get('return_image', True)
        pre_decode = base64.b64decode(data['data'])
        image_from_buffer = np.frombuffer(pre_decode, dtype=np.uint8)
        image_arr = cv2.imdecode(image_from_buffer, cv2.IMREAD_COLOR)
        image = self.preprocess_image(image_arr)
        output = self.model.run(None, {self.input_name: image})[0]
        prediction = self.post_processing(0.25, 0.45, output).tolist()
        ans = [[cla[i] for i in range(4)] + [self.classname[int(cla[-1])]] for cla in prediction if len(cla) != 0]
        return ans


if __name__ == '__main__':
    import time
    hat_model = PyModel()
    frame = cv2.imread('bus.jpg')

    img_str = cv2.imencode('.jpg', frame)[1].tobytes()
    base64_str = base64.b64encode(img_str)

    # print(
    #     {
    #         'data': base64_str,
    #         'return_image': True
    #     }
    # )
    t1 = time.time()
    boxes = hat_model.transform({
            'data': base64_str,
            'return_image': True
        })
    print("all time:",time.time()-t1)
    print(boxes)
