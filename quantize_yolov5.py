import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append("..")
from onnxruntime.quantization import quantize_static, QuantType, CalibrationMethod, CalibrationDataReader
from utils.dataloaders import LoadImages
from utils.general import check_dataset
import numpy as np

def representative_dataset_gen(dataset, ncalib=100):
    # Representative dataset generator for use with converter.representative_dataset, returns a generator of np arrays
    def data_gen():
        for n, (path, img, im0s, vid_cap, string) in enumerate(dataset):
            input = np.transpose(img, [0, 1, 2])
            input = np.expand_dims(input, axis=0).astype(np.float32)
            input /= 255
            yield [input]
    return data_gen

class CalibrationDataGenYOLO(CalibrationDataReader):
    def __init__(self,
        calib_data_gen,
        input_name
    ):
        x_train = calib_data_gen
        self.calib_data = iter([{input_name: np.array(data[0])} for data in x_train()])

    def get_next(self):
        return next(self.calib_data, None)


dataset = LoadImages(check_dataset('./data/coco128.yaml')['train'], img_size=[640, 640], auto=False)
data_generator = representative_dataset_gen(dataset)

data_reader = CalibrationDataGenYOLO(
    calib_data_gen=data_generator,
    input_name='images'
)


model_path = 'yolov5s'
# Quantize the exported model
# quantize_static(
#     f'{model_path}.onnx',
#     f'{model_path}_ort_quant.onnx',
#     calibration_data_reader=data_reader,
#     activation_type=QuantType.QUInt8,
#     weight_type=QuantType.QInt8,
#     per_channel=True,
#     reduce_range=True,
#     calibrate_method=CalibrationMethod.MinMax
#         )


# quantize_static(
#     f'{model_path}.onnx',
#     f'{model_path}_ort_quant.u8s8.exclude.bigscale.onnx',
#     calibration_data_reader=data_reader,
#     activation_type=QuantType.QUInt8,
#     weight_type=QuantType.QInt8,
#     # nodes_to_exclude=['Mul_214', 'Mul_225', 'Mul_249', 'Mul_260', 'Mul_284', 'Mul_295', 'Concat_231', 'Concat_266', 'Concat_301', 'Concat_303'],
#     per_channel=True,
#     reduce_range=True,
#     calibrate_method=CalibrationMethod.MinMax
#         )

#6.1版本
quantize_static(
     f'{model_path}.onnx',
     f'{model_path}_ort_quant.u8s8.exclude.bigscale.onnx',
      calibration_data_reader=data_reader,
      activation_type=QuantType.QUInt8,
      weight_type=QuantType.QInt8,
      nodes_to_exclude=['Mul_221', 'Mul_227', 'Mul_255', 'Mul_261', 'Mul_289', 'Mul_295',
                        "Reshape_231","Reshape_265","Reshape_299",
                        
                        'Concat_228','Concat_262','Concat_296','Concat_230', 'Concat_264', 'Concat_298', 'Concat_300'],
      per_channel=False,
      reduce_range=True,
          )

#7.0 版本
# quantize_static(
#      f'{model_path}.onnx',
#      f'{model_path}_ort_quant.u8s8.exclude.bigscale.onnx',
#       calibration_data_reader=data_reader,
#       activation_type=QuantType.QUInt8,
#       weight_type=QuantType.QInt8,
#       nodes_to_exclude=['Mul_208', 'Mul_214', 'Mul_227', 'Mul_233', 'Mul_246', 'Mul_252',
#                         "Reshape_216","Reshape_235","Reshape_254",
                        
#                         'Concat_215','Concat_234','Concat_253','Concat_255'],
#       per_channel=False,
#       reduce_range=True,
#           )