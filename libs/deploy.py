# -*- coding: utf-8 -*-  
# =================================================

import json
import requests
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import numpy as np


class PredictModelGrpc(object):
    def __init__(self, model_name, input_name, output_name, socket='0.0.0.0:8500'):
        self.socket = socket
        self.model_name = model_name
        self.input_name = input_name
        self.output_name = output_name
        self.request, self.stub = self.__get_request()

    def __get_request(self):
        channel = grpc.insecure_channel(self.socket, options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                                                              ('grpc.max_receive_message_length',
                                                               1024 * 1024 * 1024)])  # 可设置大小
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()

        request.model_spec.name = self.model_name
        request.model_spec.signature_name = "serving_default"

        return request, stub

    def inference(self, frames):
        self.request.inputs[self.input_name].CopyFrom(tf.make_tensor_proto(frames, dtype=tf.float32))  # images is input of model
        result = self.stub.Predict.future(self.request, 10.0)
        res = tf.make_ndarray(result.result().outputs[self.output_name])[0]
        return res


class PredictModelRESTAPI(object):
    def __init__(self, model_name, input_name, output_name, socket='localhost:8500'):
        self.socket = socket
        self.model_name = model_name
        self.input_name = input_name
        self.output_name = output_name
        self.url = self._get_url()

    def _get_url(self):
        url = "http://{}/v1/models/{}:predict".format(self.socket, self.model_name)
        return url

    def inference(self, data):
        payload = {
            "instances": [{self.input_name: data.tolist()}]
        }
        r = requests.post(self.url, json=payload)
        pred = json.loads(r.content.decode('utf-8'))
        pred = np.array(pred['predictions'][0])

        return pred