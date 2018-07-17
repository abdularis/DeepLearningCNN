# model_client.py
# Created by abdularis on 16/07/18

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import tensorflow as tf
import numpy as np


def inference(images):
    channel = implementations.insecure_channel('localhost', 9000)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'cnnarch1_1_do3'
    request.model_spec.signature_name = 'predict_images'
    request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(images))

    result = stub.Predict(request, 10.)

    return np.array(result.outputs['probs'].float_val), np.array(result.outputs['features'].float_val)
