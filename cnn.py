# cnn.py
# Created by abdularis on 24/04/18


from tsnn.models import Model
from tsnn.operations import Convolution, Relu, MaxPooling, Flatten, FullyConnected


def build_model():
    # model construction
    model = Model(input_shape=[None, 128, 128, 3], num_classes=6)
    model.use_name_scope('conv_1')
    model.add(Convolution(64, (3, 3)))
    model.add(Relu())

    model.use_name_scope('conv_2')
    model.add(Convolution(64, (3, 3)))
    model.add(Relu())
    model.add(MaxPooling())

    model.use_name_scope('conv_3')
    model.add(Convolution(64, (3, 3)))
    model.add(Relu())
    model.add(MaxPooling())

    model.use_name_scope('conv_4')
    model.add(Convolution(128, (3, 3)))
    model.add(Relu())
    model.add(MaxPooling())

    model.use_name_scope('conv_5')
    model.add(Convolution(128, (3, 3)))
    model.add(Relu())
    model.add(MaxPooling())

    model.use_name_scope('conv_6')
    model.add(Convolution(128, (3, 3)))
    model.add(Relu())
    model.add(MaxPooling())

    model.use_name_scope('flatten')
    model.add(Flatten())

    model.use_name_scope('fully_connected_1')
    model.add(FullyConnected(512))
    model.add(Relu())

    model.use_name_scope('fully_connected_2')
    model.add(FullyConnected(512))
    model.add(Relu())

    model.use_name_scope('fully_connected_3')
    model.add(FullyConnected(6))
    # end

    model.compile()
    return model
