# __init__.py.py
# Created by abdularis on 15/05/18
#
# This is my final year university project
# it's a wrapper around tensorflow API to make
# building sequential neural network model easier
#
# it has two module 'models' and 'operations'
# the 'models' contains a class representing a sequential
# neural network model
# and 'operations' contains a bunch of operation which can
# be added into a model, such as Convolution, Relu, MaxPooling etc.

from . import models
from . import operations
