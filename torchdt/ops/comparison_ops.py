import torch
from torchdt.autograd import DTFunction, DTNonDifferentiableFunction
from torchdt.ops import register_base_op

class DTGeFunction(DTNonDifferentiableFunction):

    @staticmethod
    def forward(ops, x, y):
        return ops.ge(x, y)

class DTGtFunction(DTNonDifferentiableFunction):

    @staticmethod
    def forward(ops, x, y):
        return ops.gt(x, y)

class DTLeFunction(DTNonDifferentiableFunction):

    @staticmethod
    def forward(ops, x, y):
        return ops.le(x, y)

class DTLtFunction(DTNonDifferentiableFunction):

    @staticmethod
    def forward(ops, x, y):
        return ops.lt(x, y)