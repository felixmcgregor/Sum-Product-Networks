import numpy as np


class Node():

    LOG_ZERO = float('-inf')

    def __init__(self, id, parent_id, children=[]):
        self.id = id
        self.children = children
        self.parent_id = parent_id
        self.logValue = Node.LOG_ZERO
        self.logDerivative = Node.LOG_ZERO

    def setLogValue(self, logValue):
        self.logValue = logValue


class LeafNode(Node):

    def __init__(self, id, parent_id, order, inverse, value=1):
        self.id = id
        self.parent_id = parent_id
        self.value = 1
        self.logValue = 0
        self.logDerivative = Node.LOG_ZERO
        self.order = order  # for feeding in evidence
        self.inverse = inverse

    def setValue(self, value):
        if value == 0:
            self.logValue = Node.LOG_ZERO
        else:
            self.logValue = np.log(value)
        #self.logValue = value

    def evaluate(self, spn):
        return self.logValue

    def evaluate_mpn(self, spn):
        return self.logValue

    def hard_backprop(self, spn):
        # do nothing for leaf node
        pass

    def backprop(self, spn):
        # do nothing for leaf node
        pass
