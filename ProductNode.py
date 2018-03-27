import numpy as np 
from Node import Node
from Node import LeafNode

class ProductNode(Node):
    """
        Product node of SPN
    """
    def __init__(self,id,parent_id):
        super().__init__(id, parent_id)
        self.children=[]

    def add_child(self, child_id, weight=1):
        self.children.append(child_id)


    ###############################################################
    #                    soft gradient descent                    #
    ###############################################################  

    def backprop(self, spn):
        # unused parent, goes here when upward pass did not pass through here
        if self.logDerivative == Node.LOG_ZERO:
            print("unused parent product", self.id)
            return

        for child_id in self.children:
            #temp = self.logDerivative + self.logValue - child.logValue
            temp = self.logDerivative

            child = spn.get_node_by_id(child_id)
            # sum in log domain for product of children
            for node_id in self.children:
                val = spn.get_node_by_id(node_id).logValue
                if val != Node.LOG_ZERO:
                    temp += val
            # divide out current child
            if child.logValue != Node.LOG_ZERO:
                temp -= child.logValue
            # pass derivative
            child.logDerivative = np.logaddexp(temp, child.logDerivative)

            #print("passing derivative", np.exp(child.logDerivative), "from", self.id, "to", child.id)
            child.backprop(spn)


    def evaluate(self, spn):
        temp = 0.0
        for child_id in self.children:
            next_node = spn.get_node_by_id(child_id)
            value = next_node.evaluate(spn)
            
            # if one child is zero the product will be zero
            if value == Node.LOG_ZERO:
                self.logValue = value
                return Node.LOG_ZERO

            # multiply children in log domain
            temp += value
        self.logValue = temp
        return temp


    ###############################################################
    #                    hard gradient descent                    #
    ############################################################### 

    def hard_backprop(self, spn):
        # same as SPN
        # unused parent, upward pass did not pass through here
        if self.logValue == Node.LOG_ZERO:
            print("unused parent", self.id)
            return

        for child_id in self.children:

            temp = self.logDerivative
            child = spn.get_node_by_id(child_id)

            # sum in log domain for product of children
            for node_id in self.children:
                    temp += spn.get_node_by_id(child_id).logValue
            
            # divide out current child
            if child.logValue != Node.LOG_ZERO:
                temp -= child.logValue

            # pass derivative
            child.logDerivative = np.logaddexp(temp, child.logDerivative)

            child.hard_backprop(spn)


    def evaluate_mpn(self, spn):
        # same as SPN
        temp = 0.0
        for child_id in self.children:
            next_node = spn.get_node_by_id(child_id)
            value = next_node.evaluate_mpn(spn)
            
            # if one child is zero the product will be zero
            if value == Node.LOG_ZERO:
                self.logValue = value
                return Node.LOG_ZERO

            # multiply children in log domain
            temp += value
        self.logValue = temp
        return temp
