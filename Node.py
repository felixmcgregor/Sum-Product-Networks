import numpy as np 

class Node():

    LOG_ZERO = float('-inf')

    def __init__(self,id,parent_id,children=[]):
        self.id = id
        self.children = children
        self.parent_id = parent_id
        self.logValue = Node.LOG_ZERO
        self.logDerivative = Node.LOG_ZERO

    def setLogValue(logValue):
        self.value = value


class LeafNode(Node):

    def __init__(self,id,parent_id,order,inverse,value=1):
        self.id = id
        self.parent_id = parent_id
        self.value = 1
        self.logValue = 0
        self.logDerivative = Node.LOG_ZERO
        self.order = order # for feeding in evidence
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

class ProductNode(Node):
    """
        Product node of SPN
    """
    def __init__(self,id,parent_id):
        super().__init__(id, parent_id)
        self.children=[]

    def add_child(self, child_id, weight=1):
        self.children.append(child_id)


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


    def evaluate_mpn(self, spn):
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


    def backprop(self, spn):
        # unused parent, goes here when upward pass did not pass through here
        if self.logDerivative == Node.LOG_ZERO:
            print("unused parent", self.id)
            return

        for child_id in self.children:

            temp = self.logDerivative

            child = spn.get_node_by_id(child_id)
            
            # dont do for leaf node
            if isinstance(child,LeafNode):
                continue

            # sum in log domain for product of children
            if child.logValue == Node.LOG_ZERO:
                for node_id in self.children:
                    temp += spn.get_node_by_id(child_id).logValue
            else:
                for node_id in self.children:
                    temp += spn.get_node_by_id(child_id).logValue
                temp -= child.logValue
            #print("temp prod", temp)

            if child.logDerivative == Node.LOG_ZERO:
                child.logDerivative = temp
            else:
                child.logDerivative = np.logaddexp(temp, child.logDerivative)

            child.backprop(spn)
            #print("Product, child id", child_id,"exp derivative", np.exp(child.logDerivative))


    def hard_backprop(self, spn):
        # unused parent, goes here when upward pass did not pass through here
        if self.logValue == Node.LOG_ZERO:
            print("unused parent", self.id)
            return

        for child_id in self.children:
            temp = 0.0
            child = spn.get_node_by_id(child_id)
            # dont do for leaf node
            if isinstance(child,LeafNode):
                continue
            if child.logValue == Node.LOG_ZERO:
                for node_id in self.children:
                    temp += spn.get_node_by_id(child_id).logValue
                #temp -= child.logValue

            else:
                # shortcut by just dividing out current child
                #temp = self.logDerivative + self.logValue - child.logValue
                for node_id in self.children:
                    temp += spn.get_node_by_id(child_id).logValue
                temp -= child.logValue


            if child.logDerivative == Node.LOG_ZERO:
                child.logDerivative = temp
            else:
                child.logDerivative = np.logaddexp(temp, child.logDerivative)

            child.hard_backprop(spn)
            #print(child_id, np.exp(child.logValue))




class SumNode(Node):
    """
        Sum node of SPN
    """
    def __init__(self,id, parent_id):
        super().__init__(id, parent_id)
        self.children = dict()

    def add_child(self, child_id, weight):
        self.children.update({child_id: [weight, 0]})


    def evaluate(self, spn):

        weighted_children = Node.LOG_ZERO
        for child_id in self.children.keys():


            # taking the log of the weight + log val
            weighted_child = spn.get_node_by_id(child_id).evaluate(spn) + np.log(self.children[child_id][0])
            if weighted_child == Node.LOG_ZERO:
                continue
            weighted_children = np.logaddexp(weighted_children, weighted_child)
        self.logValue = weighted_children
        return weighted_children
        

    def evaluate_mpn(self, spn):

        max_child = Node.LOG_ZERO
        for child_id in self.children.keys():
            child = spn.get_node_by_id(child_id).evaluate(spn) + np.log(self.children[child_id][0])
            if child > max_child:
                max_child = child
        self.logValue = max_child
        return max_child


    def backprop(self, spn):
        # unused parent, dont bother passing derivative
        if self.logDerivative == Node.LOG_ZERO:
            return
 
        for child_id in self.children.keys():
            #if self.id == 0:
            #    print("Root deriv", self.logDerivative, self.children[child_id][1])
            child = spn.get_node_by_id(child_id)
            #if self.id == 3:
            #    print("First sum deriv", self.logDerivative, self.children[child_id][1])
            #child = spn.get_node_by_id(child_id)
            
            # parent derivative times weight
            temp = self.logDerivative + np.log(self.children[child_id][0])

            if child.logDerivative == Node.LOG_ZERO:
                child.logDerivative = temp
            else:
                child.logDerivative = np.logaddexp(temp, child.logDerivative)
            child.backprop(spn)
            #print(child_id, np.exp(child.logDerivative))
            #print("Log Value", child.logValue)


            #update = np.exp(child.logValue + self.logDerivative)
            #if self.id == 0:
            #    update = np.exp(self.logValue)
            #else:
            #print(child.logValue)
            if child.logValue == Node.LOG_ZERO:
                update = 0
            else:
                update = child.logValue + self.logDerivative
                #print(child_id)
                #print("ds_dw child val", np.exp(child.logValue), "parent derivative", np.exp(self.logDerivative))
            
            #print("id", child_id,"update",update)
            # ds_dw
            self.children[child_id][1] = np.logaddexp(self.children[child_id][1], update)


    def hard_backprop(self, spn):
        # unused parent, goes here when upward pass did not pass through here
        if self.logValue == Node.LOG_ZERO:
            return
 
        for child_id in self.children.keys():
            child = spn.get_node_by_id(child_id)
            
            # NOT TRUE MUST ALSO DO FOR LEAVES
            # dont do for leaf node
            #if isinstance(child,LeafNode):
            #    continue

            if child.logValue == Node.LOG_ZERO:
                continue
            else:
                # counts
                self.children[child_id][1] += 1
            child.hard_backprop(spn)

            
    def map_inference(self):
        # what is the probability x3 = 0
        # numerator joint p(x1,x2,y)
        evidence = np.array([0.1,0.1,0.0])
        for leaf in self.get_leaves():
            var_value = evidence[leaf.order]
            
            # for not x1 situations
            if leaf.inverse:
                var_value = 1 - var_value
            leaf.setValue(var_value)

        root = self.get_root()
        num_value = root.evaluate(self)


        # denominator p(x1,x2)
        evidence = np.array([0.1,0.1])
        for leaf in self.get_leaves():
            if leaf.order < len(evidence):
                var_value = evidence[leaf.order]
            
                # for not x1 situations
                if leaf.inverse:
                    var_value = 1 - var_value
                leaf.setValue(var_value)
            else:
                leaf.setValue(1.0)

        denom_value = root.evaluate(self)
        print("parts")
        print(num_value)
        print(denom_value)
        print(num_value - denom_value)
        conditional = num_value - denom_value
        return np.exp(conditional)

