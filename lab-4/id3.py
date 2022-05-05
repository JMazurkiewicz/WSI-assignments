# Author: Jakub Mazurkiewicz
class _NodeBase:
    def __init__(self):
        pass

class _Node(_NodeBase):
    pass

class _Leaf(_NodeBase):
    pass

class Id3:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        print('test')

    def train(self, training_set: )
