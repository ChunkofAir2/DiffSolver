# set constants so we don't differentiate against them 
# in the case of multivariable DiffEqs 

from .operators import Operator
class Constant(Operator):
    """Creates a constant value that can be used during numerical integration"""

    def __init__(self, name: str=None, init=None):
        super().__init__()
        self.attrname = name if name != None else "_constant_value"
        self.set_y(init)

    def set_y(self, y):
        self.__setattr__(self.attrname, y)
    
    def op(self, t, y):
        # return self.inner(t, getattr(self, self.attrname))
        return self.__getattribute__(self.attrname)
