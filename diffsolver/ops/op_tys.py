from .operators import Operator
import inspect

def _convert_double_to_combine(op):
    def rtn_op(*args):
        cumulative = args[0]
        for arg in args[1:]:
            cumulative = op(cumulative, arg)
        return cumulative
    return rtn_op

class CombineOp(Operator):
    def __init_subclass__(cls, op):
        super().__init__(cls)
        if len(inspect.signature(op).parameters) == 2:
            op = _convert_double_to_combine(op)
        cls.combine_op = lambda _, *args: op(*args)

    def __init__(self, *args):
        self.ops = [*args]

    def op(self, t, y):
        op_res = [op(t, y) for op in self.ops]
        return self.combine_op(*op_res)
    
class SingleOp(Operator):
    one: Operator

    def __init_subclass__(cls, op):
        super().__init__(cls)
        cls.single_op = lambda _, x: op(x)

    def __init__(self, one):
        self.one = one

    def op(self, t, y):
        one = self.one(t, y)
        return self.single_op(one)