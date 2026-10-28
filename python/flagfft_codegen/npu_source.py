"""Source canonicalization for integer index arithmetic on CANN."""
import ast
import copy


def vector_remainders(source):
    """Express constant integer remainder with the existing vector quotient.

    Both Triton signed division and remainder truncate toward zero. CANN 9
    vectorizes the quotient but lowers signed remainder to a scalar loop.
    FFT index expressions are integral; floating remainder is not accepted by
    this helper's caller. Multiplication/subtraction preserve integer width.
    """
    class Rewrite(ast.NodeTransformer):
        def visit_BinOp(self,node):
            node=self.generic_visit(node)
            if (isinstance(node.op,ast.Mod) and isinstance(node.right,ast.Constant)
                    and type(node.right.value) is int and node.right.value>0):
                quotient=ast.BinOp(copy.deepcopy(node.left),ast.FloorDiv(),copy.deepcopy(node.right))
                return ast.copy_location(ast.BinOp(node.left,ast.Sub(),
                    ast.BinOp(quotient,ast.Mult(),node.right)),node)
            return node
    tree=Rewrite().visit(ast.parse(source))
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'
