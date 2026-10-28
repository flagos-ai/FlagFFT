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


def fused_codelet_arithmetic(source):
    """Contract floating arithmetic in codelets; preserve all FFT index code."""
    class Fuse(ast.NodeTransformer):
        def visit_BinOp(self,node):
            node=self.generic_visit(node)
            if not isinstance(node.op,(ast.Add,ast.Sub)):
                return node
            if isinstance(node.left,ast.BinOp) and isinstance(node.left.op,ast.Mult):
                a,b=node.left.left,node.left.right
                c=node.right if isinstance(node.op,ast.Add) else ast.UnaryOp(ast.USub(),node.right)
            elif isinstance(node.right,ast.BinOp) and isinstance(node.right.op,ast.Mult):
                a,b=node.right.left,node.right.right;c=node.left
                if isinstance(node.op,ast.Sub):a=ast.UnaryOp(ast.USub(),a)
            else:
                return node
            return ast.copy_location(ast.Call(ast.Attribute(ast.Name('tl',ast.Load()),'fma',ast.Load()),[a,b,c],[]),node)
    tree=ast.parse(source)
    for fn in tree.body:
        if isinstance(fn,ast.FunctionDef) and (fn.name.startswith(('_fwd_rad','_inv_rad')) or fn.name=='_cmul'):
            Fuse().visit(fn)
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'
