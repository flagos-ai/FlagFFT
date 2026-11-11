# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep invariant matrix operands outside a bounded loop over FFT batches."""
import ast


def batch_matrix_source(source, group):
    if type(group) is not int or group not in (4, 8, 16, 32):
        raise ValueError('Unsupported matrix batch group')
    tree = ast.parse(source)
    fn, = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if [arg.arg for arg in fn.args.args] != ['X', 'TW', 'Y']:
        raise ValueError('Matrix batch loop requires the ungrouped three-pointer ABI')
    dot = next(i for i, statement in enumerate(fn.body)
               if any(isinstance(node, ast.Call) and ast.unparse(node.func) == 'tl.dot'
                      for node in ast.walk(statement)))
    invariants, inputs = [], []
    dependent = {'X', 'batch'}
    for statement in fn.body[:dot]:
        names = [node for node in ast.walk(statement) if isinstance(node, ast.Name)]
        written = {node.id for node in names if isinstance(node.ctx, ast.Store)}
        read = {node.id for node in names if isinstance(node.ctx, ast.Load)}
        if written == {'batch'}:
            continue
        if read & dependent:
            inputs.append(statement)
            dependent |= written
        else:
            invariants.append(statement)
    loop = ast.parse(f'''batch_start=tl.program_id(1)*{group}
step=0
while step<tl.minimum({group},NB-batch_start):
    batch=(batch_start+step).to(tl.int64)
    step+=1
''').body
    loop[-1].body[1:1] = inputs + fn.body[dot:]
    fn.body = invariants + loop
    fn.args.args.append(ast.arg(arg='NB'))
    return ast.unparse(ast.fix_missing_locations(fn)) + '\n'
