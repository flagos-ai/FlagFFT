# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fuse a 13-by-7 late FFT pair after a dense aligned prefix.

Move the deferred phase to the existing prefix store. The merged constant
real basis then replaces two vector passes. Its following pass uses the
merged frequency digit in the twiddle, with no layout or workspace change.
"""
import ast
import copy


def _replace_function(source,name,body):
    node=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name==name)
    lines=source.splitlines(True)
    start=min([node.lineno]+[x.lineno for x in node.decorator_list])-1
    return ''.join(lines[:start])+body+'\n'+''.join(lines[node.end_lineno:])


def prephase(source, name, table_words, physical):
    node=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name==name)
    new=[];i=0;count=0
    def is_store(x):
        return isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and ast.unparse(x.value.func)=='tl.store'
    while i<len(node.body):
        a=node.body[i]
        if not is_store(a):
            new.append(a);i+=1;continue
        b=node.body[i+1];assert is_store(b)
        ca,cb=a.value,b.value
        oa,ob=ca.args[0].right,cb.args[0].right
        assert ast.dump(oa)==ast.dump(ob),(ast.unparse(a),ast.unparse(b))
        off=ast.unparse(oa)
        text=f'''post_r={ast.unparse(ca.args[1])}
post_i={ast.unparse(cb.args[1])}
post_c=tl.load(TW+{table_words}+({off}),({off})<{physical},0.)
post_s=tl.load(TW+{table_words+physical}+({off}),({off})<{physical},0.)
post_re=post_r*post_c-post_i*post_s
post_im=post_r*post_s+post_i*post_c
'''
        new.extend(ast.parse(text).body)
        ca.args[1]=ast.Name(id='post_re',ctx=ast.Load());cb.args[1]=ast.Name(id='post_im',ctx=ast.Load())
        new.extend((a,b));i+=2;count+=1
    assert count
    node.body=new
    return _replace_function(source,name,ast.unparse(ast.fix_missing_locations(node))+'\n')



def replace_late_pair(source,specs,batch_reuse=1):
    if type(batch_reuse) is not int or batch_reuse not in (1,8):
        raise ValueError('Unsupported late matrix batch group')
    specs=copy.deepcopy(specs)
    if (len(specs)<4 or specs[0].get('dense_prefix')!=(255,256,64,64,2)
            or not specs[1].get('deferred_twiddle')
            or specs[1]['r']*specs[2]['r']!=91
            or specs[1]['physical_prefix']!=256 or specs[2]['name']=='fused_finish'):
        raise ValueError('Late pair requires a dense aligned prefix and a nonterminal 13 x 7 pair')
    producer,first,second,following=specs[:4]
    r=first['r']*second['r'];m=second['m'];pp=first['physical_prefix']
    physical=first['physical_n']
    source=prephase(source,producer['name'],2*2*64*256,physical)
    name='dense_late_pair_batch' if batch_reuse>1 else 'dense_late_pair';tiles=pp//64
    body=f'''@triton.jit
def {name}(X,TW,Y):
    batch=tl.program_id(1).to(tl.int64)
    j=tl.program_id(0)//{tiles}
    col=(tl.program_id(0)%{tiles})*64+tl.arange(0,64)[None,:]
    digit=tl.arange(0,128)[:,None]
    off=(digit*{m}+j)*{pp}+col
    xr=tl.load(X+batch*{2*physical}+off,digit<{r},0.)
    xi=tl.load(X+batch*{2*physical}+{physical}+off,digit<{r},0.)
    row=tl.arange(0,128)[:,None]
    k=tl.arange(0,128)[None,:]
    basis=tl.load(TW+row*128+k)
    vr=tl.dot(basis,xr,input_precision="ieee")
    vi=tl.dot(basis,xi,input_precision="ieee")
    cr=al.extract_slice(vr,[0,0],[64,64],[1,1])
    sr=al.extract_slice(vr,[64,0],[64,64],[1,1])
    ci=al.extract_slice(vi,[0,0],[64,64],[1,1])
    si=al.extract_slice(vi,[64,0],[64,64],[1,1])
    freq=tl.arange(0,64)[:,None]
    out=(j*{r}+freq)*{pp}+col
    tl.store(Y+batch*{2*physical}+out,cr-si)
    tl.store(Y+batch*{2*physical}+{physical}+out,ci+sr)
    lane=tl.arange(0,4096)
    nf=64+lane//64
    take=tl.maximum(0,{r}-nf)*64+lane%64
    nr=tl.gather(tl.reshape(cr+si,(4096,)),take,0)
    ni=tl.gather(tl.reshape(ci-sr,(4096,)),take,0)
    out=(j*{r}+64+freq)*{pp}+col
    tl.store(Y+batch*{2*physical}+out,tl.reshape(nr,(64,64)),64+freq<{r})
    tl.store(Y+batch*{2*physical}+{physical}+out,tl.reshape(ni,(64,64)),64+freq<{r})
'''

    if batch_reuse>1:
        from .npu_matrix_batch import batch_matrix_source
        body=batch_matrix_source(body,batch_reuse)
    source=_replace_function(source,first['name'],body)
    source=_replace_function(source,second['name'],'')
    first.update(name=name,r=r,m=m,grid=(m*tiles,),batch_pack=batch_reuse,merged_late_pair=True)
    following['previous_prefix']=first['prefix']
    following['merged_previous_radix']=r
    del specs[2]
    return source,specs
