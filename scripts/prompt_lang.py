import torch
from lark import Lark, Transformer, v_args

class T(Transformer):

    def STRING(self, s):
        return s[1:-1]

    def FLOAT_NUMBER(self, n):
        return float(n)

    @v_args(inline=True)
    def start(self, x):
        return x

def eval_prompt_expression(text: str, text_embed_function):
    """
    Parses the prompt expression and returns the embedding of the prompt.
    """
    # create parser
    parser = Lark("""?start: sum
?sum: product
    | sum "+" product   -> add
    | sum "-" product   -> sub

?product: atom
    | product "*" atom  -> mul

?atom: FLOAT_NUMBER     -> number
     | STRING           -> prompt
     | "(" sum ")"

%import python (STRING, FLOAT_NUMBER)
%import common.WS
%ignore WS""", start="start",transformer=T(), parser="lalr")
    # parse
    tree = parser.parse(text)
    # evaluate
    return eval_tree(tree, text_embed_function)

def eval_tree(tree, text_embed_function):
    if tree.data == "number":
        return float(tree.children[0])
    elif tree.data == "prompt":
        return text_embed_function(tree.children[0])
    elif tree.data == "add":
        lhs = eval_tree(tree.children[0], text_embed_function)
        rhs = eval_tree(tree.children[1], text_embed_function)
        return torch.add(lhs, rhs)
    elif tree.data == "sub":
        lhs = eval_tree(tree.children[0], text_embed_function)
        rhs = eval_tree(tree.children[1], text_embed_function)
        return torch.sub(lhs, rhs)
    elif tree.data == "mul":
        lhs = eval_tree(tree.children[0], text_embed_function)
        rhs = eval_tree(tree.children[1], text_embed_function)
        return torch.mul(lhs, rhs)
    else:
        raise ValueError(f"Unknown tree data: {tree.data}")


