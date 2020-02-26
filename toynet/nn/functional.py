import re

from . import Node


def to_underscore_case(x):
    return re.sub('([A-Z]+)', r'_\1', x).lower().strip('_')


def register_functions():
    for cls in Node.__subclass__:
        if cls.__name__ in ['Input', 'Const', 'Param']:
            continue

        func_name = to_underscore_case(cls.__name__)

        def f(*args, **kwargs):
            node = cls(**kwargs)
            return node(*args)

        f.__name__ = func_name
        globals()[func_name] = f


# register_functions()
