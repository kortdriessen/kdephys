from functools import wraps
from polars.api import register_dataframe_namespace

def register_pldf_method(method):
    """Register a function as a method attached to the Polars DataFrame.
    Example
    -------
    .. code-block:: python
        @register_dataframe_method
        def print_column(df, col):
            '''Print the dataframe column given'''
            print(df[col])
    """

    def inner(*args, **kwargs):
        class AccessorMethod(object):
            def __init__(self, pldf_obj):
                self._obj = pldf_obj

            @wraps(method)
            def __call__(self, *args, **kwargs):
                return method(self._obj, *args, **kwargs)

        register_dataframe_namespace(method.__name__)(AccessorMethod)

        return method

    return inner()
