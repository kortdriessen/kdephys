from . import *

"""Top-level API for cudf-flavor."""
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "register": [
            "register_pldf_method",
        ],
    },
)