from functools import wraps
from time import time
from typing import Any, Callable, Optional


def timing(print_output: bool = False, fname: Optional[str] = None) -> Callable:
    def timing_decorator(f: Callable) -> Callable:
        if fname is None:
            name = f.__name__
        else:
            name = fname

        @wraps(f)
        def wrap(*args: Any, **kw: Any) -> Any:
            ts = time()
            result = f(*args, **kw)
            te = time()
            text = "func:%r took: %2.4f sec" % (name, te - ts)
            text += f" | Result: {result})" if print_output else ""
            print(text)
            # LOGGER.info(text)
            return result

        return wrap

    return timing_decorator
