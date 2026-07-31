import numpy as np


def extract(a, t, x_shape):
    if hasattr(a, "gather"):
        out = a.gather(-1, t.long())
    else:
        out = np.asarray(a)[t.astype(np.int64)]
    return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))
