import math


def sim_position_exists(x: float, y: float) -> bool:
    try:
        xf = float(x)
        yf = float(y)
    except Exception:
        return False
    if not (math.isfinite(xf) and math.isfinite(yf)):
        return False
    return not (xf == -10000.0 and yf == -10000.0)
