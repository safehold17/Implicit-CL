"""
负责判断 Nocturne 仿真中的车辆位置是否表示“真实存在”。
该模块统一处理非法数值与哨兵坐标，为状态更新和存在性逻辑提供基础判断。
Determines whether a vehicle position in Nocturne represents a valid simulated existence.
Normalizes sentinel coordinates and invalid numbers for state-update and existence logic.
"""

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
