import numpy as np
from scipy.interpolate import CubicSpline
from typing import Union


class SwerlingTarget(object):

    def __init__(self,
                 init_pos: np.ndarray = None,
                 init_vel: np.ndarray = None,
                 init_time: np.ndarray = None,
                 rcs: Union[float, list] = 1.,
                 prf: float = 0,
                 swerling_type: int = 1,
                 motion_track: np.ndarray = None):
        # Calculate out the motion of the target over time
        period = 1 / prf
        if not motion_track:
            motion_track = init_pos + init_vel * np.arange(1000) * period
            
        # Get a spline so we know the position at any (defined) time
        motion_t = init_time + np.arange(1000) * period
            
        self._pos = lambda t: np.array([CubicSpline(motion_t, motion_track[:, 0]), 
                                       CubicSpline(motion_t, motion_track[:, 1])])
        self.ttype = swerling_type
        
        self.active = (init_time, init_time + period * 1000)
        
        self.rcs = rcs
        
    @property
    def pos(self):
        return self._pos
    
    
class TargetList(list):
    
    def __init__(self):
        super().__init__()
        
    def getTargets(self, t):
        if not isinstance(t, np.ndarray):
            return [(targ.pos(t), targ.rcs) for targ in self if targ.active[0] <= t <= targ.active[1]]
        ret = []
        for targ in self:
            if nt := t[np.logical_and(targ.active[0] <= t, targ.active[1] >= t)]:
                ret.append((targ.pos(nt), targ.rcs))
        return ret

