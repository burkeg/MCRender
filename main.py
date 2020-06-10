import numbers, math
import numpy as np

class InputData:
    def __init__(self):
        pass
    
class MCRender():
    pass

class Point:
    def __init__(self, *args):
        for arg in args:
            assert isinstance(arg, numbers.Number)
        self._coord = list(args)
        if len(args) > 3 or len(args) == 0:
            raise Exception('Too many dimensions.')
        self.dim = len(self._coord)
        if len(args) > 2:
            self.z = args[2]
        if len(args) > 1:
            self.y = args[1]
        if len(args) > 0:
            self.x = args[0]

    def AsNp(self):
        return np.array(self._coord)

    @property
    def x(self):
        assert self.dim >= 1
        return self._coord[0]

    @x.setter
    def x(self, value):
        if self.dim >= 1:
            self._coord[0] = value

    @property
    def y(self):
        assert self.dim >= 2
        return self._coord[1]

    @y.setter
    def y(self, value):
        if self.dim >= 2:
            self._coord[1] = value

    @property
    def z(self):
        assert self.dim >= 3
        return self._coord[2]

    @z.setter
    def z(self, value):
        if self.dim >= 3:
            self._coord[2] = value

    def Dist(self, otherPoint):
        assert isinstance(otherPoint, Point)
        assert self.dim == otherPoint.dim, "Dimension mismatch"
        return math.sqrt(sum([(a-b)**2 for a, b in zip(self.AsTuple(), otherPoint.AsTuple())]))


def Test():
    a = Point(2, 3)
    b = Point(3, 2)
    print(a.Dist(b))


if __name__ == '__main__':
    Test()