import numbers, math
import numpy as np
from Matrix import *
from Camera import *
from scipy.interpolate import interp1d
import time

class InputData:
    def __init__(self):
        pass
    
class MCRender():
    pass

class Point(Matrix):
    def __init__(self, *args):
        for arg in args:
            assert isinstance(arg, numbers.Number)
        if len(args) > 3 or len(args) == 0:
            raise Exception('Too many dimensions.')
        self.dim = len(args)
        super().__init__(asArr=Matrix.ColumnVector(args).mat)
        # if len(args) > 2:
        #     self.z = args[2]
        # if len(args) > 1:
        #     self.y = args[1]
        # if len(args) > 0:
        #     self.x = args[0]
        super().__init__(asArr=Matrix.ColumnVector(self.AsTuple()).mat)

    def AsTuple(self):
        return tuple([x[0] for x in self.mat])

    @property
    def x(self):
        assert self.dim >= 1
        return self.mat[0][0]

    @x.setter
    def x(self, value):
        if self.dim >= 1:
            self.mat[0][0] = value

    @property
    def y(self):
        assert self.dim >= 2
        return self.mat[1][0]

    @y.setter
    def y(self, value):
        if self.dim >= 2:
            self.mat[1][0] = value

    @property
    def z(self):
        assert self.dim >= 3
        return self.mat[2][0]

    @z.setter
    def z(self, value):
        if self.dim >= 3:
            self.mat[2][0] = value

    def Dist(self, otherPoint):
        assert isinstance(otherPoint, Point)
        assert self.dim == otherPoint.dim, "Dimension mismatch"
        return math.sqrt(sum([(a-b)**2 for a, b in zip(self.AsTuple(), otherPoint.AsTuple())]))

class Shape:
    def __init__(self, points=None):
        if points is None:
            points = []
        self.points = points

    def AddPoints(self, points):
        self.points += points

    def MatrixOpPoints(self, matrixOp, *args):
        for point in self.points:
            matrixOp(point, *args)

class Screen:
    def __init__(self, height=400, width=600):
        self.height = height
        self.width = width
        self.camera = Camera()
        self.camera.ConfigureOrthographic(-100, 100, -100, 100, 0, 100)
        self.xPosCalc = interp1d([-1, 1], [0, self.width - 1])
        self.yPosCalc = interp1d([1, -1], [0, self.height - 1])
        self.inRange = lambda x: -1 <= x <= 1

    def ConvertToScreenPoints(self, shape):
        assert isinstance(shape, Shape)
        screenPoints = []
        for point in shape.points:
            assert isinstance(point, Point)
            x, y = self.camera.ToCameraPoint(point)
            if self.inRange(x) and self.inRange(y):
                screenPoints.append(Point(int(self.xPosCalc(x)), int(self.yPosCalc(y))))
        return screenPoints

    def DrawPointsAsAscii(self, shape):
        pixels = np.zeros(shape=(self.height, self.width))
        for p in self.ConvertToScreenPoints(shape):
            assert isinstance(p, Point)
            pixels[p.y][p.x] = 1
        rowStrs = ['-' * (self.width + 2)]
        for row in pixels:
            lineChars = ['|']
            for pixel in row:
                lineChars.append('#' if pixel == 1 else ' ')
            rowStrs.append(''.join(lineChars + ['|']))
        print('\n'.join(rowStrs + [rowStrs[0]]))



def Test():
    a = Point(0, 0, 10)
    b = Point(50, 0, 10)
    c = Point(50, 50, 10)
    shape = Shape([a, b, c])
    numPts = 11
    for i in range(numPts):
        screen = Screen(height=20, width=80)
        screen.DrawPointsAsAscii(shape)
        shape.MatrixOpPoints(Matrix.Translate, [15, -10, 0])
        time.sleep(0.5)



if __name__ == '__main__':
    Test()