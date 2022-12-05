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

    @staticmethod
    def Star(r=0.5):
        startShape = Shape(points=
                         [Point(math.cos(2*math.pi*k/5 + math.pi/2),
                                math.sin(2*math.pi*k/5 + math.pi/2),
                                0) for k in range(5)] +
                         [Point(r*math.cos(2*math.pi*k/5 + 7*math.pi/10),
                                r*math.sin(2*math.pi*k/5 + 7*math.pi/10),
                                0) for k in range(5)])
        return startShape

    @staticmethod
    def Teapot():
        # https://github.com/dasch/graphics/blob/master/data/teapot.data
        teapotShape = Shape(points=
                            [Point(1.40000, 0.00000, 2.40000),
                             Point(1.40000, -0.78400, 2.40000),
                             Point(0.78000, -1.40000, 2.40000),
                             Point(0.00000, -1.40000, 2.40000),
                             Point(1.33750, 0.00000, 2.53125),
                             Point(1.33750, -0.74900, 2.53125),
                             Point(0.74900, -1.33750, 2.53125),
                             Point(0.00000, -1.33750, 2.53125),
                             Point(1.43750, 0.00000, 2.53125),
                             Point(1.43750, -0.80500, 2.53125),
                             Point(0.80500, -1.43750, 2.53125),
                             Point(0.00000, -1.43750, 2.53125),
                             Point(1.50000, 0.00000, 2.40000),
                             Point(1.50000, -0.84000, 2.40000),
                             Point(0.84000, -1.50000, 2.40000),
                             Point(0.00000, -1.50000, 2.40000),
                             Point(-0.78400, -1.40000, 2.40000),
                             Point(-1.40000, -0.78400, 2.40000),
                             Point(-1.40000, 0.00000, 2.40000),
                             Point(-0.74900, -1.33750, 2.53125),
                             Point(-1.33750, -0.74900, 2.53125),
                             Point(-1.33750, 0.00000, 2.53125),
                             Point(-0.80500, -1.43750, 2.53125),
                             Point(-1.43750, -0.80500, 2.53125),
                             Point(-1.43750, 0.00000, 2.53125),
                             Point(-0.84000, -1.50000, 2.40000),
                             Point(-1.50000, -0.84000, 2.40000),
                             Point(-1.50000, 0.00000, 2.40000),
                             Point(-1.40000, 0.78400, 2.40000),
                             Point(-0.78400, 1.40000, 2.40000),
                             Point(0.00000, 1.40000, 2.40000),
                             Point(-1.33750, 0.74900, 2.53125),
                             Point(-0.74900, 1.33750, 2.53125),
                             Point(0.00000, 1.33750, 2.53125),
                             Point(-1.43750, 0.80500, 2.53125),
                             Point(-0.80500, 1.43750, 2.53125),
                             Point(0.00000, 1.43750, 2.53125),
                             Point(-1.50000, 0.84000, 2.40000),
                             Point(-0.84000, 1.50000, 2.40000),
                             Point(0.00000, 1.50000, 2.40000),
                             Point(0.78400, 1.40000, 2.40000),
                             Point(1.40000, 0.78400, 2.40000),
                             Point(0.74900, 1.33750, 2.53125),
                             Point(1.33750, 0.74900, 2.53125),
                             Point(0.80500, 1.43750, 2.53125),
                             Point(1.43750, 0.80500, 2.53125),
                             Point(0.84000, 1.50000, 2.40000),
                             Point(1.50000, 0.84000, 2.40000),
                             Point(1.75000, 0.00000, 1.87500),
                             Point(1.75000, -0.98000, 1.87500),
                             Point(0.98000, -1.75000, 1.87500),
                             Point(0.00000, -1.75000, 1.87500),
                             Point(2.00000, 0.00000, 1.35000),
                             Point(2.00000, -1.12000, 1.35000),
                             Point(1.12000, -2.00000, 1.35000),
                             Point(0.00000, -2.00000, 1.35000),
                             Point(2.00000, 0.00000, 0.90000),
                             Point(2.00000, -1.12000, 0.90000),
                             Point(1.12000, -2.00000, 0.90000),
                             Point(0.00000, -2.00000, 0.90000),
                             Point(-0.98000, -1.75000, 1.87500),
                             Point(-1.75000, -0.98000, 1.87500),
                             Point(-1.75000, 0.00000, 1.87500),
                             Point(-1.12000, -2.00000, 1.35000),
                             Point(-2.00000, -1.12000, 1.35000),
                             Point(-2.00000, 0.00000, 1.35000),
                             Point(-1.12000, -2.00000, 0.90000),
                             Point(-2.00000, -1.12000, 0.90000),
                             Point(-2.00000, 0.00000, 0.90000),
                             Point(-1.75000, 0.98000, 1.87500),
                             Point(-0.98000, 1.75000, 1.87500),
                             Point(0.00000, 1.75000, 1.87500),
                             Point(-2.00000, 1.12000, 1.35000),
                             Point(-1.12000, 2.00000, 1.35000),
                             Point(0.00000, 2.00000, 1.35000),
                             Point(-2.00000, 1.12000, 0.90000),
                             Point(-1.12000, 2.00000, 0.90000),
                             Point(0.00000, 2.00000, 0.90000),
                             Point(0.98000, 1.75000, 1.87500),
                             Point(1.75000, 0.98000, 1.87500),
                             Point(1.12000, 2.00000, 1.35000),
                             Point(2.00000, 1.12000, 1.35000),
                             Point(1.12000, 2.00000, 0.90000),
                             Point(2.00000, 1.12000, 0.90000),
                             Point(2.00000, 0.00000, 0.45000),
                             Point(2.00000, -1.12000, 0.45000),
                             Point(1.12000, -2.00000, 0.45000),
                             Point(0.00000, -2.00000, 0.45000),
                             Point(1.50000, 0.00000, 0.22500),
                             Point(1.50000, -0.84000, 0.22500),
                             Point(0.84000, -1.50000, 0.22500),
                             Point(0.00000, -1.50000, 0.22500),
                             Point(1.50000, 0.00000, 0.15000),
                             Point(1.50000, -0.84000, 0.15000),
                             Point(0.84000, -1.50000, 0.15000),
                             Point(0.00000, -1.50000, 0.15000),
                             Point(-1.12000, -2.00000, 0.45000),
                             Point(-2.00000, -1.12000, 0.45000),
                             Point(-2.00000, 0.00000, 0.45000),
                             Point(-0.84000, -1.50000, 0.22500),
                             Point(-1.50000, -0.84000, 0.22500),
                             Point(-1.50000, 0.00000, 0.22500),
                             Point(-0.84000, -1.50000, 0.15000),
                             Point(-1.50000, -0.84000, 0.15000),
                             Point(-1.50000, 0.00000, 0.15000),
                             Point(-2.00000, 1.12000, 0.45000),
                             Point(-1.12000, 2.00000, 0.45000),
                             Point(0.00000, 2.00000, 0.45000),
                             Point(-1.50000, 0.84000, 0.22500),
                             Point(-0.84000, 1.50000, 0.22500),
                             Point(0.00000, 1.50000, 0.22500),
                             Point(-1.50000, 0.84000, 0.15000),
                             Point(-0.84000, 1.50000, 0.15000),
                             Point(0.00000, 1.50000, 0.15000),
                             Point(1.12000, 2.00000, 0.45000),
                             Point(2.00000, 1.12000, 0.45000),
                             Point(0.84000, 1.50000, 0.22500),
                             Point(1.50000, 0.84000, 0.22500),
                             Point(0.84000, 1.50000, 0.15000),
                             Point(1.50000, 0.84000, 0.15000),
                             Point(-1.60000, 0.00000, 2.02500),
                             Point(-1.60000, -0.30000, 2.02500),
                             Point(-1.50000, -0.30000, 2.25000),
                             Point(-1.50000, 0.00000, 2.25000),
                             Point(-2.30000, 0.00000, 2.02500),
                             Point(-2.30000, -0.30000, 2.02500),
                             Point(-2.50000, -0.30000, 2.25000),
                             Point(-2.50000, 0.00000, 2.25000),
                             Point(-2.70000, 0.00000, 2.02500),
                             Point(-2.70000, -0.30000, 2.02500),
                             Point(-3.00000, -0.30000, 2.25000),
                             Point(-3.00000, 0.00000, 2.25000),
                             Point(-2.70000, 0.00000, 1.80000),
                             Point(-2.70000, -0.30000, 1.80000),
                             Point(-3.00000, -0.30000, 1.80000),
                             Point(-3.00000, 0.00000, 1.80000),
                             Point(-1.50000, 0.30000, 2.25000),
                             Point(-1.60000, 0.30000, 2.02500),
                             Point(-2.50000, 0.30000, 2.25000),
                             Point(-2.30000, 0.30000, 2.02500),
                             Point(-3.00000, 0.30000, 2.25000),
                             Point(-2.70000, 0.30000, 2.02500),
                             Point(-3.00000, 0.30000, 1.80000),
                             Point(-2.70000, 0.30000, 1.80000),
                             Point(-2.70000, 0.00000, 1.57500),
                             Point(-2.70000, -0.30000, 1.57500),
                             Point(-3.00000, -0.30000, 1.35000),
                             Point(-3.00000, 0.00000, 1.35000),
                             Point(-2.50000, 0.00000, 1.12500),
                             Point(-2.50000, -0.30000, 1.12500),
                             Point(-2.65000, -0.30000, 0.93750),
                             Point(-2.65000, 0.00000, 0.93750),
                             Point(-2.00000, -0.30000, 0.90000),
                             Point(-1.90000, -0.30000, 0.60000),
                             Point(-1.90000, 0.00000, 0.60000),
                             Point(-3.00000, 0.30000, 1.35000),
                             Point(-2.70000, 0.30000, 1.57500),
                             Point(-2.65000, 0.30000, 0.93750),
                             Point(-2.50000, 0.30000, 1.12500),
                             Point(-1.90000, 0.30000, 0.60000),
                             Point(-2.00000, 0.30000, 0.90000),
                             Point(1.70000, 0.00000, 1.42500),
                             Point(1.70000, -0.66000, 1.42500),
                             Point(1.70000, -0.66000, 0.60000),
                             Point(1.70000, 0.00000, 0.60000),
                             Point(2.60000, 0.00000, 1.42500),
                             Point(2.60000, -0.66000, 1.42500),
                             Point(3.10000, -0.66000, 0.82500),
                             Point(3.10000, 0.00000, 0.82500),
                             Point(2.30000, 0.00000, 2.10000),
                             Point(2.30000, -0.25000, 2.10000),
                             Point(2.40000, -0.25000, 2.02500),
                             Point(2.40000, 0.00000, 2.02500),
                             Point(2.70000, 0.00000, 2.40000),
                             Point(2.70000, -0.25000, 2.40000),
                             Point(3.30000, -0.25000, 2.40000),
                             Point(3.30000, 0.00000, 2.40000),
                             Point(1.70000, 0.66000, 0.60000),
                             Point(1.70000, 0.66000, 1.42500),
                             Point(3.10000, 0.66000, 0.82500),
                             Point(2.60000, 0.66000, 1.42500),
                             Point(2.40000, 0.25000, 2.02500),
                             Point(2.30000, 0.25000, 2.10000),
                             Point(3.30000, 0.25000, 2.40000),
                             Point(2.70000, 0.25000, 2.40000),
                             Point(2.80000, 0.00000, 2.47500),
                             Point(2.80000, -0.25000, 2.47500),
                             Point(3.52500, -0.25000, 2.49375),
                             Point(3.52500, 0.00000, 2.49375),
                             Point(2.90000, 0.00000, 2.47500),
                             Point(2.90000, -0.15000, 2.47500),
                             Point(3.45000, -0.15000, 2.51250),
                             Point(3.45000, 0.00000, 2.51250),
                             Point(2.80000, 0.00000, 2.40000),
                             Point(2.80000, -0.15000, 2.40000),
                             Point(3.20000, -0.15000, 2.40000),
                             Point(3.20000, 0.00000, 2.40000),
                             Point(3.52500, 0.25000, 2.49375),
                             Point(2.80000, 0.25000, 2.47500),
                             Point(3.45000, 0.15000, 2.51250),
                             Point(2.90000, 0.15000, 2.47500),
                             Point(3.20000, 0.15000, 2.40000),
                             Point(2.80000, 0.15000, 2.40000),
                             Point(0.00000, 0.00000, 3.15000),
                             Point(0.00000, -0.00200, 3.15000),
                             Point(0.00200, 0.00000, 3.15000),
                             Point(0.80000, 0.00000, 3.15000),
                             Point(0.80000, -0.45000, 3.15000),
                             Point(0.45000, -0.80000, 3.15000),
                             Point(0.00000, -0.80000, 3.15000),
                             Point(0.00000, 0.00000, 2.85000),
                             Point(0.20000, 0.00000, 2.70000),
                             Point(0.20000, -0.11200, 2.70000),
                             Point(0.11200, -0.20000, 2.70000),
                             Point(0.00000, -0.20000, 2.70000),
                             Point(-0.00200, 0.00000, 3.15000),
                             Point(-0.45000, -0.80000, 3.15000),
                             Point(-0.80000, -0.45000, 3.15000),
                             Point(-0.80000, 0.00000, 3.15000),
                             Point(-0.11200, -0.20000, 2.70000),
                             Point(-0.20000, -0.11200, 2.70000),
                             Point(-0.20000, 0.00000, 2.70000),
                             Point(0.00000, 0.00200, 3.15000),
                             Point(-0.80000, 0.45000, 3.15000),
                             Point(-0.45000, 0.80000, 3.15000),
                             Point(0.00000, 0.80000, 3.15000),
                             Point(-0.20000, 0.11200, 2.70000),
                             Point(-0.11200, 0.20000, 2.70000),
                             Point(0.00000, 0.20000, 2.70000),
                             Point(0.45000, 0.80000, 3.15000),
                             Point(0.80000, 0.45000, 3.15000),
                             Point(0.11200, 0.20000, 2.70000),
                             Point(0.20000, 0.11200, 2.70000),
                             Point(0.40000, 0.00000, 2.55000),
                             Point(0.40000, -0.22400, 2.55000),
                             Point(0.22400, -0.40000, 2.55000),
                             Point(0.00000, -0.40000, 2.55000),
                             Point(1.30000, 0.00000, 2.55000),
                             Point(1.30000, -0.72800, 2.55000),
                             Point(0.72800, -1.30000, 2.55000),
                             Point(0.00000, -1.30000, 2.55000),
                             Point(1.30000, 0.00000, 2.40000),
                             Point(1.30000, -0.72800, 2.40000),
                             Point(0.72800, -1.30000, 2.40000),
                             Point(0.00000, -1.30000, 2.40000),
                             Point(-0.22400, -0.40000, 2.55000),
                             Point(-0.40000, -0.22400, 2.55000),
                             Point(-0.40000, 0.00000, 2.55000),
                             Point(-0.72800, -1.30000, 2.55000),
                             Point(-1.30000, -0.72800, 2.55000),
                             Point(-1.30000, 0.00000, 2.55000),
                             Point(-0.72800, -1.30000, 2.40000),
                             Point(-1.30000, -0.72800, 2.40000),
                             Point(-1.30000, 0.00000, 2.40000),
                             Point(-0.40000, 0.22400, 2.55000),
                             Point(-0.22400, 0.40000, 2.55000),
                             Point(0.00000, 0.40000, 2.55000),
                             Point(-1.30000, 0.72800, 2.55000),
                             Point(-0.72800, 1.30000, 2.55000),
                             Point(0.00000, 1.30000, 2.55000),
                             Point(-1.30000, 0.72800, 2.40000),
                             Point(-0.72800, 1.30000, 2.40000),
                             Point(0.00000, 1.30000, 2.40000),
                             Point(0.22400, 0.40000, 2.55000),
                             Point(0.40000, 0.22400, 2.55000),
                             Point(0.72800, 1.30000, 2.55000),
                             Point(1.30000, 0.72800, 2.55000),
                             Point(0.72800, 1.30000, 2.40000),
                             Point(1.30000, 0.72800, 2.40000),
                             Point(0.00000, 0.00000, 0.00000),
                             Point(1.50000, 0.00000, 0.15000),
                             Point(1.50000, 0.84000, 0.15000),
                             Point(0.84000, 1.50000, 0.15000),
                             Point(0.00000, 1.50000, 0.15000),
                             Point(1.50000, 0.00000, 0.07500),
                             Point(1.50000, 0.84000, 0.07500),
                             Point(0.84000, 1.50000, 0.07500),
                             Point(0.00000, 1.50000, 0.07500),
                             Point(1.42500, 0.00000, 0.00000),
                             Point(1.42500, 0.79800, 0.00000),
                             Point(0.79800, 1.42500, 0.00000),
                             Point(0.00000, 1.42500, 0.00000),
                             Point(-0.84000, 1.50000, 0.15000),
                             Point(-1.50000, 0.84000, 0.15000),
                             Point(-1.50000, 0.00000, 0.15000),
                             Point(-0.84000, 1.50000, 0.07500),
                             Point(-1.50000, 0.84000, 0.07500),
                             Point(-1.50000, 0.00000, 0.07500),
                             Point(-0.79800, 1.42500, 0.00000),
                             Point(-1.42500, 0.79800, 0.00000),
                             Point(-1.42500, 0.00000, 0.00000),
                             Point(-1.50000, -0.84000, 0.15000),
                             Point(-0.84000, -1.50000, 0.15000),
                             Point(0.00000, -1.50000, 0.15000),
                             Point(-1.50000, -0.84000, 0.07500),
                             Point(-0.84000, -1.50000, 0.07500),
                             Point(0.00000, -1.50000, 0.07500),
                             Point(-1.42500, -0.79800, 0.00000),
                             Point(-0.79800, -1.42500, 0.00000),
                             Point(0.00000, -1.42500, 0.00000),
                             Point(0.84000, -1.50000, 0.15000),
                             Point(1.50000, -0.84000, 0.15000),
                             Point(0.84000, -1.50000, 0.07500),
                             Point(1.50000, -0.84000, 0.07500),
                             Point(0.79800, -1.42500, 0.00000),
                             Point(1.42500, -0.79800, 0.00000)])
        teapotShape.MatrixOpPoints(Matrix.Scale, [0.5, 0.5, 1])
        return teapotShape

class Screen:
    def __init__(self, height=400, width=600):
        self.height = height
        self.width = width
        self.camera = Camera()
        self.camera.ConfigureOrthographic(-1, 1, -1, 1, 1, 100)
        # self.camera.ConfigurePerspective(90, 1, 100)
        # self.camera.pos.x = 1
        self.camera.pos.x = 0
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
        pixels = np.zeros(shape=(self.height, self.width), dtype=np.float32)
        for p in self.ConvertToScreenPoints(shape):
            assert isinstance(p, Point)
            pixels[int(p.y)][int(p.x)] = 1
        rowStrs = ['-' * (self.width + 2)]
        for row in pixels:
            lineChars = ['|']
            for pixel in row:
                lineChars.append('#' if pixel == 1 else ' ')
            rowStrs.append(''.join(lineChars + ['|']))
        print('\n'.join(rowStrs + [rowStrs[0]]))
        if MyFloat.logFailures:
            pp.pprint(MyFloat.failureDict)

class Control:
    def __init__(self):
        pass

def TestShape(shape, degPerTurn=30, scale=None, rotate=None, translate=None):
    scale = scale if scale is not None else [1, 1, 1]
    rotate = rotate if rotate is not None else [0, 0, 0]
    translate = translate if translate is not None else [0, 0, 0]
    screen = Screen(height=20, width=50)
    shape.MatrixOpPoints(Matrix.Scale, scale)
    shape.MatrixOpPoints(Matrix.Rotate, rotate)
    shape.MatrixOpPoints(Matrix.Translate, translate)

    for i in range(360//degPerTurn):
        shape.MatrixOpPoints(Matrix.Rotate, [degPerTurn, 0, 0])
        screen.DrawPointsAsAscii(shape)
        time.sleep(0.1)
    for i in range(360//degPerTurn):
        shape.MatrixOpPoints(Matrix.Rotate, [0, degPerTurn, 0])
        screen.DrawPointsAsAscii(shape)
        time.sleep(0.1)
    for i in range(360//degPerTurn):
        shape.MatrixOpPoints(Matrix.Rotate, [0, 0, degPerTurn])
        screen.DrawPointsAsAscii(shape)
        time.sleep(0.1)



if __name__ == '__main__':
    TestShape(Shape.Teapot(), scale=[0.3, 0.3, 0.3], rotate=[-90, 0, 0], translate=[1, 0, 0])
