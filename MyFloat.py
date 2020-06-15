import numpy as np
import math
import struct
from enum import Enum
# http://weitz.de/ieee/
# https://ieeexplore.ieee.org/abstract/document/4380621/authors#authors

class FloatFlag(Enum):
    OK = 0
    INVALID = 1 << 0
    DIV0 = 1 << 1
    OVERFLOW = 1 << 2
    UNDERFLOW = 1 << 3
    INEXACT = 1 << 4

class MyFloat:
    status = FloatFlag.OK

    def __init__(self, value=None, valueFormat=None):
        self.format = None
        self.original = 0.0
        if value is not None:
            if valueFormat is not None:
                self.format = valueFormat
                self.original = valueFormat(value)
            else:
                self.format = type(value)
                self.original = value
        else:
            assert valueFormat is not None
            self.format = valueFormat
            self.original = valueFormat(0.0)
        self.CommonName = None
        self.Base = None
        self.SignificandBits = None
        self.DecimalDigits = None
        self.ExponentBits = None
        self.DecimalEMax = None
        self.ExponentBias = None
        self.EMin = None
        self.EMax = None
        self.S = None
        self.E = None
        self.T = None
        self.PopulateData()

    def __str__(self):
        return str(self.original) + ': ' + '{' + ','.join([self.S, self.E, self.T]) + '}'
    def __unicode__(self):
        return str(self)
    def __repr__(self):
        return str(self)
    def __add__(self, other):
        return MyFloat.Add(self, other)
    def __radd__(self, other):
        return MyFloat.Add(self, other)
    def __mul__(self, other):
        return MyFloat.Multiply(self, other)
    def __rmul__(self, other):
        return MyFloat.Multiply(self, other)

    @staticmethod
    def SetFlag(flag):
        MyFloat.status |= flag

    @staticmethod
    def ClearFlag(flag):
        MyFloat.status &= ~flag

    def Reinterpret(self):
        self.original = self.bin2float(self.S + self.E + self. T)
        pass

    # https://stackoverflow.com/a/59594903
    def bin2float(self, b):
        ''' Convert binary string to a float.

        Attributes:
            :b: Binary string to transform.
        '''
        if self.format == np.float16:
            h = int(b, 2).to_bytes(2, byteorder="big")
            return struct.unpack('>e', h)[0]
        elif self.format == np.float32:
            h = int(b, 2).to_bytes(4, byteorder="big")
            return struct.unpack('>f', h)[0]
        elif self.format == np.float64:
            h = int(b, 2).to_bytes(8, byteorder="big")
            return struct.unpack('>d', h)[0]

    def float2bin(self, f):
        ''' Convert float to n-bit binary string.

        Attributes:
            :f: Float number to transform.
        '''
        if self.format == np.float16:
            [d] = struct.unpack(">H", struct.pack(">e", f))
            return f'{d:016b}'
        elif self.format == np.float32:
            [d] = struct.unpack(">I", struct.pack(">f", f))
            return f'{d:032b}'
        elif self.format == np.float64:
            [d] = struct.unpack(">Q", struct.pack(">d", f))
            return f'{d:064b}'

    def PopulateData(self):
        if self.format == np.float16:
            self.CommonName = 'Half Precision'
            self.Base = 2
            self.SignificandBits = 10
            self.DecimalDigits = 3.31
            self.ExponentBits = 5
            self.DecimalEMax = 4.51
            self.ExponentBias = 2**4-1
            self.EMin = -14
            self.EMax = 15
            bitsSoFar = 0
            self.S = self.float2bin(self.original)[bitsSoFar:(bitsSoFar+1)]
            bitsSoFar += 1
            self.E = self.float2bin(self.original)[bitsSoFar:(bitsSoFar+self.ExponentBits)]
            bitsSoFar += self.ExponentBits
            self.T = self.float2bin(self.original)[bitsSoFar:(bitsSoFar + self.SignificandBits)]
            assert bitsSoFar + self.SignificandBits == 16
        elif self.format == np.float32:
            self.CommonName = 'Single Precision'
            self.Base = 2
            self.SignificandBits = 23
            self.DecimalDigits = 7.22
            self.ExponentBits = 8
            self.DecimalEMax = 38.23
            self.ExponentBias = 2**7-1
            self.EMin = -126
            self.EMax = 127
            bitsSoFar = 0
            self.S = self.float2bin(self.original)[bitsSoFar:(bitsSoFar+1)]
            bitsSoFar += 1
            self.E = self.float2bin(self.original)[bitsSoFar:(bitsSoFar+self.ExponentBits)]
            bitsSoFar += self.ExponentBits
            self.T = self.float2bin(self.original)[bitsSoFar:(bitsSoFar + self.SignificandBits)]
            assert bitsSoFar + self.SignificandBits == 32
        else:
            raise Exception('Unacceptable format')

    @staticmethod
    def Add(a, b, manual=False):
        assert isinstance(a, MyFloat) and isinstance(b, MyFloat) and a.format == b.format
        if not manual:
            return MyFloat(a.original + b.original)
        infty = False
        inexact = False

        c = MyFloat(valueFormat=a.format)
        AS, AE, AT = [int(_, 2) for _ in [a.S, a.E, a.T]]
        BS, BE, BT = [int(_, 2) for _ in [b.S, b.E, b.T]]
        CS, CE, CT = [int(_, 2) for _ in [c.S, c.E, c.T]]
        # Make sure A has the largest exponent
        if AE < BE:
            tmpS, tmpE, tmpT = AS, AE, AT
            AS, AE, AT = BS, BE, BT
            BS, BE, BT = tmpS, tmpE, tmpT

        # Shift the exponent of the smaller number to match that of the large number
        shiftAmt = AE - BE
        CE = AE

        # Add implicit leading 1 to significand
        # TODO the implicit 1 doesn't exist in denormal numbers
        AT += (1 << a.SignificandBits)
        BT += (1 << b.SignificandBits)
        # TODO
        # detect loss of info in B here
        BT = (BT >> shiftAmt)

        if AS == BS:
            CT = AT + BT
            CS = AS
        else:
            CT = abs(AT - BT)
            CS = AS if abs(AT) > abs(BT) else BS
            # Make sure to set sign bit correctly
            pass

        CE, CT, infty, inexact = MyFloat.Normalize(CE, CT, c.ExponentBits, c.SignificandBits)

        if infty:
            CE = (1 << c.ExponentBits) - 1
            CT = 1 << (c.ExponentBits - 1)

        c.S = format(CS, '01b')
        c.E = format(CE, '0' + str(a.ExponentBits) + 'b')
        c.T = format(CT, '0' + str(a.SignificandBits) + 'b')
        # We probably messed up normalization if these assertions fail
        assert len(c.S) == 1
        assert len(c.E) == a.ExponentBits
        assert len(c.T) == a.SignificandBits
        c.Reinterpret()
        return c

    @staticmethod
    def Normalize(exponent, significand, exponentBits, significandBits):
        bitsShifted = 0
        infty = False
        infoLost = False
        while significand >= (1 << (significandBits + 1)):
            bitsShifted += 1
            if significand % 1:
                infoLost = True
            exponent += 1
            significand >>= 1

        significand %= 1 << significandBits

        if exponent >= (1 << exponentBits):
            infty = True

        return exponent, significand, infty, infoLost


    @staticmethod
    def Multiply(a, b, manual=False):
        assert isinstance(a, MyFloat) and isinstance(b, MyFloat) and a.format == b.format
        if not manual:
            return MyFloat(a.original * b.original)
        else:
            raise NotImplementedError()


def TestAdd():
    testCases = [
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, -0.0],
        [-0.0, 0.0],
        [-0.0, -0.0],
        [1.0/3.0, 2.0/3.0],
        [-27, np.infty],
        [np.infty, np.infty],
        [np.infty, -np.infty],
        [-np.infty, -np.infty],
        [-np.infty, np.infty],
        [np.NaN, 0]
    ]
    for a, b in testCases:
        A = MyFloat(a, np.float16)
        B = MyFloat(b, np.float16)
        C = MyFloat.Add(A, B, True)
        print(A, '+', B)
        mine = C.original
        actual = np.float16(a) + np.float16(b)
        print('Mine: ', mine, ' Actual: ', actual)
        assert mine == actual

def Test():
    print(MyFloat(np.float16(np.inf)))
    print(MyFloat(np.float16(np.inf-np.inf)))
    print(MyFloat(np.float16(np.NaN)))
    print(MyFloat(np.float16(0)))
    print(MyFloat(np.float16(1)))
    print(MyFloat(np.float16(2**14)))
    print(MyFloat(np.float16(2**15)))
    print(MyFloat(np.float16(2**16)))


if __name__ == '__main__':
    TestAdd()