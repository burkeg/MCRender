import numpy as np
import math
import struct
# http://weitz.de/ieee/
class MyFloat:
    def __init__(self, value=None, valueFormat=None):
        self.format = type(value)
        self.original = value
        if value is None:
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
        c = MyFloat(valueFormat=a.format)
        AS, AE, AT = [int(_, 2) for _ in [a.S, a.E, a.T]]
        BS, BE, BT = [int(_, 2) for _ in [b.S, b.E, b.T]]
        CS, CE, CT = [int(_, 2) for _ in [c.S, c.E, c.T]]
        # Make sure A has the largest exponent
        if AE < BE:
            tmpS, tmpE, tmpT = AS, AE, AT
            AS, AE, AT = BS, BE, BT
            BS, BE, BT = tmpS, tmpE, tmpT

        shiftAmt = AE - BE
        CE = AE
        BT >>= shiftAmt

        if AS == BS:
            Acombined = (AE << a.SignificandBits) + AT
            Bcombined = (BE << b.SignificandBits) + BT
            Ccombined = Acombined + Bcombined
            CT = Ccombined % 1 << a.SignificandBits
            CE = Ccombined >> a.SignificandBits
            # CT = AT + BT
            # if CT >= 1 << c.SignificandBits:
            #     CE += 1
            #     # Losing some precision due to rounding here
            #     CT %= 1 << c.SignificandBits

            if CE >= (1 << a.ExponentBits):
                # Signs matched, set INF
                CS = AS
                CE = (1 << a.ExponentBits) - 1
                CT = 1 << (a.ExponentBits - 1)

            CS = AS
        else:
            # Make sure to set sign bit correctly
            pass

        # Make sure to normalize.
        # TODO

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
    def Multiply(a, b, manual=False):
        assert isinstance(a, MyFloat) and isinstance(b, MyFloat) and a.format == b.format
        if not manual:
            return MyFloat(a.original * b.original)
        else:
            raise NotImplementedError()

def TestAdd():
    mf1 = MyFloat(np.float16(3))
    mf2 = MyFloat(np.float16(7))
    mfSum = MyFloat.Add(mf1, mf2, True)
    print(mfSum)

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