from MyFloat import MyFloat
import numpy as np
import operator
import random as rand
import itertools

# I am intentionally hitting failure cases to make sure I handle them properly
np.seterr(all='ignore')
op2str = {
    operator.add: '+',
    operator.sub: '-',
    operator.mul: '*',
}
randomizedCases = 10_000

class FloatTesting:
    def __init__(self, typeToTest=np.float16, manualCases=True, utahCases=False,
                 randomCases=False, previouslyFailedRandomCases=False, specialCases=False,
                 specialAndRandomCases=True):
        self.cases = []
        self.typeToTest = typeToTest
        self.operators = [
            # operator.add,
            # operator.sub,
            operator.mul,
            # operator.truediv
        ]
        if manualCases:
            self.cases.extend(self.buildManualCases())
        if utahCases:
            self.cases.extend(self.buildUtahCases())
        if randomCases:
            self.cases.extend(self.buildRandomCases())
        if previouslyFailedRandomCases:
            self.cases.extend(self.buildPreviouslyFailedRandomCases())
        if specialCases:
            self.cases.extend(self.buildSpecialCases())
        if specialAndRandomCases:
            self.cases.extend(self.buildSpecialAndRandomCases())
            print(len(self.cases))

    def buildManualCases(self):
        return [
            [3.05e-5, 3.05e-5],
            [6e-08, 0.9995],
            [0.9995, -1.0],
            [300, 700],
            [1.0, 1.0],
            [0.5874, -0.1327],
            [2050, 3],
            [0.7217, 1.865E-3],
            [1.0/3.0, 2.0/3.0],
            [2048, 3],
            [1, 0],
            [1, -2],
            [1, -1],
            [1.001, 1.0],
            [0.0, 0.0],
            [0.0, -0.0],
            [-0.0, 0.0],
            [-0.0, -0.0],
            [np.infty - np.infty, np.nan], # NaN from invalid calc + plain NaN
            [np.nan, np.infty - np.infty], # plain NaN + NaN from invalid calc
            [3, 7],
            [30, 70],
            [300, 700],
            [3000, 7000],
            [3, 7000],
            [1, 2047],
            [2, 2047],
            [-27, np.infty],
            [np.infty, np.infty],
            [np.infty, -np.infty],
            [-np.infty, -np.infty],
            [-np.infty, np.infty],
            [np.NaN, 0],
            [3.052E-5, 3.052E-5], #subnormal + subnormal = normal
            [6.104E-5, -3.052E-5], #normal + subnormal = subnormal
            [6.104E-5, -6.11E-5], #normal + normal = subnormal
        ]

    def buildUtahCases(self):
        utahCases = []
        with open('UtahActualCalculationErrors.txt') as f:
            utahCases.extend([[self.typeToTest(x) for x in line.split(' ')] for line in f.readlines()])
        return utahCases

    def buildRandomCases(self):
        randomCases = []
        for _ in range(randomizedCases):
            randomCases.append([self.RandFloat(), self.RandFloat()])
        return randomCases

    def buildPreviouslyFailedRandomCases(self):
        previouslyFailedRandomCases = []
        with open('RandomTestFailures.txt') as f:
            previouslyFailedRandomCases.extend([[self.typeToTest(x) for x in line.split(' ')] for line in f.readlines()])
        return previouslyFailedRandomCases

    def buildSpecialCases(self):
        f = MyFloat(valueFormat=self.typeToTest)
        def getNegative(toNegate):
            assert isinstance(toNegate, MyFloat)
            negated = MyFloat(toNegate.original)
            negated.S = '1' if negated.S == '0' else '0'
            negated.Reinterpret()
            return negated

        specialValues = [
            f.Zero(),
            f.MinSubNorm(),
            f.NextMinSubNorm(),
            f.MidSubNorm(),
            f.PrevMaxSubNorm(),
            f.MaxSubNorm(),
            f.MinNorm(),
            f.NextMinNorm(),
            f.PrevOne(),
            f.One(),
            f.NextOne(),
            f.MidNorm(),
            f.PrevMaxNorm(),
            f.MaxNorm(),
            f.Infinity(),
            f.DefaultNaN()
        ]
        specialValues = specialValues + [getNegative(x) for x in specialValues]
        return [[lhs.original, rhs.original] for lhs, rhs in itertools.combinations_with_replacement(specialValues, 2)]

    def buildSpecialAndRandomCases(self):
        f = MyFloat(valueFormat=self.typeToTest)
        def getNegative(toNegate):
            assert isinstance(toNegate, MyFloat)
            negated = MyFloat(toNegate.original)
            negated.S = '1' if negated.S == '0' else '0'
            negated.Reinterpret()
            return negated

        specialValues = [
            f.Zero(),
            f.MinSubNorm(),
            f.NextMinSubNorm(),
            f.MidSubNorm(),
            f.PrevMaxSubNorm(),
            f.MaxSubNorm(),
            f.MinNorm(),
            f.NextMinNorm(),
            f.PrevOne(),
            f.One(),
            f.NextOne(),
            f.MidNorm(),
            f.PrevMaxNorm(),
            f.MaxNorm(),
            f.Infinity(),
            f.DefaultNaN()
        ]
        specialValues += [MyFloat(self.RandFloat()) for _ in range(50)]
        specialValues = specialValues + [getNegative(x) for x in specialValues]
        return [[lhs.original, rhs.original] for lhs, rhs in itertools.combinations_with_replacement(specialValues, 2)]

    def RandFloat(self):
        return MyFloat.bin2floatStatic(
            self.typeToTest,
            format(
                rand.getrandbits(np.dtype(self.typeToTest).itemsize * 8),
                '01b'))


    def RunTesting(self):
        for operation in self.operators:
            passed = 0
            total = 0
            for a, b in self.cases:
                A = MyFloat(a, self.typeToTest)
                B = MyFloat(b, self.typeToTest)
                C = operation(A, B)
                mine = C.original
                actual = operation(A.original, B.original)
                if C.EqualsFloatBits(actual):
                    passed += 1
                    # print('----------------')
                    # print(A, op2str[operation], B)
                    # print('Matches: ', MyFloat(actual))
                    pass
                else:
                    print('----------------')
                    print(A, op2str[operation], B)
                    # print(A.original, B.original)
                    print('FAILED: ', C, ' Actual: ', MyFloat(actual))
                    pass
                total += 1
            print(op2str[operation] + ':', str(100*passed/total) + '%')

if __name__ == '__main__':
    FloatTesting(typeToTest=np.float16).RunTesting()