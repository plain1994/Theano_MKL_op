import theano
from theano import tensor
import numpy
import mkl_elementwise_mul_op
import unittest


class testMklAdd(unittest.TestCase):
    def test_Float(self):
        x0 = tensor.fmatrix('x0')
        y0 = tensor.fmatrix('y0')
        z0 = mkl_elementwise_mul_op.ElementwiseMultiply()(x0, y0)
        f0 = theano.function([x0, y0], z0)
        a0 = numpy.random.rand(6, 8).astype(numpy.float32)
        b0 = numpy.random.rand(6, 8).astype(numpy.float32)
        # print(a0)
        # print(b0)
        o0 = f0(a0, b0)
        # print(o0)
        self.assertTrue(numpy.allclose(o0, a0*b0))

    def test_Double(self):
        x1 = tensor.dmatrix('x1')
        y1 = tensor.dmatrix('y1')
        z1 = mkl_elementwise_mul_op.ElementwiseMultiply()(x1, y1)
        f1 = theano.function([x1, y1], z1)
        theano.printing.pydotprint(
            f1, outfile='mkl_mul.png', var_with_name_simple=True)
        a1 = numpy.random.rand(6, 8).astype(numpy.float64)
        b1 = numpy.random.rand(6, 8).astype(numpy.float64)
        # print(a1)
        # print(b1)
        o1 = f1(a1, b1)
        # print(o1)
        self.assertTrue(numpy.allclose(o1, a1*b1))

    def test_1D_dvector(self):
        x = tensor.dvector('x')
        y = tensor.dvector('y')

        z = mkl_elementwise_mul_op.ElementwiseMultiply()(x, y)
        f = theano.function([x, y], z)

        a = numpy.random.rand(6).astype(numpy.float64)
        b = numpy.random.rand(6).astype(numpy.float64)

        o = f(a, b)
        self.assertTrue(numpy.allclose(o, a*b))

    def test_2D_dmatrix(self):
        x = tensor.dmatrix('x')
        y = tensor.dmatrix('y')

        z = mkl_elementwise_mul_op.ElementwiseMultiply()(x, y)
        f = theano.function([x, y], z)

        a = numpy.random.rand(6, 7).astype(numpy.float64)
        b = numpy.random.rand(6, 7).astype(numpy.float64)

        o = f(a, b)
        self.assertTrue(numpy.allclose(o, a*b))

    def test_3D_dtensor(self):
        x = tensor.dtensor3('x')
        y = tensor.dtensor3('y')

        z = mkl_elementwise_mul_op.ElementwiseMultiply()(x, y)
        f = theano.function([x, y], z)

        a = numpy.random.rand(6, 7, 8).astype(numpy.float64)
        b = numpy.random.rand(6, 7, 8).astype(numpy.float64)

        o = f(a, b)
        self.assertTrue(numpy.allclose(o, a*b))

    def test_4D_dtensor(self):
        x = tensor.dtensor4('x')
        y = tensor.dtensor4('y')

        z = mkl_elementwise_mul_op.ElementwiseMultiply()(x, y)
        f = theano.function([x, y], z)

        a = numpy.random.rand(6, 7, 8, 9).astype(numpy.float64)
        b = numpy.random.rand(6, 7, 8, 9).astype(numpy.float64)

        o = f(a, b)
        self.assertTrue(numpy.allclose(o, a*b))

    def test_5D_dtensor(self):
        x = tensor.dtensor5('x')
        y = tensor.dtensor5('y')

        z = mkl_elementwise_mul_op.ElementwiseMultiply()(x, y)
        f = theano.function([x, y], z)

        a = numpy.random.rand(6, 7, 8, 9, 10).astype(numpy.float64)
        b = numpy.random.rand(6, 7, 8, 9, 10).astype(numpy.float64)

        o = f(a, b)
        self.assertTrue(numpy.allclose(o, a*b))

    def test_1D_fvector(self):
        x = tensor.fvector('x')
        y = tensor.fvector('y')

        z = mkl_elementwise_mul_op.ElementwiseMultiply()(x, y)
        f = theano.function([x, y], z)

        a = numpy.random.rand(6).astype(numpy.float32)
        b = numpy.random.rand(6).astype(numpy.float32)

        o = f(a, b)
        self.assertTrue(numpy.allclose(o, a*b))

    def test_2D_fMatrix(self):
        x = tensor.fmatrix('x')
        y = tensor.fmatrix('y')

        z = mkl_elementwise_mul_op.ElementwiseMultiply()(x, y)
        f = theano.function([x, y], z)

        a = numpy.random.rand(6, 7).astype(numpy.float32)
        b = numpy.random.rand(6, 7).astype(numpy.float32)

        o = f(a, b)
        self.assertTrue(numpy.allclose(o, a*b))

    def test_3D_ftensor(self):
        x = tensor.ftensor3('x')
        y = tensor.ftensor3('y')

        z = mkl_elementwise_mul_op.ElementwiseMultiply()(x, y)
        f = theano.function([x, y], z)

        a = numpy.random.rand(6, 7, 8).astype(numpy.float32)
        b = numpy.random.rand(6, 7, 8).astype(numpy.float32)

        o = f(a, b)
        self.assertTrue(numpy.allclose(o, a*b))

    def test_4D_ftensor(self):
        x = tensor.ftensor4('x')
        y = tensor.ftensor4('y')

        z = mkl_elementwise_mul_op.ElementwiseMultiply()(x, y)
        f = theano.function([x, y], z)

        a = numpy.random.rand(6, 7, 8, 9).astype(numpy.float32)
        b = numpy.random.rand(6, 7, 8, 9).astype(numpy.float32)

        o = f(a, b)
        self.assertTrue(numpy.allclose(o, a*b))

    def test_5D_ftensor(self):
        x = tensor.ftensor5('x')
        y = tensor.ftensor5('y')

        z = mkl_elementwise_mul_op.ElementwiseMultiply()(x, y)
        f = theano.function([x, y], z)

        a = numpy.random.rand(6, 7, 8, 9, 10).astype(numpy.float32)
        b = numpy.random.rand(6, 7, 8, 9, 10).astype(numpy.float32)

        o = f(a, b)
        self.assertTrue(numpy.allclose(o, a*b))


if __name__ == '__main__':
    unittest.main()
