import theano
from theano import tensor
import numpy
import mkl_elementwise_mul_op
import unittest 

class testMklAdd(unittest.TestCase):
    def testFloat(self):
        x0 = tensor.fmatrix('x0')
        y0 = tensor.fmatrix('y0')
        z0 = mkl_elementwise_mul_op.ElementwiseMultiply()(x0, y0)
        f0 = theano.function([x0, y0], z0)
        a0 = numpy.random.rand(6, 8).astype(numpy.float32)
        b0 = numpy.random.rand(6, 8).astype(numpy.float32)
        #print(a0)
        #print(b0)
        o0 = f0(a0, b0)
        #print(o0)
        self.assertTrue(numpy.allclose(o0, a0*b0))

    def testDouble(self):
        x1 = tensor.dmatrix('x1')
        y1 = tensor.dmatrix('y1')
        z1 = mkl_elementwise_mul_op.ElementwiseMultiply()(x1, y1)
        f1 = theano.function([x1, y1], z1)
        theano.printing.pydotprint(f1, outfile='mkl_mul.png', var_with_name_simple=True)
        a1 = numpy.random.rand(6, 8).astype(numpy.float64)
        b1 = numpy.random.rand(6, 8).astype(numpy.float64)
        #print(a1)
        #print(b1)
        o1 = f1(a1, b1)
        #print(o1)
        self.assertTrue(numpy.allclose(o1, a1*b1))
    


if __name__ =='__main__': 
  unittest.main()