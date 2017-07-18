import theano
from theano import tensor, gof
import numpy

class ElementwiseMultiply(gof.Op):
    __props__ = ()

    def make_node(self, *inputs):
        # Validate the inputs' type
        inputs = list(map(tensor.as_tensor_variable, inputs))

        if len(inputs) != 2:
            raise TypeError('ElementwiseAdd: 2 arguments required, %d given' %len(inputs))

        if inputs[0].ndim != 2:
            raise TypeError('ElementwiseAdd: x must be a 2-d matrix')
        if inputs[1].ndim != 2:
            raise TypeError('ElementwiseAdd: y must be a 2-d matrix')

        if inputs[0].dtype != inputs[1].dtype:
            raise TypeError('ElementwiseAdd: elements in A and B should have same dtype')
        
        output_var = inputs[0].type()

        return gof.Apply(self, inputs, [output_var])

    def c_headers(self):
        headers = ['<mkl.h>']
        return headers

    def c_code(self, node, name, inputs, outputs, sub):
        if node.inputs[0].dtype is 'float32':
            ccode_type = 'float'
            ccode_mulfunc = 'vsMul'
        elif node.inputs[0].dtype is 'float64':
            ccode_type = 'double'
            ccode_mulfunc = 'vdMul'
        else:
            raise TypeError('ElementwiseAdd: dtype %s is not supported.'
                            % (node.inputs[0].dtype))



        x, y = inputs
        z, = outputs

        dtype_x = node.inputs[0].dtype
        dtype_y = node.inputs[1].dtype
        dtype_z = node.outputs[0].dtype

        itemsize_x = numpy.dtype(dtype_x).itemsize
        itemsize_y = numpy.dtype(dtype_y).itemsize
        itemsize_z = numpy.dtype(dtype_z).itemsize


        


        typenum_z = numpy.dtype(dtype_z).num

        fail = sub['fail']



        c_code = """

        // Validate that the output storage exists.
        if (NULL == %(z)s)
        {
            /* Reference received to invalid output variable.
            Decrease received reference's ref count and allocate new
            output variable */
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*)PyArray_EMPTY(2,
                                                PyArray_DIMS(%(x)s),
                                                %(typenum_z)s,
                                                0);

            if (!%(z)s) {
                %(fail)s;
            }
        }

        //int num1 = PyArray_DIMS(%(x)s)[0];
        //printf("num1: %%d \\n", num1);
        //int num2 = PyArray_DIMS(%(x)s)[1];
        //printf("num2: %%d \\n", num2);
	    //printf(\"%%llp, %%llp, %%llp\\n\", %(x)s, %(y)s, %(z)s);

        //Perform the MKL Add (matrix elementwise add)
        %(ccode_type)s *px = (%(ccode_type)s*)PyArray_DATA(%(x)s);
        %(ccode_type)s *py = (%(ccode_type)s*)PyArray_DATA(%(y)s);
        %(ccode_type)s *pz = (%(ccode_type)s*)PyArray_DATA(%(z)s);


        %(ccode_mulfunc)s(PyArray_DIMS(%(x)s)[0]*PyArray_DIMS(%(x)s)[1], px, py, pz);
        """

        return c_code % locals()






    def c_code_cache_version(self):
        return (1, 0, 0)