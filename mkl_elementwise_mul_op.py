from theano import tensor, gof
import numpy


class ElementwiseMultiply(gof.Op):
    __props__ = ()

    def make_node(self, *inputs):
        # Validate the inputs' type
        inputs = list(map(tensor.as_tensor_variable, inputs))

        if len(inputs) != 2:
            raise TypeError(
                'ElementwiseMultiply: 2 arguments required, %d given'
                % len(inputs))

        if inputs[0].ndim > 5:
            raise TypeError("ElementwiseMultiply: A's dimension is to large")
        if inputs[1].ndim > 5:
            raise TypeError("ElementwiseMultiply: B's dimension is to large")
        if inputs[0].ndim != inputs[1].ndim:
            raise TypeError(
                "ElementwiseMultiply: \
                elements in A and B should have same dimension")

        if inputs[0].dtype != inputs[1].dtype:
            raise TypeError(
                'ElementwiseMultiply: \
                elements in A and B should have same dtype')

        output_var = inputs[0].type()

        return gof.Apply(self, inputs, [output_var])

    def c_headers(self):
        headers = ['<mkl.h>']
        return headers

    def c_code(self, node, name, inputs, outputs, sub):
        if node.inputs[0].dtype is 'float32':
            ccode_type = 'float'
            ccode_addfunc = 'vsMul'
        elif node.inputs[0].dtype is 'float64':
            ccode_type = 'double'
            ccode_addfunc = 'vdMul'
        else:
            raise TypeError('ElementwiseMultiply: dtype %s is not supported.'
                            % (node.inputs[0].dtype))

        x, y = inputs
        z, = outputs

        dim_x = node.inputs[0].ndim

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
            %(z)s = (PyArrayObject*)PyArray_EMPTY(%(dim_x)s,
                                                PyArray_DIMS(%(x)s),
                                                %(typenum_z)s,
                                                0);

            if (!%(z)s) {
                %(fail)s;
            }
        }

        //Get the number of elements to be calculated from PyArray_DIMS
        int i;
        int cal_num = 1;
        for (i = 0; i < %(dim_x)s; i++){
            cal_num *= PyArray_DIMS(%(x)s)[i];
        }

        //Perform the MKL Mul (elementwise mul)
        %(ccode_type)s *px = (%(ccode_type)s*)PyArray_DATA(%(x)s);
        %(ccode_type)s *py = (%(ccode_type)s*)PyArray_DATA(%(y)s);
        %(ccode_type)s *pz = (%(ccode_type)s*)PyArray_DATA(%(z)s);

        %(ccode_addfunc)s(cal_num, px, py, pz);
        """

        return c_code % locals()

    def grad(self, inp, grads):
        x, y = inp
        gz, = grads
        return ElementwiseMultiply()(gz, y), ElementwiseMultiply()(gz, x)

    def c_code_cache_version(self):
        return (1, 0, 0)
