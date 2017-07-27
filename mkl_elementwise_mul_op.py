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
            c_type = 'float'
            ccode_addfunc = 'vsMul'
        elif node.inputs[0].dtype is 'float64':
            c_type = 'double'
            ccode_addfunc = 'vdMul'
        else:
            raise TypeError('ElementwiseMultiply: dtype %s is not supported.'
                            % (node.inputs[0].dtype))

        x, y = inputs
        z, = outputs

        dim_x = node.inputs[0].ndim
        dim_y = node.inputs[1].ndim
        if dim_x > dim_y:
            dim_max = dim_x
            dim_min = dim_y
            max_input = x
            min_input = y
        elif dim_x < dim_y:
            dim_max = dim_y
            dim_min = dim_x
            max_input = y
            min_input = x
        else:
            dim_max = dim_x
            dim_min = dim_x
            max_input = x
            min_input = x

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
            %(z)s = (PyArrayObject*)PyArray_EMPTY(%(dim_max)s,
                                                PyArray_DIMS(%(max_input)s),
                                                %(typenum_z)s,
                                                0);

            if (!%(z)s) {
                %(fail)s;
            }
        }

        if (%(dim_x)s == %(dim_y)s){
            //Get the number of elements from PyArray_SIZE
            int cal_num = PyArray_SIZE(%(x)s);

            //Perform the MKL Add (elementwise add)
            %(c_type)s *px = (%(c_type)s*)PyArray_DATA(%(x)s);
            %(c_type)s *py = (%(c_type)s*)PyArray_DATA(%(y)s);
            %(c_type)s *pz = (%(c_type)s*)PyArray_DATA(%(z)s);

            %(ccode_addfunc)s(cal_num, px, py, pz);

        }else{
            //calculate the computation times
            int i;
            int loop_num = 1;
            for (i = 0; i < %(dim_max)s - %(dim_min)s; i++){
                loop_num *= PyArray_DIMS(%(max_input)s)[i];
            }

            //calculate the number of elements to be computed each time
            int cal_num = 1;
            for (i = 0; i < %(dim_min)s; i++){
                cal_num *= PyArray_DIMS(%(min_input)s)[i];
            }

            //Perform the MKL Mul (elementwise mul)
            %(c_type)s *pmax = (%(c_type)s*)PyArray_DATA(%(max_input)s);
            %(c_type)s *pmin = (%(c_type)s*)PyArray_DATA(%(min_input)s);
            %(c_type)s *pz = (%(c_type)s*)PyArray_DATA(%(z)s);

            for (i = 0; i < loop_num; i++){
                %(ccode_addfunc)s(cal_num, pmax, pmin, pz);
                pmax += cal_num;
                pz += cal_num;
            }
        }
        """

        return c_code % locals()

    def grad(self, inp, grads):
        x, y = inp
        gz, = grads
        return ElementwiseMultiply()(gz, y), ElementwiseMultiply()(gz, x)

    def c_code_cache_version(self):
        return (1, 0, 0)
