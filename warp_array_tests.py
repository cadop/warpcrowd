'''
Examples of using basic functions like loops and arrays
'''

import warp as wp

wp.init()
wp.set_device("cpu")

########################################################
# ### Test for making a full array and assigning
row_cnt = 1000
twoD_floats = wp.full((row_cnt, 2), 9999999.0, dtype=float, device='cpu') # would be nice to have inf
# twoD_floats[0,0] = 1 
####### >>> TypeError: 'array' object does not support item assignment


########################################################
twoD_vec = wp.full((row_cnt), 9999999.0, dtype=wp.vec2, device='cpu') # would be nice to have inf
vec2Data = wp.vec2(666.0,666.0)
# vec2Data[0] = vec2Data 
####### >>> TypeError: incompatible types, vec2f instance instead of c_float instance
########################################################


####################################################################################
twoD_vec_shape = wp.full(shape=(1024,2), value=9999999.0, dtype=wp.vec2, device='cpu')
# twoD_vec_shape[0,0] = wp.vec2(666.0,666.0) 
####### >>> TypeError: 'array' object does not support item assignment


@wp.kernel
def simple_kernel(a: wp.array(dtype=wp.vec3),
                  b: wp.array(dtype=wp.vec3),
                  c: wp.array(dtype=float)):

    # get thread index
    tid = wp.tid()

    # load two vec3s
    x = a[tid]
    y = b[tid]

    # Do this in a function instead
    # r = wp.dot(x, y)
    r = get_r(x, y)

    ######################################################################
    # Modify twoD_floats in kernel
    # twoD_floats[0,0] = 1.0
    ####### >>> 'array' object has no attribute 'type'
    #############################################################################

    # write result back to memory
    c[tid] = r



@wp.func
def get_r(x: wp.vec3,
          y: wp.vec3
          ):

    # Modify the x value
    x[0] = 1.0 + x[0]

    #############################################################################
    # Modify twoD_floats in func
    # twoD_floats[0,0] = 1 
    ####### >>> 'array' object has no attribute 'type'
    #############################################################################
    
    # compute the dot product between vectors
    r = wp.dot(x, y)

    return r

a = wp.full(shape=(1024), value=2.0, dtype=wp.vec3, device='cpu')
b = wp.full(shape=(1024), value=2.0, dtype=wp.vec3, device='cpu')
c = wp.full(shape=(1024), value=99999.0, dtype=float, device='cpu')


wp.launch(kernel=simple_kernel, # kernel to launch
          dim=1024,             # number of threads
          inputs=[a, b, c],     # parameters
          device="cpu")        # execution device

