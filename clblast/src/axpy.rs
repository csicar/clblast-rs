use std::ptr;

use num_complex::{Complex32, Complex64};
use ocl::{OclPrm, Queue};

use crate::{Error, ReprSys, VectorBuffer};

use typed_builder::TypedBuilder;

use clblast_sys::{CLBlastCaxpy, CLBlastDaxpy, CLBlastSaxpy, CLBlastZaxpy};

/// Performs the operation `$y = alpha * x + y$`, in which `x` and `y` are vectors and `alpha` is a scalar constant.
#[derive(TypedBuilder)]
struct VectorAxpy<'a, T: OclPrm> {
    /// OpenCL command queue associated with a context and device to execute the routine on.
    queue: &'a Queue,

    ///  Input scalar constant `alpha`
    alpha: T,

    /// number of values to swap
    n: usize,

    // OpenCl buffer to store the output x vector
    x_vector: &'a VectorBuffer<T>,
    // OpenCl buffer to store the output y vector
    y_vector: &'a VectorBuffer<T>,

    /// Stride/increment of the output x vector. This value must be greater than 0.
    #[builder(default = 1)]
    x_stride: usize,
    /// Stride/increment of the output y vector. This value must be greater than 0.
    #[builder(default = 1)]
    y_stride: usize,
}

trait RunVectorCopy {
    unsafe fn run(self) -> Result<(), Error>;
}

fn assert_dimensions<'a, T: OclPrm>(params: &VectorAxpy<'a, T>) {
    assert!(
        params.x_vector.buffer.len() > params.n * params.x_stride,
        "x buffer is too short for n and x_stride"
    );
    assert!(
        params.y_vector.buffer.len() > params.n * params.y_stride,
        "y buffer is too short for n and y_stride"
    );
}

impl<'a> RunVectorCopy for VectorAxpy<'a, f32> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastSaxpy(
            self.n as u64,
            self.alpha,
            self.x_vector.buffer.as_ptr(),
            self.x_vector.offset as u64,
            self.x_stride as u64,
            self.y_vector.buffer.as_ptr(),
            self.y_vector.offset as u64,
            self.y_stride as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorCopy for VectorAxpy<'a, f64> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastDaxpy(
            self.n as u64,
            self.alpha,
            self.x_vector.buffer.as_ptr(),
            self.x_vector.offset as u64,
            self.x_stride as u64,
            self.y_vector.buffer.as_ptr(),
            self.y_vector.offset as u64,
            self.y_stride as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorCopy for VectorAxpy<'a, Complex32> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastCaxpy(
            self.n as u64,
            self.alpha.to_c(),
            self.x_vector.buffer.as_ptr(),
            self.x_vector.offset as u64,
            self.x_stride as u64,
            self.y_vector.buffer.as_ptr(),
            self.y_vector.offset as u64,
            self.y_stride as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorCopy for VectorAxpy<'a, Complex64> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastZaxpy(
            self.n as u64,
            self.alpha.to_c(),
            self.x_vector.buffer.as_ptr(),
            self.x_vector.offset as u64,
            self.x_stride as u64,
            self.y_vector.buffer.as_ptr(),
            self.y_vector.offset as u64,
            self.y_stride as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::VectorBuffer;

    #[test]
    fn test_float() {
        use ocl::ProQue;
        let pro_que = ProQue::builder().src("").dims(20).build().unwrap();
        let a_buffer = pro_que.create_buffer::<f32>().unwrap();
        let b_buffer = pro_que.create_buffer::<f32>().unwrap();
        let a_matrix = VectorBuffer::builder().buffer(a_buffer).build();
        let b_matrix = VectorBuffer::builder().buffer(b_buffer).build();
        let task = VectorAxpy::builder()
            .queue(&pro_que.queue())
            .x_vector(&a_matrix)
            .y_vector(&b_matrix)
            .alpha(2.0)
            .n(10)
            .build();
        unsafe { task.run().unwrap() }
    }
}
