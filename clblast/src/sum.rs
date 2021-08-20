use std::ptr;

use num_complex::{Complex32, Complex64};
use ocl::{OclPrm, Queue};

use crate::{Error, VectorBuffer};

use typed_builder::TypedBuilder;

use clblast_sys::{CLBlastScsum, CLBlastDsum, CLBlastSsum, CLBlastDzsum};

/// Sum of values in a vector (non-BLAS function)
/// Accumulates the values of n elements in the x vector. The results are stored in the sum buffer. This routine is the non-absolute version of the xASUM BLAS routine.
#[derive(TypedBuilder)]
struct VectorSum<'a, T: OclPrm> {
    /// OpenCL command queue associated with a context and device to execute the routine on.
    queue: &'a Queue,

    /// number of values to accumulate
    n: usize,

    // OpenCl buffer to store the output x vector
    sum_vector: &'a VectorBuffer<T>,
    // OpenCl buffer to store the output y vector
    x_vector: &'a VectorBuffer<T>,

    /// Stride/increment of the output x vector. This value must be greater than 0.
    #[builder(default = 1)]
    x_stride: usize,
}

trait RunVectorSum {
    unsafe fn run(self) -> Result<(), Error>;
}

fn assert_dimensions<'a, T: OclPrm>(params: &VectorSum<'a, T>) {
    assert!(
        params.sum_vector.buffer.len() > params.n * params.x_stride,
        "x buffer is too short for n and x_stride"
    );
}

impl<'a> RunVectorSum for VectorSum<'a, f32> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastSsum(
            self.n as u64,
            self.sum_vector.buffer.as_ptr(),
            self.sum_vector.offset as u64,
            self.x_vector.buffer.as_ptr(),
            self.x_stride as u64,
            self.x_vector.offset as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorSum for VectorSum<'a, f64> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastDsum(
            self.n as u64,
            self.sum_vector.buffer.as_ptr(),
            self.sum_vector.offset as u64,
            self.x_vector.buffer.as_ptr(),
            self.x_stride as u64,
            self.x_vector.offset as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorSum for VectorSum<'a, Complex32> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastScsum(
            self.n as u64,
            self.sum_vector.buffer.as_ptr(),
            self.sum_vector.offset as u64,
            self.x_vector.buffer.as_ptr(),
            self.x_stride as u64,
            self.x_vector.offset as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorSum for VectorSum<'a, Complex64> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastDzsum(
            self.n as u64,
            self.sum_vector.buffer.as_ptr(),
            self.sum_vector.offset as u64,
            self.x_vector.buffer.as_ptr(),
            self.x_stride as u64,
            self.x_vector.offset as u64,
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
        let x_vector = pro_que.create_buffer::<f32>().unwrap();
        let sum_buffer = pro_que.create_buffer::<f32>().unwrap();
        let x_vector = VectorBuffer::builder().buffer(x_vector).build();
        let a_sum = VectorBuffer::builder().buffer(sum_buffer).build();
        let task = VectorSum::builder()
            .queue(&pro_que.queue())
            .x_vector(&x_vector)
            .sum_vector(&a_sum)
            .n(10)
            .build();
        unsafe { task.run().unwrap() }
    }
}
