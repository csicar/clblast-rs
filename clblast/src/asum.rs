use std::ptr;

use num_complex::{Complex32, Complex64};
use ocl::{OclPrm, Queue};

use crate::{Error, VectorBuffer};

use typed_builder::TypedBuilder;

use clblast_sys::{CLBlastScasum, CLBlastDasum, CLBlastSasum, CLBlastDzasum};

/// Absolute sum of values in a vector
/// Accumulates the absolute value of n elements in the x vector. The results are stored in the asum buffer.
#[derive(TypedBuilder)]
struct VectorAbsoluteSum<'a, T: OclPrm> {
    /// OpenCL command queue associated with a context and device to execute the routine on.
    queue: &'a Queue,

    /// number of values to accumulate
    n: usize,

    // OpenCl buffer to store the output x vector
    asum_vector: &'a VectorBuffer<T>,
    // OpenCl buffer to store the output y vector
    x_vector: &'a VectorBuffer<T>,

    /// Stride/increment of the output x vector. This value must be greater than 0.
    #[builder(default = 1)]
    x_stride: usize,
}

trait RunVectorAbsoluteSum {
    unsafe fn run(self) -> Result<(), Error>;
}

fn assert_dimensions<'a, T: OclPrm>(params: &VectorAbsoluteSum<'a, T>) {
    assert!(
        params.asum_vector.buffer.len() > params.n * params.x_stride,
        "x buffer is too short for n and x_stride"
    );
}

impl<'a> RunVectorAbsoluteSum for VectorAbsoluteSum<'a, f32> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastSasum(
            self.n as u64,
            self.asum_vector.buffer.as_ptr(),
            self.asum_vector.offset as u64,
            self.x_vector.buffer.as_ptr(),
            self.x_stride as u64,
            self.x_vector.offset as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorAbsoluteSum for VectorAbsoluteSum<'a, f64> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastDasum(
            self.n as u64,
            self.asum_vector.buffer.as_ptr(),
            self.asum_vector.offset as u64,
            self.x_vector.buffer.as_ptr(),
            self.x_stride as u64,
            self.x_vector.offset as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorAbsoluteSum for VectorAbsoluteSum<'a, Complex32> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastScasum(
            self.n as u64,
            self.asum_vector.buffer.as_ptr(),
            self.asum_vector.offset as u64,
            self.x_vector.buffer.as_ptr(),
            self.x_stride as u64,
            self.x_vector.offset as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorAbsoluteSum for VectorAbsoluteSum<'a, Complex64> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastDzasum(
            self.n as u64,
            self.asum_vector.buffer.as_ptr(),
            self.asum_vector.offset as u64,
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
        let asum_buffer = pro_que.create_buffer::<f32>().unwrap();
        let x_vector = VectorBuffer::builder().buffer(x_vector).build();
        let a_sum = VectorBuffer::builder().buffer(asum_buffer).build();
        let task = VectorAbsoluteSum::builder()
            .queue(&pro_que.queue())
            .x_vector(&x_vector)
            .asum_vector(&a_sum)
            .n(10)
            .build();
        unsafe { task.run().unwrap() }
    }
}
