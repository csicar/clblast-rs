use std::ptr;

use num_complex::{Complex32, Complex64};
use ocl::{OclPrm, Queue};

use crate::{Error, VectorBuffer};

use typed_builder::TypedBuilder;

use clblast_sys::{CLBlastScnrm2, CLBlastDnrm2, CLBlastSnrm2, CLBlastDznrm2};

/// Accumulates the square of n elements in the x vector and takes the square root. The resulting L2 norm is stored in the nrm2 buffer.
#[derive(TypedBuilder)]
struct VectorEuclidianNorm<'a, T: OclPrm> {
    /// OpenCL command queue associated with a context and device to execute the routine on.
    queue: &'a Queue,

    /// number of values to swap
    n: usize,

    // OpenCl buffer to store the output nrm vector
    nrm2_vector: &'a VectorBuffer<T>,
    // OpenCl buffer to store the output x vector
    x_vector: &'a VectorBuffer<T>,

    /// Stride/increment of the output y vector. This value must be greater than 0.
    #[builder(default = 1)]
    x_stride: usize,
}

trait RunVectorEuclidianNorm {
    unsafe fn run(self) -> Result<(), Error>;
}

fn assert_dimensions<'a, T: OclPrm>(params: &VectorEuclidianNorm<'a, T>) {
    assert!(
        params.x_vector.buffer.len() > params.n * params.x_stride,
        "y buffer is too short for n and y_stride"
    );
}

impl<'a> RunVectorEuclidianNorm for VectorEuclidianNorm<'a, f32> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastSnrm2(
            self.n as u64,
            self.nrm2_vector.buffer.as_ptr(),
            self.nrm2_vector.offset as u64,
            self.x_vector.buffer.as_ptr(),
            self.x_vector.offset as u64,
            self.x_stride as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorEuclidianNorm for VectorEuclidianNorm<'a, f64> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastDnrm2(
            self.n as u64,
            self.nrm2_vector.buffer.as_ptr(),
            self.nrm2_vector.offset as u64,
            self.x_vector.buffer.as_ptr(),
            self.x_vector.offset as u64,
            self.x_stride as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorEuclidianNorm for VectorEuclidianNorm<'a, Complex32> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastScnrm2(
            self.n as u64,
            self.nrm2_vector.buffer.as_ptr(),
            self.nrm2_vector.offset as u64,
            self.x_vector.buffer.as_ptr(),
            self.x_vector.offset as u64,
            self.x_stride as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorEuclidianNorm for VectorEuclidianNorm<'a, Complex64> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastDznrm2(
            self.n as u64,
            self.nrm2_vector.buffer.as_ptr(),
            self.nrm2_vector.offset as u64,
            self.x_vector.buffer.as_ptr(),
            self.x_vector.offset as u64,
            self.x_stride as u64,
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
        let x_buffer = pro_que.create_buffer::<f32>().unwrap();
        let nrm2_buffer = pro_que.create_buffer::<f32>().unwrap();
        let x_vector = VectorBuffer::builder().buffer(x_buffer).build();
        let nrm2_vector = VectorBuffer::builder().buffer(nrm2_buffer).build();
        let task = VectorEuclidianNorm::builder()
            .queue(&pro_que.queue())
            .x_vector(&x_vector)
            .nrm2_vector(&nrm2_vector)
            .n(10)
            .build();
        unsafe { task.run().unwrap() }
    }
}
