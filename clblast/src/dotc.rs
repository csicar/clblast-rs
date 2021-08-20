use std::ptr;

use num_complex::{Complex32, Complex64};
use ocl::{OclPrm, Queue};

use crate::{Error, VectorBuffer};

use typed_builder::TypedBuilder;

use clblast_sys::{CLBlastCdotc, CLBlastCdotu, CLBlastDdot, CLBlastSdot, CLBlastZdotc, CLBlastZdotu};

/// Multiplies n elements of the vectors x and y element-wise and accumulates the results. The sum is stored in the dot buffer.
#[derive(TypedBuilder)]
struct VectorDotConjucate<'a, T: OclPrm> {
    /// OpenCL command queue associated with a context and device to execute the routine on.
    queue: &'a Queue,

    /// number of values to swap
    n: usize,

    // OpenCl buffer to store the result in
    dot_buffer: &'a VectorBuffer<T>,

    // OpenCl buffer containing the x vector
    x_vector: &'a VectorBuffer<T>,
    // OpenCl buffer containing the y vector
    y_vector: &'a VectorBuffer<T>,

    /// Stride/increment of the output x vector. This value must be greater than 0.
    #[builder(default = 1)]
    x_stride: usize,
    /// Stride/increment of the output y vector. This value must be greater than 0.
    #[builder(default = 1)]
    y_stride: usize,
}

trait RunVectorDot {
    unsafe fn run(self) -> Result<(), Error>;
}

fn assert_dimensions<'a, T: OclPrm>(params: &VectorDotConjucate<'a, T>) {
    assert!(
        params.x_vector.buffer.len() > params.n * params.x_stride,
        "x buffer is too short for n and x_stride"
    );
    assert!(
        params.y_vector.buffer.len() > params.n * params.y_stride,
        "y buffer is too short for n and y_stride"
    );
}


/// called `xDOTC` in clblast: Dot product of two complex vectors, one conjugated
impl<'a> RunVectorDot for VectorDotConjucate<'a, Complex32> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastCdotc(
            self.n as u64,
            self.dot_buffer.buffer.as_ptr(),
            self.dot_buffer.offset as u64,
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

/// called `xDOTC` in clblast: Dot product of two complex vectors, one conjugated
impl<'a> RunVectorDot for VectorDotConjucate<'a, Complex64> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastZdotc(
            self.n as u64,
            self.dot_buffer.buffer.as_ptr(),
            self.dot_buffer.offset as u64,
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
        let a_buffer = pro_que.create_buffer::<Complex32>().unwrap();
        let b_buffer = pro_que.create_buffer::<Complex32>().unwrap();
        let dot_buffer = pro_que.create_buffer::<Complex32>().unwrap();
        let a_vector = VectorBuffer::builder().buffer(a_buffer).build();
        let b_vector = VectorBuffer::builder().buffer(b_buffer).build();
        let dot_vector = VectorBuffer::builder().buffer(dot_buffer).build();
        let task = VectorDotConjucate::builder()
            .queue(&pro_que.queue())
            .dot_buffer(&dot_vector)
            .x_vector(&a_vector)
            .y_vector(&b_vector)
            .n(10)
            .build();
        unsafe { task.run().unwrap() }
    }
}
