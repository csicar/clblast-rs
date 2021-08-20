use std::ptr;

use num_complex::{Complex32, Complex64};
use ocl::{OclPrm, Queue};

use crate::{Error, VectorBuffer};

use typed_builder::TypedBuilder;

use clblast_sys::{CLBlastCcopy, CLBlastDcopy, CLBlastScopy, CLBlastZcopy};

/// Copies the contents of vector x into vector y.
#[derive(TypedBuilder)]
struct VectorCopy<'a, T: OclPrm> {
    /// OpenCL command queue associated with a context and device to execute the routine on.
    queue: &'a Queue,

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

fn assert_dimensions<'a, T: OclPrm>(params: &VectorCopy<'a, T>) {
    assert!(
        params.x_vector.buffer.len() > params.n * params.x_stride,
        "x buffer is too short for n and x_stride"
    );
    assert!(
        params.y_vector.buffer.len() > params.n * params.y_stride,
        "y buffer is too short for n and y_stride"
    );
}

impl<'a> RunVectorCopy for VectorCopy<'a, f32> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastScopy(
            self.n as u64,
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

impl<'a> RunVectorCopy for VectorCopy<'a, f64> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastDcopy(
            self.n as u64,
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

impl<'a> RunVectorCopy for VectorCopy<'a, Complex32> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastCcopy(
            self.n as u64,
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

impl<'a> RunVectorCopy for VectorCopy<'a, Complex64> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastZcopy(
            self.n as u64,
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
        let task = VectorCopy::builder()
            .queue(&pro_que.queue())
            .x_vector(&a_matrix)
            .y_vector(&b_matrix)
            .n(10)
            .build();
        unsafe { task.run().unwrap() }
    }
}
