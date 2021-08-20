use std::ptr;

use num_complex::{Complex32, Complex64};
use ocl::{OclPrm, Queue};

use crate::{Error, VectorBuffer};

use typed_builder::TypedBuilder;

use clblast_sys::{CLBlastiCamin, CLBlastiDamin, CLBlastiSamin, CLBlastiZamin};

///  Index of absolute minimum value in a vector
/// Finds the index of a minimum (not necessarily the first if there are multiple) of the absolute values in the x vector. The resulting integer index is stored in the imin buffer.
#[derive(TypedBuilder)]
struct VectorAbsoluteMinIndex<'a, T: OclPrm> {
    /// OpenCL command queue associated with a context and device to execute the routine on.
    queue: &'a Queue,

    /// number of values to accumulate
    n: usize,

    // OpenCl buffer to store the output imin vector
    imin_vector: &'a VectorBuffer<T>,
    // OpenCl buffer to store the output x vector
    x_vector: &'a VectorBuffer<T>,

    /// Stride/increment of the output x vector. This value must be greater than 0.
    #[builder(default = 1)]
    x_stride: usize,
}

trait RunVectorAbsoluteMinIndex {
    unsafe fn run(self) -> Result<(), Error>;
}

fn assert_dimensions<'a, T: OclPrm>(params: &VectorAbsoluteMinIndex<'a, T>) {
    assert!(
        params.imin_vector.buffer.len() > params.n * params.x_stride,
        "x buffer is too short for n and x_stride"
    );
}

impl<'a> RunVectorAbsoluteMinIndex for VectorAbsoluteMinIndex<'a, f32> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastiSamin(
            self.n as u64,
            self.imin_vector.buffer.as_ptr(),
            self.imin_vector.offset as u64,
            self.x_vector.buffer.as_ptr(),
            self.x_stride as u64,
            self.x_vector.offset as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorAbsoluteMinIndex for VectorAbsoluteMinIndex<'a, f64> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastiDamin(
            self.n as u64,
            self.imin_vector.buffer.as_ptr(),
            self.imin_vector.offset as u64,
            self.x_vector.buffer.as_ptr(),
            self.x_stride as u64,
            self.x_vector.offset as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorAbsoluteMinIndex for VectorAbsoluteMinIndex<'a, Complex32> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastiCamin(
            self.n as u64,
            self.imin_vector.buffer.as_ptr(),
            self.imin_vector.offset as u64,
            self.x_vector.buffer.as_ptr(),
            self.x_stride as u64,
            self.x_vector.offset as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorAbsoluteMinIndex for VectorAbsoluteMinIndex<'a, Complex64> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastiZamin(
            self.n as u64,
            self.imin_vector.buffer.as_ptr(),
            self.imin_vector.offset as u64,
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
        let pro_que = ProQue::builder().src("").dims(30).build().unwrap();
        let x_vector = pro_que.create_buffer::<f32>().unwrap();
        let sum_buffer = pro_que.create_buffer::<f32>().unwrap();
        let x_vector = VectorBuffer::builder().buffer(x_vector).build();
        let imin_vector = VectorBuffer::builder().buffer(sum_buffer).build();
        let task = VectorAbsoluteMinIndex::builder()
            .queue(&pro_que.queue())
            .x_vector(&x_vector)
            .imin_vector(&imin_vector)
            .n(10)
            .build();
        unsafe { task.run().unwrap() }
    }
}
