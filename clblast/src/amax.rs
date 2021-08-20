use std::ptr;

use num_complex::{Complex32, Complex64};
use ocl::{OclPrm, Queue};

use crate::{Error, VectorBuffer};

use typed_builder::TypedBuilder;

use clblast_sys::{CLBlastiSamax, CLBlastiDamax, CLBlastiCamax, CLBlastiZamax};

///  Index of absolute maximum value in a vector
/// Finds the index of a maximum (not necessarily the first if there are multiple) of the absolute values in the x vector. The resulting integer index is stored in the imax buffer.
#[derive(TypedBuilder)]
struct VectorAbsoluteMaxIndex<'a, T: OclPrm> {
    /// OpenCL command queue associated with a context and device to execute the routine on.
    queue: &'a Queue,

    /// number of values to accumulate
    n: usize,

    // OpenCl buffer to store the output imax vector
    imax_vector: &'a VectorBuffer<T>,
    // OpenCl buffer to store the output x vector
    x_vector: &'a VectorBuffer<T>,

    /// Stride/increment of the output x vector. This value must be greater than 0.
    #[builder(default = 1)]
    x_stride: usize,
}

trait RunVectorAbsoluteMaxIndex {
    unsafe fn run(self) -> Result<(), Error>;
}

fn assert_dimensions<'a, T: OclPrm>(params: &VectorAbsoluteMaxIndex<'a, T>) {
    assert!(
        params.imax_vector.buffer.len() > params.n * params.x_stride,
        "x buffer is too short for n and x_stride"
    );
}

impl<'a> RunVectorAbsoluteMaxIndex for VectorAbsoluteMaxIndex<'a, f32> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastiSamax(
            self.n as u64,
            self.imax_vector.buffer.as_ptr(),
            self.imax_vector.offset as u64,
            self.x_vector.buffer.as_ptr(),
            self.x_stride as u64,
            self.x_vector.offset as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorAbsoluteMaxIndex for VectorAbsoluteMaxIndex<'a, f64> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastiDamax(
            self.n as u64,
            self.imax_vector.buffer.as_ptr(),
            self.imax_vector.offset as u64,
            self.x_vector.buffer.as_ptr(),
            self.x_stride as u64,
            self.x_vector.offset as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorAbsoluteMaxIndex for VectorAbsoluteMaxIndex<'a, Complex32> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastiCamax(
            self.n as u64,
            self.imax_vector.buffer.as_ptr(),
            self.imax_vector.offset as u64,
            self.x_vector.buffer.as_ptr(),
            self.x_stride as u64,
            self.x_vector.offset as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorAbsoluteMaxIndex for VectorAbsoluteMaxIndex<'a, Complex64> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastiZamax(
            self.n as u64,
            self.imax_vector.buffer.as_ptr(),
            self.imax_vector.offset as u64,
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
        let imax_vector = VectorBuffer::builder().buffer(sum_buffer).build();
        let task = VectorAbsoluteMaxIndex::builder()
            .queue(&pro_que.queue())
            .x_vector(&x_vector)
            .imax_vector(&imax_vector)
            .n(10)
            .build();
        unsafe { task.run().unwrap() }
    }
}
