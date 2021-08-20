use std::ptr;

use num_complex::{Complex32, Complex64};
use ocl::{OclPrm, Queue};

use crate::{Error, ReprSys, VectorBuffer};

use typed_builder::TypedBuilder;

use clblast_sys::{CLBlastCscal, CLBlastCswap, CLBlastDscal, CLBlastDswap, CLBlastSscal, CLBlastSswap, CLBlastZscal, CLBlastZswap};

/// Multiplies `n` elements of vector `x` by a scalar constant `alpha`.
#[derive(TypedBuilder)]
struct VectorScale<'a, T: OclPrm> {
    /// OpenCL command queue associated with a context and device to execute the routine on.
    queue: &'a Queue,

    /// number of values to scale
    n: usize,

    /// scaling constant `alpha`
    alpha: T,

    // OpenCl buffer to store the output x vector
    x_vector: &'a VectorBuffer<T>,

    /// Stride/increment of the output x vector. This value must be greater than 0.
    #[builder(default = 1)]
    x_stride: usize,
}

trait RunVectorScale {
    unsafe fn run(self) -> Result<(), Error>;
}

fn assert_dimensions<'a, T: OclPrm>(params: &VectorScale<'a, T>) {
    assert!(
        params.x_vector.buffer.len() > params.n * params.x_stride,
        "x buffer is too short for n and x_stride"
    );
}

impl<'a> RunVectorScale for VectorScale<'a, f32> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastSscal(
            self.n as u64,
            self.alpha,
            self.x_vector.buffer.as_ptr(),
            self.x_vector.offset as u64,
            self.x_stride as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorScale for VectorScale<'a, f64> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastDscal(
            self.n as u64,
            self.alpha,
            self.x_vector.buffer.as_ptr(),
            self.x_vector.offset as u64,
            self.x_stride as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorScale for VectorScale<'a, Complex32> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastCscal(
          self.n as u64,
          self.alpha.to_c(),
          self.x_vector.buffer.as_ptr(),
          self.x_vector.offset as u64,
          self.x_stride as u64,
          &mut self.queue.as_ptr(),
          &mut ptr::null_mut(),
      );

        Error::from_c_either(res)
    }
}

impl<'a> RunVectorScale for VectorScale<'a, Complex64> {
    unsafe fn run(self) -> Result<(), Error> {
        assert_dimensions(&self);

        let res = CLBlastZscal(
          self.n as u64,
          self.alpha.to_c(),
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
        let a_buffer = pro_que.create_buffer::<f32>().unwrap();
        let a_matrix = VectorBuffer::builder().buffer(a_buffer).build();
        let task = VectorScale::builder()
            .queue(&pro_que.queue())
            .alpha(10.0)
            .x_vector(&a_matrix)
            .n(10)
            .build();
        unsafe { task.run().unwrap() }
    }
}
