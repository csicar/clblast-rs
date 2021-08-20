use std::marker::PhantomData;
use std::ptr;

use clblast_sys::cl_double2;
use clblast_sys::cl_float2;
use clblast_sys::CLBlastCgemm;
use clblast_sys::CLBlastDgemm;
use clblast_sys::CLBlastHgemm;
use clblast_sys::CLBlastLayout;
use clblast_sys::CLBlastLayout__CLBlastLayoutColMajor;
use clblast_sys::CLBlastLayout__CLBlastLayoutRowMajor;
use clblast_sys::CLBlastSgemm;
use clblast_sys::CLBlastSide;
use clblast_sys::CLBlastSide__CLBlastSideLeft;
use clblast_sys::CLBlastSide__CLBlastSideRight;
use clblast_sys::CLBlastTranspose__CLBlastTransposeConjugate;
use clblast_sys::CLBlastTranspose__CLBlastTransposeNo;
use clblast_sys::CLBlastTranspose__CLBlastTransposeYes;
use clblast_sys::CLBlastTriangle__CLBlastTriangleLower;
use clblast_sys::CLBlastTriangle__CLBlastTriangleUpper;
use clblast_sys::CLBlastZgemm;
use num_complex::Complex32;
use num_complex::Complex64;
use ocl::ffi::c_uint;
use ocl::Buffer;
use ocl::OclPrm;
use ocl::Queue;
use typed_builder::TypedBuilder;

use crate::Error;
use crate::MatrixBuffer;
use crate::MatrixLayout;
use crate::MatrixTranspose;
use crate::NeutralAdd;
use crate::NeutralMul;
use crate::ReprSys;

#[derive(TypedBuilder)]
pub struct Gemm<'a, T, L>
where
    T: OclPrm + NeutralAdd + NeutralMul,
    L: MatrixLayout,
{
    //Queue
    queue: &'a Queue,

    // Matrices
    a: &'a MatrixBuffer<T, L>,
    b: &'a MatrixBuffer<T, L>,
    c: &'a mut MatrixBuffer<T, L>,

    // factors
    #[builder(default=NeutralMul::ONE)]
    alpha: T,
    #[builder(default=NeutralAdd::ZERO)]
    beta: T,

    // transpose
    #[builder(default=MatrixTranspose::No)]
    transpose_a: MatrixTranspose,
    #[builder(default=MatrixTranspose::No)]
    transpose_b: MatrixTranspose,
}

fn assert_dimensions<'a, T: OclPrm + NeutralAdd + NeutralMul, L: MatrixLayout>(
    params: &Gemm<'a, T, L>,
) -> (usize, usize, usize) {
    assert_eq!(params.a.columns, params.b.rows, "a.columns /= b.rows (k)");
    let k = params.a.columns;

    assert_eq!(
        params.b.columns, params.c.columns,
        "b.columns /= c.columns (n)"
    );
    let n = params.b.columns;

    assert_eq!(params.c.rows, params.a.rows, "c.columns /= a.rows (m)");
    let m = params.c.columns;

    (k, n, m)
}
pub trait RunGemm {
    unsafe fn run(self) -> Result<(), Error>;
}

impl<'a, L> RunGemm for Gemm<'a, f32, L>
where
    L: MatrixLayout,
{
    unsafe fn run(self) -> Result<(), Error> {
        let (k, n, m) = assert_dimensions(&self);

        let res = CLBlastSgemm(
            self.a.layout.to_c(),
            self.transpose_a.to_c(),
            self.transpose_b.to_c(),
            m as u64,
            n as u64,
            k as u64,
            self.alpha,
            self.a.buffer.as_ptr(),
            self.a.offset as u64,
            k as u64,
            self.b.buffer.as_ptr(),
            self.b.offset as u64,
            n as u64,
            self.beta,
            self.c.buffer.as_ptr(),
            self.c.offset as u64,
            n as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a, L> RunGemm for Gemm<'a, f64, L>
where
    L: MatrixLayout,
{
    unsafe fn run(self) -> Result<(), Error> {
        let (k, n, m) = assert_dimensions(&self);

        let res = CLBlastDgemm(
            self.a.layout.to_c(),
            self.transpose_a.to_c(),
            self.transpose_b.to_c(),
            m as u64,
            n as u64,
            k as u64,
            self.alpha,
            self.a.buffer.as_ptr(),
            self.a.offset as u64,
            k as u64,
            self.b.buffer.as_ptr(),
            self.b.offset as u64,
            n as u64,
            self.beta,
            self.c.buffer.as_ptr(),
            self.c.offset as u64,
            n as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a, L> RunGemm for Gemm<'a, Complex32, L>
where
    L: MatrixLayout,
{
    unsafe fn run(self) -> Result<(), Error> {
        let (k, n, m) = assert_dimensions(&self);
        let alpha = cl_float2 {
            s: [self.alpha.re, self.alpha.im],
        };
        let alpha = cl_float2 {
            s: [self.alpha.re, self.alpha.im],
        };
        let res = CLBlastCgemm(
            self.a.layout.to_c(),
            self.transpose_a.to_c(),
            self.transpose_b.to_c(),
            m as u64,
            n as u64,
            k as u64,
            self.alpha.to_c(),
            self.a.buffer.as_ptr(),
            self.a.offset as u64,
            k as u64,
            self.b.buffer.as_ptr(),
            self.b.offset as u64,
            n as u64,
            self.beta.to_c(),
            self.c.buffer.as_ptr(),
            self.c.offset as u64,
            n as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

impl<'a, L> RunGemm for Gemm<'a, Complex64, L>
where
    L: MatrixLayout,
{
    unsafe fn run(self) -> Result<(), Error> {
        let (k, n, m) = assert_dimensions(&self);

        let res = CLBlastZgemm(
            self.a.layout.to_c(),
            self.transpose_a.to_c(),
            self.transpose_b.to_c(),
            m as u64,
            n as u64,
            k as u64,
            self.alpha.to_c(),
            self.a.buffer.as_ptr(),
            self.a.offset as u64,
            k as u64,
            self.b.buffer.as_ptr(),
            self.b.offset as u64,
            n as u64,
            self.beta.to_c(),
            self.c.buffer.as_ptr(),
            self.c.offset as u64,
            n as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

/// Computes `C := alpha * A * B + beta * C`
///
/// # Arguments
/// - Matrix A: K⨯M (K Wide, M High)
/// - Matrix B: N⨯K (N Wide, K High)
/// - Matrix C: M⨯N (N Wide, M High)
///
/// For details see: https://cnugteren.github.io/tutorial/pages/page2.html
///
/// # Example
/// ```no_run
/// use ocl::ProQue;
/// use crate::clblast::gemm::{MultiplicationExecutor, RunMatrixMultiplication};
/// use crate::clblast::{MatrixBuffer, LayoutRowMajor};
/// let pro_que = ProQue::builder().src("").dims(1).build().unwrap();
/// let k = 40;
/// let m = 20;
/// let n = 10;
/// let a_matrix = MatrixBuffer::new_default(&pro_que, k, m, 1.0, LayoutRowMajor);
/// let b_matrix = MatrixBuffer::new_default(&pro_que, n, k, 1.0, LayoutRowMajor);
/// let mut c_matrix = MatrixBuffer::new_default(&pro_que, n, m, 1.0, LayoutRowMajor);
/// let task = pro_que
///         .queue()
///         .gemm()
///         .a(&a_matrix)
///         .b(&b_matrix)
///         .c(&mut c_matrix)
///         .build();
/// # println!("still fine");
/// # unsafe { task.run() }.unwrap();
/// # println!("still still fine");
/// # drop(pro_que);
/// # drop(a_matrix);
/// # drop(b_matrix);
/// # drop(c_matrix);
/// # println!("fine still fine");
/// ```

#[cfg(test)]
mod test {
    use ocl::{flags, ProQue};
    use rand::prelude::*;
    use rand_chacha::ChaCha20Rng;
    use std::error::Error;
    use std::time::Instant;

    use crate::LayoutRowMajor;

    use super::*;

    fn reference_implementation(
        no_samples: usize,
        no_streams: usize,
        samples: &Vec<f32>,
        matrix: &Vec<f32>,
    ) -> Vec<f32> {
        let mut out = vec![0.0; no_samples * no_streams];

        for x in 0..no_samples {
            for y in 0..no_streams {
                let mut sum = 0.0_f64;
                for k in 0..no_streams {
                    sum += (samples[x + no_samples * k] * matrix[y * no_streams + k]) as f64;
                }
                out[x + y * no_samples] = sum as f32;
            }
        }
        out
    }
    #[test]
    fn test_doc_comment() {
        use ocl::ProQue;
        let pro_que = ProQue::builder().src("").dims(1).build().unwrap();
        let k = 40;
        let m = 20;
        let n = 10;
        let a_matrix = MatrixBuffer::new_default(&pro_que, k, m, 1.0, LayoutRowMajor);
        let b_matrix = MatrixBuffer::new_default(&pro_que, n, k, 1.0, LayoutRowMajor);
        let mut c_matrix = MatrixBuffer::new_default(&pro_que, n, m, 1.0, LayoutRowMajor);
        let task = Gemm::builder()
            .queue(&pro_que.queue())
            .a(&a_matrix)
            .b(&b_matrix)
            .c(&mut c_matrix)
            .build();
        unsafe { task.run().unwrap() }
    }

    #[test]
    fn test_reference() {
        let res = reference_implementation(
            3,
            5,
            &vec![
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
                7.0, 8.0, 9.0, //
                7.0, 8.0, 9.0, //
            ],
            &vec![
                0.0, 2.0, 1.0, 0.0, 2.0, //
                0.0, 1.0, 1.0, 2.0, 3.0, //
                1.0, 2.0, 3.0, 4.0, 5.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 3.0, 9.0, 2.0, //
            ],
        );
        println!("{:?}", res);
        let expected_result = vec![
            29.0, 34.0, 39.0, 46.0, 53.0, 60.0, 93.0, 108.0, 123.0, 0.0, 0.0, 0.0, 102.0, 117.0,
            132.0,
        ];
        assert_eq!(res, expected_result);
    }

    #[test]
    fn test_gemm() -> Result<(), Box<dyn Error>> {
        let src = r#"
            __kernel void add(__global float* buffer, float scalar) {
                buffer[get_global_id(0)] += scalar;
            }
        "#;
        let no_streams = 64;
        let no_samples = 64;
        let mut rng = ChaCha20Rng::seed_from_u64(1337110);

        let pro_que = ProQue::builder()
            .src(src)
            .dims(no_streams * no_samples)
            .build()
            .unwrap();

        let a_val = (0..no_streams * no_streams)
            .map(|_| rng.gen::<f32>())
            .collect::<Vec<_>>();

        let a_buffer = pro_que
            .buffer_builder()
            .flags(flags::MEM_READ_WRITE)
            .len(no_streams * no_streams)
            .build()
            .unwrap();

        a_buffer.write(&a_val[..]).enq().unwrap();

        let a = MatrixBuffer::new(no_streams, no_streams, a_buffer, LayoutRowMajor);

        let b_val = (0..no_streams * no_samples)
            .map(|_| rng.gen::<f32>())
            .collect::<Vec<_>>();

        let b_buffer = pro_que
            .buffer_builder()
            .flags(flags::MEM_READ_WRITE)
            .len(no_streams * no_samples)
            .fill_val(4f32)
            .build()
            .unwrap();

        b_buffer.write(&b_val[..]).enq().unwrap();

        let b = MatrixBuffer::new(no_samples, no_streams, b_buffer, LayoutRowMajor);

        let c_buffer = pro_que
            .buffer_builder()
            .flags(flags::MEM_READ_WRITE)
            .len(no_streams * no_samples)
            .fill_val(0f32)
            .build()
            .unwrap();
        let mut c = MatrixBuffer::new(no_samples, no_streams, c_buffer, LayoutRowMajor);

        let before = Instant::now();
        println!("run..");
        let task = Gemm::builder().queue(&pro_que.queue()).a(&a).b(&b).c(&mut c).build();
        unsafe { task.run()? };

        let mut c_dat = vec![0.0; no_streams * no_samples];
        c.buffer.read(&mut c_dat[..]).enq().unwrap();

        println!("gpu: {:?} {:?}", &c_dat[0..10], before.elapsed());
        let reference_result = reference_implementation(no_samples, no_streams, &b_val, &a_val);
        println!("ref: {:?}", &reference_result[0..10]);
        assert_eq!(c_dat.len(), reference_result.len());
        c_dat
            .iter()
            .zip(reference_result)
            .for_each(|(&res, ref_res)| {
                assert!((res - ref_res) < 0.1, "{} ~= {}", res, ref_res);
            });
        Ok(())
    }
}
