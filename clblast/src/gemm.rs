use std::ptr;

use num_complex::Complex32;
use num_complex::Complex64;
use ocl::OclPrm;
use ocl::Queue;
use typed_builder::TypedBuilder;

use clblast_sys::cl_float2;
use clblast_sys::CLBlastCgemm;
use clblast_sys::CLBlastDgemm;
use clblast_sys::CLBlastSgemm;
use clblast_sys::CLBlastZgemm;

use crate::Error;
use crate::MatrixBuffer;
use crate::MatrixLayout;
use crate::MatrixTranspose;
use crate::NeutralAdd;
use crate::NeutralMul;
use crate::ReprSys;

/// Computes `C := alpha * A * B + beta * C`
///
/// # Arguments
/// - Matrix A: K⨯M (K Wide, M High)
/// - Matrix B: N⨯K (N Wide, K High)
/// - Matrix C: M⨯N (N Wide, M High)
///
/// ![](https://cnugteren.github.io/tutorial/images/gemm1.png)
///
/// For details see: <https://cnugteren.github.io/tutorial/pages/page2.html>
///
/// See also: <https://petewarden.com/2015/10/25/an-engineers-guide-to-gemm/>
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

fn assert_dimensions<T: OclPrm + NeutralAdd + NeutralMul, L: MatrixLayout>(
    params: &Gemm<T, L>,
) -> (usize, usize, usize) {
    // Performs the matrix product C = alpha * A * B + beta * C,
    // in which A (m rows by k columns) and B (k rows by n columns) are two general rectangular input matrices,
    // C (m rows by n column) is the matrix to be updated, and alpha and beta are scalar values.
    // The matrices A and/or B can optionally be transposed before performing the operation.

    assert_eq!(params.a.columns, params.b.rows, "a.columns /= b.rows (k)");
    let k = params.a.columns;

    assert_eq!(
        params.b.columns, params.c.columns,
        "b.columns /= c.columns (n)"
    );
    let n = params.b.columns;

    assert_eq!(params.c.rows, params.a.rows, "c.columns /= a.rows (m)");
    let m = params.c.rows;

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
            L::default().to_c(),
            self.transpose_a.to_c(),
            self.transpose_b.to_c(),
            m as u64,
            n as u64,
            k as u64,
            self.alpha,
            self.a.buffer.as_ptr(),
            self.a.offset as u64,
            self.a.stride as u64,
            self.b.buffer.as_ptr(),
            self.b.offset as u64,
            self.b.stride as u64,
            self.beta,
            self.c.buffer.as_ptr(),
            self.c.offset as u64,
            self.c.stride as u64,
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
            self.a.stride as u64,
            self.b.buffer.as_ptr(),
            self.b.offset as u64,
            self.b.stride as u64,
            self.beta,
            self.c.buffer.as_ptr(),
            self.c.offset as u64,
            self.c.stride as u64,
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
            self.a.stride as u64,
            self.b.buffer.as_ptr(),
            self.b.offset as u64,
            self.b.stride as u64,
            self.beta.to_c(),
            self.c.buffer.as_ptr(),
            self.c.offset as u64,
            self.c.stride as u64,
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
            self.a.stride as u64,
            self.b.buffer.as_ptr(),
            self.b.offset as u64,
            self.b.stride as u64,
            self.beta.to_c(),
            self.c.buffer.as_ptr(),
            self.c.offset as u64,
            self.c.stride as u64,
            &mut self.queue.as_ptr(),
            &mut ptr::null_mut(),
        );

        Error::from_c_either(res)
    }
}

#[cfg(test)]
mod test {
    use std::error::Error;
    use std::time::Instant;

    use ocl::{flags, ProQue};
    use pretty_assertions as pretty;
    use rand::prelude::*;
    use rand_chacha::ChaCha20Rng;

    use crate::LayoutRowMajor;

    use super::*;

    fn reference_implementation(
        no_samples: usize,
        no_streams: usize,
        samples: &Vec<f32>,
        matrix: &Vec<f32>,
    ) -> Vec<Vec<f32>> {
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
        out.chunks_exact(no_samples).map(|r| r.to_vec()).collect()
    }

    fn format_as_rows(matrix: Vec<Vec<f32>>) -> Vec<String> {
        matrix
            .iter()
            .map(|row| {
                row.iter()
                    .map(|c| format!("{:>5.2}", c))
                    .collect::<Vec<String>>()
                    .join("\n")
            })
            .collect()
    }

    fn read_buffer_to_matrix(buf: MatrixBuffer<f32, LayoutRowMajor>) -> Vec<Vec<f32>> {
        let mut out = vec![0.0; buf.size()];
        buf.buffer().read(&mut out[..]).enq().unwrap();

        out.chunks_exact(buf.columns)
            .map(|row| row.to_vec())
            .collect()
    }

    fn compare_reference_impl(
        no_samples: usize,
        no_streams: usize,
        samples: &Vec<f32>,
        configuration: &Vec<f32>,
    ) {
        pretty::assert_eq!(no_streams * no_streams, configuration.len());
        pretty::assert_eq!(no_samples * no_streams, samples.len());
        let reference_result =
            reference_implementation(no_samples, no_streams, samples, configuration);

        let gpu_result = {
            let pro_que = ProQue::builder().src("").dims(21).build().unwrap();
            let k = no_streams;
            let m = no_streams;
            let n = no_samples;
            let a_matrix = MatrixBuffer::new_default(&pro_que, k, m, -1.0, LayoutRowMajor);
            a_matrix.buffer().write(&configuration[..]).enq().unwrap();

            let b_matrix = MatrixBuffer::new_default(&pro_que, n, k, -1.0, LayoutRowMajor);
            b_matrix.buffer().write(&samples[..]).enq().unwrap();

            let mut c_matrix = MatrixBuffer::new_default(&pro_que, n, m, -1.0, LayoutRowMajor);

            let task = Gemm::builder()
                .queue(&pro_que.queue())
                .a(&a_matrix)
                .b(&b_matrix)
                .c(&mut c_matrix)
                .build();
            unsafe { task.run().unwrap() }

            read_buffer_to_matrix(c_matrix)
        };

        pretty::assert_eq!(
            format_as_rows(reference_result),
            format_as_rows(gpu_result),
            "Expected <left>, but got <right>"
        );
    }

    #[test]
    fn test_compare_1_3() {
        compare_reference_impl(3, 1, &vec![1.0, 2.0, 3.0], &vec![1.0]);
    }

    #[test]
    fn test_compare_20_10() {
        let no_samples = 20;
        let no_streams = 10;
        compare_reference_impl(
            no_samples,
            no_streams,
            &vec![1.2; no_samples * no_streams],
            &vec![1.0; no_streams * no_streams],
        );
    }

    #[test]
    fn test_compare_23_45_rand() {
        let mut rng = ChaCha20Rng::seed_from_u64(133701010);
        let no_samples = 45;
        let no_streams = 23;

        let samples = (0..no_streams * no_samples)
            .map(|_| rng.gen::<f32>())
            .collect::<Vec<_>>();

        let configuration = (0..no_streams * no_streams)
            .map(|_| rng.gen::<f32>())
            .collect::<Vec<_>>();

        compare_reference_impl(
            no_samples,
            no_streams,
            &vec![1.2; no_samples * no_streams],
            &vec![1.0; no_streams * no_streams],
        );
    }

    #[test]
    fn test_reference_1_3() {
        let res = reference_implementation(3, 1, &vec![1.0, 2.0, 3.0], &vec![1.0]);
        println!("{:?}", res);
        let expected_result = vec![vec![1.0, 2.0, 3.0]];
        pretty::assert_eq!(res, expected_result);
    }

    #[test]
    fn test_reference_3_5() {
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
            vec![ 29.,  34.,  39.],
            vec![ 46.,  53.,  60.],
            vec![ 93.,  108.,  123.],
            vec![0., 0., 0.],
            vec![ 102.,  117.,  132.],
        ];
        pretty::assert_eq!(res, expected_result);
    }
}
