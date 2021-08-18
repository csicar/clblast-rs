use std::marker::PhantomData;
use std::ptr;

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
use ocl::ffi::c_uint;
use ocl::Buffer;
use ocl::OclPrm;
use ocl::OclScl;
use ocl::Queue;
use ocl_core::wait_for_events;
use ocl_core::Event;
use ocl_core::OclNum;
use typed_builder::TypedBuilder;
mod result;

use result::Error;

pub trait MatrixLayout {
    fn to_c() -> c_uint;
}

pub struct LayoutColMajor;
impl MatrixLayout for LayoutColMajor {
    fn to_c() -> c_uint {
        CLBlastLayout__CLBlastLayoutColMajor
    }
}
pub struct LayoutRowMajor;
impl MatrixLayout for LayoutRowMajor {
    fn to_c() -> c_uint {
        CLBlastLayout__CLBlastLayoutRowMajor
    }
}

pub enum MatrixTranspose {
    Yes,
    No,
    Conjugate,
}

impl MatrixTranspose {
    fn to_c(&self) -> c_uint {
        match self {
            Self::Yes => CLBlastTranspose__CLBlastTransposeYes,
            Self::No => CLBlastTranspose__CLBlastTransposeNo,
            Self::Conjugate => CLBlastTranspose__CLBlastTransposeConjugate,
        }
    }
}

pub enum MultiplicationSide {
    Left,
    Right,
}
impl MultiplicationSide {
    fn to_c(self: &Self) -> CLBlastSide {
        match self {
            MultiplicationSide::Left => CLBlastSide__CLBlastSideLeft,
            MultiplicationSide::Right => CLBlastSide__CLBlastSideRight,
        }
    }
}

pub enum TriangleLayout {
    Upper,
    Lower,
}

impl TriangleLayout {
    fn to_c(self: &Self) -> CLBlastLayout {
        match self {
            TriangleLayout::Upper => CLBlastTriangle__CLBlastTriangleUpper,
            TriangleLayout::Lower => CLBlastTriangle__CLBlastTriangleLower,
        }
    }
}

pub struct MatrixBuffer<T: OclPrm, L: MatrixLayout> {
    rows: usize,
    columns: usize,
    offset: usize,
    pub buffer: Buffer<T>,
    layout: PhantomData<L>,
}

impl<T, L> MatrixBuffer<T, L>
where
    T: OclPrm,
    L: MatrixLayout,
{
    pub fn new(columns: usize, rows: usize, buffer: Buffer<T>, layout: L) -> Self {
        assert!(rows * columns <= buffer.len());
        MatrixBuffer {
            rows,
            columns,
            offset: 0,
            buffer,
            layout: PhantomData::<L>,
        }
    }
}

pub trait NeutralAdd {
    const zero: Self;
}

impl NeutralAdd for f32 {
    const zero: f32 = 0.0;
}

pub trait NeutralMul {
    const one: Self;
}

impl NeutralMul for f32 {
    const one: f32 = 1.0;
}

#[derive(TypedBuilder)]
pub struct MatrixMultiplication<'a, T, L>
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
    #[builder(default=NeutralMul::one)]
    alpha: T,
    #[builder(default=NeutralAdd::zero)]
    beta: T,

    // transpose
    #[builder(default=MatrixTranspose::No)]
    transpose_a: MatrixTranspose,
    #[builder(default=MatrixTranspose::No)]
    transpose_b: MatrixTranspose,
}

impl<'a, L> MatrixMultiplication<'a, f32, L>
where
    L: MatrixLayout,
{
    unsafe fn run(self) -> Result<(), Error> {
        assert_eq!(self.a.columns, self.b.rows, "a.columns /= b.rows (k)");
        let k = self.a.columns;

        assert_eq!(self.b.columns, self.c.columns, "b.columns /= c.columns (n)");
        let n = self.b.columns;

        assert_eq!(self.c.rows, self.a.rows, "c.columns /= a.rows (m)");
        let m = self.c.columns;

        let res = CLBlastSgemm(
            <L as MatrixLayout>::to_c(),
            MatrixTranspose::No.to_c(),
            MatrixTranspose::No.to_c(),
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

pub trait MultiplicationExecutor<'a, T, L: 'a>
where
    T: OclPrm + NeutralMul + NeutralAdd,
    L: MatrixLayout,
{
    // /// Computes `C := alpha * A * B + beta * C` on single precision floats
    // ///
    // /// # Arguments
    // /// - Matrix A: K⨯M (K Wide, M High)
    // /// - Matrix B: N⨯K (N Wide, K High)
    // /// - Matrix C: M⨯N (N Wide, M High)
    // ///
    // /// For details see: https://cnugteren.github.io/tutorial/pages/page2.html
    // unsafe fn multiply(
    //     self: &'a Self,
    //     task: MatrixMultiplication<'a, T, L>
    // ) -> Result<(), Error>;

    fn gemm(
        self: &'a Self,
    ) -> MatrixMultiplicationBuilder<'a, ((&'a Queue,), (), (), (), (), (), (), ()), T, L>;
}

impl<'a, T, L: 'a> MultiplicationExecutor<'a, T, L> for Queue
where
    T: OclPrm + NeutralMul + NeutralAdd,
    L: MatrixLayout,
{
    fn gemm(
        self: &'a Self,
    ) -> MatrixMultiplicationBuilder<'a, ((&'a Queue,), (), (), (), (), (), (), ()), T, L> {
        MatrixMultiplication::<'a, T, L>::builder().queue(&self)
    }
}

#[cfg(test)]
mod test {
    use ocl::{flags, ProQue};
    use rand::prelude::*;
    use rand_chacha::ChaCha20Rng;
    use std::thread::sleep;
    use std::time::{Duration, Instant};
    use std::{error::Error, ptr};

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
        unsafe { pro_que.queue().gemm().a(&a).b(&b).c(&mut c).build().run()? };

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
