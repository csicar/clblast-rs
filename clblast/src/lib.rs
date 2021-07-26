use clblast_sys::Error;
use clblast_sys::blast_sgemm;
use clblast_sys::clear_cache;
use clblast_sys::MatrixLayout;
use clblast_sys::MatrixTranspose;
use ocl::Buffer;
use ocl::OclPrm;
use ocl::Queue;
use ocl_core::wait_for_events;
use ocl_core::Event;
mod builder;

pub struct RowMatrix<T: OclPrm> {
    rows: usize,
    columns: usize,
    offset: usize,
    pub buffer: Buffer<T>,
}

impl<T> RowMatrix<T>
where
    T: OclPrm,
{
    pub fn new(columns: usize, rows: usize, buffer: Buffer<T>) -> Self {
        assert!(rows * columns <= buffer.len());
        RowMatrix {
            rows,
            columns,
            offset: 0,
            buffer,
        }
    }
}

trait MultiplicationExecutor<T>
where
    T: OclPrm,
{
    /// Computes `C := alpha * A * B + beta * C` on single precision floats
    ///
    /// # Arguments
    /// - Matrix A: K⨯M (K Wide, M High)
    /// - Matrix B: N⨯K (N Wide, K High)
    /// - Matrix C: M⨯N (N Wide, M High)
    ///
    /// For details see: https://cnugteren.github.io/tutorial/pages/page2.html
    unsafe fn multiply(
        self: &Self,
        a: &RowMatrix<T>,
        b: &RowMatrix<T>,
        c: &mut RowMatrix<T>,
        alpha: T,
        beta: T,
    ) -> Result<(), Error>;
}

impl MultiplicationExecutor<f32> for Queue {
    unsafe fn multiply(
        self: &Queue,
        a: &RowMatrix<f32>,
        b: &RowMatrix<f32>,
        c: &mut RowMatrix<f32>,
        alpha: f32,
        beta: f32,
    ) -> Result<(), Error> {
        assert_eq!(a.columns, b.rows, "a.columns /= b.rows (k)");
        let k = a.columns;

        assert_eq!(b.columns, c.columns, "b.columns /= c.columns (n)");
        let n = b.columns;

        assert_eq!(c.rows, a.rows, "c.columns /= a.rows (m)");
        let m = c.columns;

        let res = blast_sgemm(
            MatrixLayout::RowMajor,
            MatrixTranspose::No,
            MatrixTranspose::No,
            m,
            n,
            k,
            alpha,
            &a.buffer,
            a.offset,
            k,
            &b.buffer,
            b.offset,
            n,
            beta,
            &c.buffer,
            c.offset,
            n,
            self.as_core(),
            None::<()>,
        );

        res
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
        let no_streams = 10240;
        let no_samples = 800;
        let mut rng = ChaCha20Rng::seed_from_u64(133701010);

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

        let a = RowMatrix::new(no_streams, no_streams, a_buffer);

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

        let b = RowMatrix::new(no_samples, no_streams, b_buffer);

        let c_buffer = pro_que
            .buffer_builder()
            .flags(flags::MEM_READ_WRITE)
            .len(no_streams * no_samples)
            .fill_val(0f32)
            .build()
            .unwrap();
        let mut c = RowMatrix::new(no_samples, no_streams, c_buffer);

        let before = Instant::now();
        println!("run..");
        unsafe {
            pro_que.queue().multiply(&a, &b, &mut c, 1.0, 0.0)?;
        }

        let mut c_dat = vec![0.0; no_streams * no_samples];
        c.buffer.read(&mut c_dat[..]).enq().unwrap();

        println!("{:?} {:?}", &c_dat[0..10], before.elapsed());
        let reference_result = reference_implementation(no_samples, no_streams, &b_val, &a_val);
        assert_eq!(c_dat.len(), reference_result.len());
        c_dat
            .iter()
            .zip(reference_result)
            .for_each(|(&res, ref_res)| {
                assert!((res - ref_res) < 0.01, "{} ~= {}", res, ref_res);
            });
        Ok(())
    }

    #[test]
    fn test_mem_leak() -> Result<(), Box<dyn Error>> {
        for _ in 0..40 {
            println!("run..");
            test_gemm()?;
            println!("waiting..");
            sleep(Duration::from_secs(1));
            println!("waiting done");
        }
        Ok(())
    }
}
