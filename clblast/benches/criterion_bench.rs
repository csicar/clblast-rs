use std::fmt::Display;
use std::time::Instant;

use clblast::LayoutRowMajor;
use clblast::MatrixBuffer;
use clblast::gemm::Gemm;
use clblast::gemm::RunGemm;
use criterion::BenchmarkId;
use ocl::flags;
use ocl::MemFlags;
use ocl::ProQue;
use rand::prelude::*;
use rand_chacha::ChaCha20Rng;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

struct Parameters {
    no_streams: usize,
    no_samples: usize,
}

impl Display for Parameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Streams: {}, Samples: {}",
            self.no_streams, self.no_samples
        )
    }
}

fn bench_sgemm(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(133701010);
    let src = "";
    let pro_que = ProQue::builder().src(src).dims(10 * 10).build().unwrap();
    println!("start!!!!\n\n\n\n");
    let mut group = c.benchmark_group("gemm");

    for &no_streams in [5_000, 10_000].iter() {
        for &no_samples in [160, 320, 640].iter() {
            group.bench_with_input(
                BenchmarkId::from_parameter(Parameters {
                    no_streams,
                    no_samples,
                }),
                &(no_streams, no_samples),
                |bencher, &(no_streams, no_samples)| {
                    println!("Number of Streams: {}", no_streams);
                    let a_buffer = pro_que
                        .buffer_builder()
                        .flags(flags::MEM_READ_WRITE)
                        .len(no_streams * no_streams)
                        .fill_val(0.2f32)
                        .build()
                        .unwrap();
                    a_buffer
                        .write(
                            &(0..no_streams * no_streams)
                                .map(|_| rng.gen::<f32>())
                                .collect::<Vec<_>>(),
                        )
                        .enq()
                        .unwrap();
                    let a = MatrixBuffer::new(no_streams, no_streams, a_buffer, LayoutRowMajor);

                    let b_buffer = pro_que
                        .buffer_builder()
                        .flags(flags::MEM_READ_WRITE)
                        .len(no_streams * no_samples)
                        .fill_val(4f32)
                        .build()
                        .unwrap();
                    b_buffer
                        .write(
                            &(0..no_streams * no_samples)
                                .map(|_| rng.gen::<f32>())
                                .collect::<Vec<_>>(),
                        )
                        .enq()
                        .unwrap();
                    let b = MatrixBuffer::new(no_samples, no_streams, b_buffer, LayoutRowMajor);

                    let c_buffer = pro_que
                        .buffer_builder()
                        .flags(flags::MEM_READ_WRITE)
                        .len(no_streams * no_samples)
                        .fill_val(-1f32)
                        .build()
                        .unwrap();
                    let mut c = MatrixBuffer::new(no_samples, no_streams, c_buffer, LayoutRowMajor);

                    bencher.iter(|| {
                        let before_write = Instant::now();
                        a.buffer()
                            .write(
                                &(0..no_streams * no_samples)
                                    .map(|_| rng.gen::<f32>())
                                    .collect::<Vec<_>>(),
                            )
                            .enq()
                            .unwrap();
                        println!("write time: {:?}", before_write.elapsed());
                        let before = Instant::now();
                        unsafe { Gemm::builder().queue(&pro_que.queue()).a(&a).b(&b).c(&mut c).build(); };

                        let mut c_dat = vec![0.0; no_streams * no_samples];
                        c.buffer().read(&mut c_dat[..]).enq().unwrap();

                        println!("{:?} {:?}", &c_dat[..10], before.elapsed());
                    });
                },
            );
        }
    }
    group.finish();
}

pub fn criterion_benchmark(c: &mut Criterion) {
    bench_sgemm(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
