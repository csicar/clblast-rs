use std::time::Instant;

use clblast::sgemm;
use clblast::RowMatrix;
use criterion::BenchmarkId;
use ocl::flags;
use ocl::MemFlags;
use ocl::ProQue;
use rand::prelude::*;
use rand_chacha::ChaCha20Rng;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_sgemm(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(133701010);
    let src = r#"
                    __kernel void add(__global float* buffer, float scalar) {
                            buffer[get_global_id(0)] += scalar;
                        }
                    "#;

    let pro_que = ProQue::builder().src(src).dims(10 * 10).build().unwrap();
    println!("start!!!!\n\n\n\n");
    let mut group = c.benchmark_group("sgemm");

    for no_streams in [5_000, 10_000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(no_streams),
            no_streams,
            |bencher, &size| {
                println!("Number of Streams: {}", size);
                let no_streams = size;
                let no_samples = 800;

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
                let a = RowMatrix::new(no_streams, no_streams, a_buffer);

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
                let b = RowMatrix::new(no_samples, no_streams, b_buffer);

                let c_buffer = pro_que
                    .buffer_builder()
                    .flags(flags::MEM_READ_WRITE)
                    .len(no_streams * no_samples)
                    .fill_val(-1f32)
                    .build()
                    .unwrap();
                let mut c = RowMatrix::new(no_samples, no_streams, c_buffer);
                
                bencher.iter(|| {
                    c.buffer
                        .write(
                            &(0..no_streams * no_samples)
                                .map(|_| rng.gen::<f32>())
                                .collect::<Vec<_>>(),
                        )
                        .enq()
                        .unwrap();

                    let before = Instant::now();
                    let err_code = unsafe { sgemm(&a, &b, &mut c, 1.0, 0.0, &pro_que.queue()) };
                    println!("err code: {}", err_code);

                    let mut c_dat = vec![0.0; no_streams * no_samples];
                    c.buffer.read(&mut c_dat[..]).enq().unwrap();

                    println!("{:?} {:?}", &c_dat[..10], before.elapsed());
                });
            },
        );
    }
    group.finish();
}

pub fn criterion_benchmark(c: &mut Criterion) {
    bench_sgemm(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
