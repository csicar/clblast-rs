use std::marker::PhantomData;
use std::ptr;

use clblast_sys::cl_double2;
use clblast_sys::cl_float2;
use clblast_sys::CLBlastDgemm;
use clblast_sys::CLBlastHgemm;
use clblast_sys::CLBlastLayout;
use clblast_sys::CLBlastLayout__CLBlastLayoutColMajor;
use clblast_sys::CLBlastLayout__CLBlastLayoutRowMajor;
use clblast_sys::CLBlastSgemm;
use clblast_sys::CLBlastSide;
use clblast_sys::CLBlastSide__CLBlastSideLeft;
use clblast_sys::CLBlastSide__CLBlastSideRight;
use clblast_sys::CLBlastTranspose;
use clblast_sys::CLBlastTranspose__CLBlastTransposeConjugate;
use clblast_sys::CLBlastTranspose__CLBlastTransposeNo;
use clblast_sys::CLBlastTranspose__CLBlastTransposeYes;
use clblast_sys::CLBlastTriangle__CLBlastTriangleLower;
use clblast_sys::CLBlastTriangle__CLBlastTriangleUpper;
use num_complex::Complex32;
use num_complex::Complex64;
use ocl::Buffer;
use ocl::OclPrm;
use ocl::Queue;
use typed_builder::TypedBuilder;
pub mod gemm;
mod result;
mod swap;
mod scal;
mod copy;
mod axpy;
mod dot;
mod dotc;
mod nrm2;
mod asum;
mod sum;
mod amax;
mod amin;
mod max;
mod min;


pub use result::Error;

pub trait ReprSys {
    type Representation;
    fn to_c(&self) -> Self::Representation;
}

impl ReprSys for Complex32 {
    type Representation = cl_float2;

    fn to_c(&self) -> cl_float2 {
        cl_float2 {
            s: [self.re, self.im],
        }
    }
}

impl ReprSys for Complex64 {
    type Representation = cl_double2;

    fn to_c(&self) -> cl_double2 {
        cl_double2 {
            s: [self.re, self.im],
        }
    }
}

pub trait MatrixLayout: ReprSys<Representation = CLBlastLayout> {}

pub struct LayoutColMajor;
impl ReprSys for LayoutColMajor {
    type Representation = CLBlastLayout;

    fn to_c(&self) -> CLBlastLayout {
        CLBlastLayout__CLBlastLayoutColMajor
    }
}
impl MatrixLayout for LayoutColMajor {}
pub struct LayoutRowMajor;
impl ReprSys for LayoutRowMajor {
    type Representation = CLBlastLayout;

    fn to_c(&self) -> CLBlastLayout {
        CLBlastLayout__CLBlastLayoutRowMajor
    }
}
impl MatrixLayout for LayoutRowMajor {}

pub enum MatrixTranspose {
    Yes,
    No,
    Conjugate,
}

impl ReprSys for MatrixTranspose {
    type Representation = CLBlastTranspose;
    fn to_c(&self) -> CLBlastTranspose {
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
impl ReprSys for MultiplicationSide {
    type Representation = CLBlastSide;

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

impl ReprSys for TriangleLayout {
    type Representation = CLBlastLayout;

    fn to_c(self: &Self) -> CLBlastLayout {
        match self {
            TriangleLayout::Upper => CLBlastTriangle__CLBlastTriangleUpper,
            TriangleLayout::Lower => CLBlastTriangle__CLBlastTriangleLower,
        }
    }
}

#[derive(TypedBuilder)]
pub struct MatrixBuffer<T: OclPrm, L: MatrixLayout> {
    buffer: Buffer<T>,

    rows: usize,
    columns: usize,

    #[builder(default = 0)]
    offset: usize,
    layout: L,
}

impl<T : OclPrm, L : MatrixLayout> MatrixBuffer<T, L> {
    pub fn size(&self) -> usize {
        self.rows * self.columns
    }
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
            layout,
        }
    }

    pub fn new_default(
        pro_que: &ocl::ProQue,
        columns: usize,
        rows: usize,
        fill_val: T,
        layout: L,
    ) -> Self {
        let buffer = pro_que
            .buffer_builder()
            .len(columns * rows)
            .fill_val(fill_val)
            .build()
            .unwrap();
        Self::new(columns, rows, buffer, layout)
    }

    pub fn buffer(&self) -> &Buffer<T> {
        &self.buffer
    }
}

#[derive(TypedBuilder)]
pub struct VectorBuffer<T: OclPrm> {
    buffer: Buffer<T>,

    #[builder(default = 0)]
    offset: usize,
}

pub trait NeutralAdd {
    const ZERO: Self;
}

impl NeutralAdd for f32 {
    const ZERO: f32 = 0.0;
}

impl NeutralAdd for f64 {
    const ZERO: f64 = 0.0;
}

impl NeutralAdd for Complex32 {
    const ZERO: Complex32 = Complex32 { re: 0.0, im: 0.0 };
}

impl NeutralAdd for Complex64 {
    const ZERO: Complex64 = Complex64 { re: 0.0, im: 0.0 };
}

pub trait NeutralMul {
    const ONE: Self;
}

impl NeutralMul for f32 {
    const ONE: f32 = 1.0;
}

impl NeutralMul for f64 {
    const ONE: f64 = 1.0;
}

impl NeutralMul for Complex32 {
    const ONE: Complex32 = Complex32 { re: 1.0, im: 0.0 };
}

impl NeutralMul for Complex64 {
    const ONE: Complex64 = Complex64 { re: 1.0, im: 0.0 };
}
