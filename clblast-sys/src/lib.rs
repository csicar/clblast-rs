use libc::c_uint;

mod internal;

use cl_sys::{c_int, c_void, clWaitForEvents};
use internal::*;
use ocl_core::ClNullEventPtr;
use snafu::{ensure, Backtrace, ErrorCompat, ResultExt, Snafu};
use std::ptr;

pub enum MatrixLayout {
    ColMajor,
    RowMajor,
}
impl MatrixLayout {
    fn to_c(&self) -> c_uint {
        match self {
            Self::ColMajor => CLBlastLayout__CLBlastLayoutColMajor,
            Self::RowMajor => CLBlastLayout__CLBlastLayoutRowMajor,
        }
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

/// Computes `C := alpha * A * B + beta * C` on single precision floats
///
/// # Arguments
/// - Matrix A: K⨯M (K Wide, M High)
/// - Matrix B: N⨯K (N Wide, K High)
/// - Matrix C: M⨯N (N Wide, M High)
///
/// For details see: https://cnugteren.github.io/tutorial/pages/page2.html
pub unsafe fn blast_sgemm<En: ClNullEventPtr>(
    layout: MatrixLayout,
    a_transpose: MatrixTranspose,
    b_transpose: MatrixTranspose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a_buffer: &ocl_core::Mem,
    a_offset: usize,
    a_ld: usize,
    b_buffer: &ocl_core::Mem,
    b_offset: usize,
    b_ld: usize,
    beta: f32,
    c_buffer: &ocl_core::Mem,
    c_offset: usize,
    c_ld: usize,
    queue: &ocl_core::CommandQueue,
    event: Option<En>,
) -> Result<(), Error> {
    let mut q = queue.as_ptr();
    let ev: *mut *mut c_void = match event {
        None => &mut ptr::null_mut::<c_void>(),
        Some(mut event) => &mut event.alloc_new().cast::<c_void>(),
    };

    let status_code = CLBlastSgemm(
        layout.to_c(),
        a_transpose.to_c(),
        b_transpose.to_c(),
        m as u64,
        n as u64,
        k as u64,
        alpha,
        a_buffer.as_ptr(),
        a_offset as u64,
        a_ld as u64,
        b_buffer.as_ptr(),
        b_offset as u64,
        b_ld as u64,
        beta,
        c_buffer.as_ptr(),
        c_offset as u64,
        c_ld as u64,
        &mut q,
        ev,
    );

    Error::from_c_either(status_code)
}

pub unsafe fn blast_dgemm<En: ClNullEventPtr>(
    layout: MatrixLayout,
    a_transpose: MatrixTranspose,
    b_transpose: MatrixTranspose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a_buffer: &ocl_core::Mem,
    a_offset: usize,
    a_ld: usize,
    b_buffer: &ocl_core::Mem,
    b_offset: usize,
    b_ld: usize,
    beta: f64,
    c_buffer: &ocl_core::Mem,
    c_offset: usize,
    c_ld: usize,
    queue: &ocl_core::CommandQueue,
    event: Option<En>,
) -> Result<(), Error> {
    let mut q = queue.as_ptr();
    let ev: *mut *mut c_void = match event {
        None => &mut ptr::null_mut::<c_void>(),
        Some(mut event) => &mut event.alloc_new().cast::<c_void>(),
    };

    let status_code = CLBlastDgemm(
        layout.to_c(),
        a_transpose.to_c(),
        b_transpose.to_c(),
        m as u64,
        n as u64,
        k as u64,
        alpha,
        a_buffer.as_ptr(),
        a_offset as u64,
        a_ld as u64,
        b_buffer.as_ptr(),
        b_offset as u64,
        b_ld as u64,
        beta,
        c_buffer.as_ptr(),
        c_offset as u64,
        c_ld as u64,
        &mut q,
        ev,
    );

    Error::from_c_either(status_code)
}

pub enum MultiplicationSide {
    Left,
    Right
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
    Lower
}

impl TriangleLayout{ 
    fn to_c(self: &Self) -> CLBlastLayout {
        match self {
            TriangleLayout::Upper => CLBlastTriangle__CLBlastTriangleUpper,
            TriangleLayout::Lower => CLBlastTriangle__CLBlastTriangleLower,
        }
    }
}

/// Same operation as sGEMM, but `A` is symmetric instead. 
/// - In case of `side == Left`, `A` is a symmetric `m` by `m` matrix and `C = alpha * A * B + beta * C` is performed
/// - In case of `side == kRight`, `A` is a symmtric `n` by `n` matrix and `C = alpha * B * A + beta * C` is performed.
pub unsafe fn blast_ssymm<En: ClNullEventPtr>(
    layout: MatrixLayout,
    side: MultiplicationSide,
    triangle: TriangleLayout,
    m: usize,
    n: usize,
    alpha: f32,
    a_buffer: &ocl_core::Mem,
    a_offset: usize,
    a_ld: usize,
    b_buffer: &ocl_core::Mem,
    b_offset: usize,
    b_ld: usize,
    beta: f32,
    c_buffer: &ocl_core::Mem,
    c_offset: usize,
    c_ld: usize,
    queue: &ocl_core::CommandQueue,
    event: Option<En>,
) -> Result<(), Error> {
    let mut q = queue.as_ptr();
    let ev: *mut *mut c_void = match event {
        None => &mut ptr::null_mut::<c_void>(),
        Some(mut event) => &mut event.alloc_new().cast::<c_void>(),
    };

    let status_code = CLBlastSsymm(
        layout.to_c(),
        side.to_c(),
        triangle.to_c(),
        m as u64,
        n as u64,
        alpha,
        a_buffer.as_ptr(),
        a_offset as u64,
        a_ld as u64,
        b_buffer.as_ptr(),
        b_offset as u64,
        b_ld as u64,
        beta,
        c_buffer.as_ptr(),
        c_offset as u64,
        c_ld as u64,
        &mut q,
        ev,
    );

    Error::from_c_either(status_code)
}


/// Same operation as dGEMM, but `A` is symmetric instead. 
/// - In case of `side == Left`, `A` is a symmetric `m` by `m` matrix and `C = alpha * A * B + beta * C` is performed
/// - In case of `side == kRight`, `A` is a symmtric `n` by `n` matrix and `C = alpha * B * A + beta * C` is performed.
pub unsafe fn blast_dsymm<En: ClNullEventPtr>(
    layout: MatrixLayout,
    side: MultiplicationSide,
    triangle: TriangleLayout,
    m: usize,
    n: usize,
    alpha: f64,
    a_buffer: &ocl_core::Mem,
    a_offset: usize,
    a_ld: usize,
    b_buffer: &ocl_core::Mem,
    b_offset: usize,
    b_ld: usize,
    beta: f64,
    c_buffer: &ocl_core::Mem,
    c_offset: usize,
    c_ld: usize,
    queue: &ocl_core::CommandQueue,
    event: Option<En>,
) -> Result<(), Error> {
    let mut q = queue.as_ptr();
    let ev: *mut *mut c_void = match event {
        None => &mut ptr::null_mut::<c_void>(),
        Some(mut event) => &mut event.alloc_new().cast::<c_void>(),
    };

    let status_code = CLBlastDsymm(
        layout.to_c(),
        side.to_c(),
        triangle.to_c(),
        m as u64,
        n as u64,
        alpha,
        a_buffer.as_ptr(),
        a_offset as u64,
        a_ld as u64,
        b_buffer.as_ptr(),
        b_offset as u64,
        b_ld as u64,
        beta,
        c_buffer.as_ptr(),
        c_offset as u64,
        c_ld as u64,
        &mut q,
        ev,
    );

    Error::from_c_either(status_code)
}

#[derive(Debug, Snafu)]
pub enum OclError {
    OpenCLCompilerNotAvailable,
    TempBufferAllocFailure,
    OpenCLOutOfResources,
    OpenCLOutOfHostMemory,
    OpenCLBuildProgramFailure,
    InvalidValue,
    InvalidCommandQueue,
    InvalidMemObject,
    InvalidBinary,
    InvalidBuildOptions,
    InvalidProgram,
    InvalidProgramExecutable,
    InvalidKernelName,
    InvalidKernelDefinition,
    InvalidKernel,
    InvalidArgIndex,
    InvalidArgValue,
    InvalidArgSize,
    InvalidKernelArgs,
    InvalidLocalNumDimensions,
    InvalidLocalThreadsTotal,
    InvalidLocalThreadsDim,
    InvalidGlobalOffset,
    InvalidEventWaitList,
    InvalidEvent,
    InvalidOperation,
    InvalidBufferSize,
    InvalidGlobalWorkSize,
}

impl OclError {
    fn from_c(status_code: c_int) -> Option<OclError> {
        #![allow(non_upper_case_globals)]

        match status_code {
            CLBlastStatusCode__CLBlastSuccess => None,
            CLBlastStatusCode__CLBlastOpenCLCompilerNotAvailable => {
                Some(OclError::OpenCLCompilerNotAvailable)
            }
            CLBlastStatusCode__CLBlastTempBufferAllocFailure => {
                Some(OclError::TempBufferAllocFailure)
            }
            CLBlastStatusCode__CLBlastOpenCLOutOfResources => Some(OclError::OpenCLOutOfResources),
            CLBlastStatusCode__CLBlastOpenCLOutOfHostMemory => {
                Some(OclError::OpenCLOutOfHostMemory)
            }
            CLBlastStatusCode__CLBlastOpenCLBuildProgramFailure => {
                Some(OclError::OpenCLBuildProgramFailure)
            }
            CLBlastStatusCode__CLBlastInvalidValue => Some(OclError::InvalidValue),
            CLBlastStatusCode__CLBlastInvalidCommandQueue => Some(OclError::InvalidCommandQueue),
            CLBlastStatusCode__CLBlastInvalidMemObject => Some(OclError::InvalidMemObject),
            CLBlastStatusCode__CLBlastInvalidBinary => Some(OclError::InvalidBinary),
            CLBlastStatusCode__CLBlastInvalidBuildOptions => Some(OclError::InvalidBuildOptions),
            CLBlastStatusCode__CLBlastInvalidProgram => Some(OclError::InvalidProgram),
            CLBlastStatusCode__CLBlastInvalidProgramExecutable => {
                Some(OclError::InvalidProgramExecutable)
            }
            CLBlastStatusCode__CLBlastInvalidKernelName => Some(OclError::InvalidKernelName),
            CLBlastStatusCode__CLBlastInvalidKernelDefinition => {
                Some(OclError::InvalidKernelDefinition)
            }
            CLBlastStatusCode__CLBlastInvalidKernel => Some(OclError::InvalidKernel),
            CLBlastStatusCode__CLBlastInvalidArgIndex => Some(OclError::InvalidArgIndex),
            CLBlastStatusCode__CLBlastInvalidArgValue => Some(OclError::InvalidArgValue),
            CLBlastStatusCode__CLBlastInvalidArgSize => Some(OclError::InvalidArgSize),
            CLBlastStatusCode__CLBlastInvalidKernelArgs => Some(OclError::InvalidKernelArgs),
            CLBlastStatusCode__CLBlastInvalidLocalNumDimensions => {
                Some(OclError::InvalidLocalNumDimensions)
            }
            CLBlastStatusCode__CLBlastInvalidLocalThreadsTotal => {
                Some(OclError::InvalidLocalThreadsTotal)
            }
            CLBlastStatusCode__CLBlastInvalidLocalThreadsDim => {
                Some(OclError::InvalidLocalThreadsDim)
            }
            CLBlastStatusCode__CLBlastInvalidGlobalOffset => Some(OclError::InvalidGlobalOffset),
            CLBlastStatusCode__CLBlastInvalidEventWaitList => Some(OclError::InvalidEventWaitList),
            CLBlastStatusCode__CLBlastInvalidEvent => Some(OclError::InvalidEvent),
            CLBlastStatusCode__CLBlastInvalidOperation => Some(OclError::InvalidOperation),
            CLBlastStatusCode__CLBlastInvalidBufferSize => Some(OclError::InvalidBufferSize),
            CLBlastStatusCode__CLBlastInvalidGlobalWorkSize => {
                Some(OclError::InvalidGlobalWorkSize)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Snafu)]
pub enum BlasError {
    NotImplemented,
    InvalidMatrixA,
    InvalidMatrixB,
    InvalidMatrixC,
    InvalidVectorX,
    InvalidVectorY,
    InvalidDimension,
    InvalidLeadDimA,
    InvalidLeadDimB,
    InvalidLeadDimC,
    InvalidIncrementX,
    InvalidIncrementY,
    InsufficientMemoryA,
    InsufficientMemoryB,
    InsufficientMemoryC,
    InsufficientMemoryX,
    InsufficientMemoryY,
}

impl BlasError {
    fn from_c(status_code: c_int) -> Option<BlasError> {
        #![allow(non_upper_case_globals)]

        match status_code {
            CLBlastStatusCode__CLBlastNotImplemented => Some(BlasError::NotImplemented),
            CLBlastStatusCode__CLBlastInvalidMatrixA => Some(BlasError::InvalidMatrixA),
            CLBlastStatusCode__CLBlastInvalidMatrixB => Some(BlasError::InvalidMatrixB),
            CLBlastStatusCode__CLBlastInvalidMatrixC => Some(BlasError::InvalidMatrixC),
            CLBlastStatusCode__CLBlastInvalidVectorX => Some(BlasError::InvalidVectorX),
            CLBlastStatusCode__CLBlastInvalidVectorY => Some(BlasError::InvalidVectorY),
            CLBlastStatusCode__CLBlastInvalidDimension => Some(BlasError::InvalidDimension),
            CLBlastStatusCode__CLBlastInvalidLeadDimA => Some(BlasError::InvalidLeadDimA),
            CLBlastStatusCode__CLBlastInvalidLeadDimB => Some(BlasError::InvalidLeadDimB),
            CLBlastStatusCode__CLBlastInvalidLeadDimC => Some(BlasError::InvalidLeadDimC),
            CLBlastStatusCode__CLBlastInvalidIncrementX => Some(BlasError::InvalidIncrementX),
            CLBlastStatusCode__CLBlastInvalidIncrementY => Some(BlasError::InvalidIncrementY),
            CLBlastStatusCode__CLBlastInsufficientMemoryA => Some(BlasError::InsufficientMemoryA),
            CLBlastStatusCode__CLBlastInsufficientMemoryB => Some(BlasError::InsufficientMemoryB),
            CLBlastStatusCode__CLBlastInsufficientMemoryC => Some(BlasError::InsufficientMemoryC),
            CLBlastStatusCode__CLBlastInsufficientMemoryX => Some(BlasError::InsufficientMemoryX),
            CLBlastStatusCode__CLBlastInsufficientMemoryY => Some(BlasError::InsufficientMemoryY),
            _ => None,
        }
    }
}

#[derive(Debug, Snafu)]
pub enum BlastError {
    InsufficientMemoryTemp,
    InvalidBatchCount,
    InvalidOverrideKernel,
    MissingOverrideParameter,
    InvalidLocalMemUsage,
    NoHalfPrecision,
    NoDoublePrecision,
    InvalidVectorScalar,
    InsufficientMemoryScalar,
    DatabaseError,
    UnknownError,
    UnexpectedError,
}

impl BlastError {
    fn from_c(status_code: c_int) -> Option<BlastError> {
        #![allow(non_upper_case_globals)]

        match status_code {
            CLBlastStatusCode__CLBlastInsufficientMemoryTemp => {
                Some(BlastError::InsufficientMemoryTemp)
            }
            CLBlastStatusCode__CLBlastInvalidBatchCount => Some(BlastError::InvalidBatchCount),
            CLBlastStatusCode__CLBlastInvalidOverrideKernel => {
                Some(BlastError::InvalidOverrideKernel)
            }
            CLBlastStatusCode__CLBlastMissingOverrideParameter => {
                Some(BlastError::MissingOverrideParameter)
            }
            CLBlastStatusCode__CLBlastInvalidLocalMemUsage => {
                Some(BlastError::InvalidLocalMemUsage)
            }
            CLBlastStatusCode__CLBlastNoHalfPrecision => Some(BlastError::NoHalfPrecision),
            CLBlastStatusCode__CLBlastNoDoublePrecision => Some(BlastError::NoDoublePrecision),
            CLBlastStatusCode__CLBlastInvalidVectorScalar => Some(BlastError::InvalidVectorScalar),
            CLBlastStatusCode__CLBlastInsufficientMemoryScalar => {
                Some(BlastError::InsufficientMemoryScalar)
            }
            CLBlastStatusCode__CLBlastDatabaseError => Some(BlastError::DatabaseError),
            CLBlastStatusCode__CLBlastUnknownError => Some(BlastError::UnknownError),
            CLBlastStatusCode__CLBlastUnexpectedError => Some(BlastError::UnexpectedError),
            _ => None,
        }
    }
}

#[derive(Debug, Snafu)]
pub enum Error {
    Ocl { source: OclError },
    Blas { source: BlasError },
    Blast { source: BlastError },
    Unknown { status_code: i32 },
}

impl Error {
    fn from_c_either(status_code: c_int) -> Result<(), Error> {
        match Error::from_c(status_code) {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
    fn from_c(status_code: c_int) -> Option<Error> {
        if status_code == CLBlastStatusCode__CLBlastSuccess {
            None
        } else {
            OclError::from_c(status_code)
                .map(|s| Error::Ocl {source: s})
                .or_else(|| BlasError::from_c(status_code).map(|s| Error::Blas {source: s}))
                .or_else(|| BlastError::from_c(status_code).map(|s| Error::Blast {source : s}))
                .or_else(|| Some(Error::Unknown { status_code }))
        }
    }
}

pub unsafe fn clear_cache() -> CLBlastStatusCode {
    CLBlastClearCache()
}

#[cfg(test)]
mod test {
    use ocl::{flags, ProQue};
    use std::error::Error;
    use std::time::Instant;

    use super::*;

    #[test]
    fn test_gemm() -> Result<(), Box<dyn Error>> {
        let src = r#"
            __kernel void add(__global float* buffer, float scalar) {
                buffer[get_global_id(0)] += scalar;
            }
        "#;
        let width = 32;
        let height = 32;

        let pro_que = ProQue::builder()
            .src(src)
            .dims(width * height)
            .build()
            .unwrap();

        let a_buffer = pro_que
            .buffer_builder()
            .flags(flags::MEM_READ_WRITE)
            .len(width * width)
            .fill_val(0.2f32)
            .build()
            .unwrap();
        let b_buffer = pro_que
            .buffer_builder()
            .flags(flags::MEM_READ_WRITE)
            .len(width * height)
            .fill_val(4f32)
            .build()
            .unwrap();
        let c_buffer = pro_que
            .buffer_builder()
            .flags(flags::MEM_READ_WRITE)
            .len(width * height)
            .fill_val(-1f32)
            .build()
            .unwrap();
        let before = Instant::now();
        unsafe {
            blast_sgemm(
                MatrixLayout::ColMajor,
                MatrixTranspose::No,
                MatrixTranspose::No,
                32,
                32,
                32,
                1.0,
                a_buffer.as_core(),
                0,
                32,
                b_buffer.as_core(),
                0,
                32,
                0.0,
                c_buffer.as_core(),
                0,
                32,
                pro_que.as_core(),
                None::<()>,
            )
        }?;

        let mut c_dat = vec![0.0; width * height];
        c_buffer.read(&mut c_dat[..]).enq().unwrap();

        println!("{:?} {:?}", &c_dat[0..10], before.elapsed());

        Ok(())
    }
}
