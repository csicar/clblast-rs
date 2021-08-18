#![allow(non_snake_case)]
use clblast_sys::*;
use ocl::ffi::c_int;
use snafu::{Snafu};


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
          _ if { status_code == CLBlastStatusCode__CLBlastInsufficientMemoryTemp } => {
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
  pub fn from_c_either(status_code: c_int) -> Result<(), Error> {
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