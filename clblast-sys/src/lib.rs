#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(warnings, unused)]
use libc::*;
// Reuse cl_mem ... definitions from cl_sys
use cl_sys::*;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
