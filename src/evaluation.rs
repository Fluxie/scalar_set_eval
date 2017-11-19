extern crate ro_scalar_set;
extern crate std;
extern crate rayon;
extern crate rand;
extern crate memmap;

#[cfg(feature="gpu")]
extern crate ocl;

#[cfg(feature="gpu")]
use self::ocl::ProQue;
#[cfg(feature="gpu")]
use self::ocl::Buffer;
#[cfg(feature="gpu")]
use self::ocl::MemFlags;

use std::slice;

use memmap::{Mmap, Protection};
use self::rayon::prelude::*;
use rand::distributions::{Range};

use enumerations::*;
use traits::*;
use utility;

/// Evaluates integer sets.
pub fn evaluate<'a, T>(
    file: &String,
    values_in_set: i32,
    min_value: i32,
    max_value: i32,
    eval_engine: &EvaluationEngine,
) -> ( u32, std::time::Duration )
where
    T: FromI32 + std::clone::Clone + std::marker::Send + std::marker::Sync + ro_scalar_set::Value + WithGpu,
{
    // Construct test vector.
    let between = Range::new( min_value, max_value );
    let test_set = utility::generate_values( values_in_set, &between );

    // Open file for reading.
    let file = std::fs::File::open( file ).expect( "Failed to open the file." );
    let file = Mmap::open( &file, Protection::Read ).expect( "Failed to map the file" );
    {
        let integer_count = file.len() / 4;
        let buffer: *const T = file.ptr() as *const T;
        let buffer = as_slice( buffer, integer_count );
        {
            // Divide the buffer into sets.
            let sets = attach_buffer( &buffer );

            // Run tests for each set.
            let ( match_counter, duration ) = match *eval_engine
            {
                EvaluationEngine::Cpu => sets.evaluate_with_cpu( &ro_scalar_set::RoScalarSet::new( &test_set ) ),
                EvaluationEngine::Gpu => sets.evaluate_sets_gpu( &test_set ),
            };
            return ( match_counter, duration );
        }
    }
}

/// Declares a set that can be evaluated.
pub struct SetsForEvaluation<'a,T,>
where
    T: 'a + FromI32 + std::clone::Clone + std::marker::Send + std::marker::Sync + ro_scalar_set::Value + WithGpu
{
    #[cfg(feature="gpu")]
    raw_data: &'a[T],
    sets: Vec<ro_scalar_set::RoScalarSet<'a,T>>,
}

impl<'a,T> SetsForEvaluation<'a,T>
where
    T: FromI32 + std::clone::Clone + std::marker::Send + std::marker::Sync + ro_scalar_set::Value + WithGpu,
{
    /// Initializes new set evaluator from a collection of sets.
    #[cfg(feature="gpu")]
    pub fn new(
        raw_data: &'a[T],
        sets: Vec<ro_scalar_set::RoScalarSet<'a,T>>,
    ) -> SetsForEvaluation<'a,T>
    {
        return SetsForEvaluation { raw_data: raw_data, sets: sets };
    }

    /// Initializes new set evaluator from a collection of sets.
    #[cfg(not(feature="gpu"))]
    pub fn new(
        _raw_data: &'a[T],
        sets: Vec<ro_scalar_set::RoScalarSet<'a,T>>,
    ) -> SetsForEvaluation<'a,T>
    {
        return SetsForEvaluation { sets: sets };
    }

    /// Evaluates the sets with CPU.
    pub fn evaluate_with_cpu(
        &self,
        test_set: &ro_scalar_set::RoScalarSet<T>,
    ) -> ( u32, std::time::Duration )
    {
        // Evaluate the sets in parallel.
        let start = std::time::Instant::now();
        let match_counter = self.sets.par_iter()
                .map( |s| evaluate_set_cpu( test_set, &s ) )
                .sum();
        let stop = std::time::Instant::now();
        let duration = stop.duration_since( start );
        return ( match_counter, duration );
    }


    /// GPU evaluation enabled?
    #[cfg(not(feature="gpu"))]
    pub fn evaluate_sets_gpu(
        &self,
        _test_set: &[T],
    ) -> ( u32, std::time::Duration )
    {
        panic!("GPU evaluation support not enabled.");
    }

    /// Evaluates the sets with GPU.
    #[cfg(feature="gpu")]
    pub fn evaluate_sets_gpu(
        &self,
        test_set: &[T],
    ) -> ( u32, std::time::Duration )
    {
        // Delegate to appropriate implementation depending on the data type.
        let start = std::time::Instant::now();
        let match_counter = WithGpu::evaluate_with_gpu( self.raw_data, &self.sets, test_set );
        let stop = std::time::Instant::now();
        let duration = stop.duration_since( start );
        return ( match_counter, duration );
    }
}

/// Trait for evaluating values with GPU.
#[cfg(feature="gpu")]
 pub trait WithGpu
 where
    Self: traits::FromI32 + std::clone::Clone + std::marker::Send + std::marker::Sync + ro_scalar_set::Value
{
    /// Evaluates the given data set with GPU.
    fn evaluate_with_gpu(
        raw_data: &[Self],
        sets: &Vec<ro_scalar_set::RoScalarSet<Self>>,
        test_set: &[Self],
    ) -> u32;
}

/// GPU evaluation support for integers.
#[cfg(feature="gpu")]
impl WithGpu for i32
{
        /// Evaluates the given data set with GPU.
    fn evaluate_with_gpu(
        _raw_data: &[i32],
        _sets: &Vec<ro_scalar_set::RoScalarSet<i32>>,
        _test_set: &[i32],
    ) -> u32
    {
        panic!("Not implemented");
    }
}

/// GPU evaluation support for floats.
#[cfg(feature="gpu")]
impl WithGpu for f32
{
        /// Evaluates the given data set with GPU.
    fn evaluate_with_gpu(
        raw_data: &[f32],
        sets: &Vec<ro_scalar_set::RoScalarSet<f32>>,
        test_set: &[f32],
    ) -> u32
    {
        let src = r#"
                __kernel void search(
                    __global float* buffer,
                    __global int* begin_indexes,
                    __global int* end_indexes,
                    __global float* test_set,
                    __private int const test_set_size
                )
                {
                    /* Determine the range of values we need to scan. */
                    int iBegin = begin_indexes[get_global_id(0)];
                    int iEnd = end_indexes[get_global_id(0)];
                    int iMatches = 0;
                    for( int i = iBegin; i < iEnd; ++i )
                    {
                        for( int t = 0; t < test_set_size; ++t )
                        {
                            float f = fabs( buffer[ i ] - test_set[ t ] );
                            if( f < 0.1 )
                            {
                                return;
                            }
                        }
                    }

                }
            "#;

        // Prepare environment.
        let pro_que = ProQue::builder()
            .src( src )
            .dims( sets.len() )
            .build().unwrap();

        // Load raw data.
        let raw_data_length = raw_data.len();
        let raw_data = Buffer::builder()
                .queue( pro_que.queue().clone() )
                .flags( MemFlags::new().read_only().copy_host_ptr() )
                .dims( raw_data_length )
                .host_data( &raw_data )
                .build().unwrap();

        // Calculate indexes of scalar sets in the raw buffer.
        // These indexes will we be transmitted to the GPU.
        let mut begin_indexes: Vec<i32> = Vec::new();
        let mut end_indexes: Vec<i32> = Vec::new();
        begin_indexes.reserve( sets.len() );
        end_indexes.reserve( sets.len() );
        let mut set_start = 0;
        for s in sets
        {
            let buckets = s.bucket_count();
            let size = s.size();
            let total_size = 1 + buckets + 1 + size;

            // Calculate the indexes.
            let begin_index = set_start + 1 + buckets as i32 + 1;
            let end_index = set_start + total_size as i32;
            begin_indexes.push( begin_index );
            end_indexes.push( end_index );
            set_start = end_index;
        }

        // Load the indexes to GPU.
        let begin_indexes = Buffer::builder()
                .queue( pro_que.queue().clone() )
                .flags( MemFlags::new().read_only().copy_host_ptr() )
                .dims( begin_indexes.len() )
                .host_data( &begin_indexes )
                .build().unwrap();
        let end_indexes = Buffer::builder()
                .queue( pro_que.queue().clone() )
                .flags( MemFlags::new().read_only().copy_host_ptr() )
                .dims( end_indexes.len() )
                .host_data( &end_indexes )
                .build().unwrap();

        // Load test set.
        let test_set = Buffer::builder()
                .queue( pro_que.queue().clone() )
                .flags( MemFlags::new().read_only().copy_host_ptr() )
                .dims( test_set.len() )
                .host_data( &test_set )
                .build().unwrap();

        // Load the program.
        let kernel = pro_que.create_kernel("search").unwrap()
                .arg_buf(&raw_data)
                .arg_buf(&begin_indexes)
                .arg_buf(&end_indexes)
                .arg_buf(&test_set)
                .arg_scl( test_set.len() as i32 );

        let start_calculation = std::time::Instant::now();
        unsafe { kernel.enq().unwrap(); }
        let stop_calculation = std::time::Instant::now();
        let calculation_duration = stop_calculation.duration_since( start_calculation );
        println!("{}.{:06} s", calculation_duration.as_secs(), calculation_duration.subsec_nanos() / 1000 );
        0
    }
}

/// Dummy implementation when GPU support is not included.
#[cfg(not(feature="gpu"))]
 pub trait WithGpu
 {
 }

#[cfg(not(feature="gpu"))]
impl WithGpu for i32
{
}

#[cfg(not(feature="gpu"))]
impl WithGpu for f32
{
}


/// Attaches the buffer into scalar sets.
fn attach_buffer<'a, T>( data: &'a [T] ) -> SetsForEvaluation<T>
where
    T: FromI32 + std::clone::Clone + std::marker::Send + std::marker::Sync + ro_scalar_set::Value + WithGpu,
{

    // Divide to buffers.
    let mut buffer = data;
    let mut buffers: Vec<ro_scalar_set::RoScalarSet<T>> = Vec::new();
    loop
    {

        // Attach scalar set to the buffer.
        let result = match ro_scalar_set::RoScalarSet::attach( buffer )
        {
            Ok( result ) => result,
            Err( _ ) => break,
        };
        buffer = result.1;
        buffers.push( result.0 );
    }
    return SetsForEvaluation::new( data, buffers );
}

/// Evaluates a single set.
fn evaluate_set_cpu<T>(
    test_set: &ro_scalar_set::RoScalarSet<T>,
    set: &ro_scalar_set::RoScalarSet<T>,
) -> u32
where
    T: FromI32 + std::clone::Clone + std::marker::Send + std::marker::Sync + ro_scalar_set::Value,
{
    // Test if any of values in the set are found from the current scalar set.
    if test_set.any( set ) { 1 } else { 0 }
}


/// Converts a slice to 32-bit integer.
fn as_slice<'a, T>(
    buffer: *const T,
    integer_count: usize,
) -> &'a [T]
{
    unsafe {
        return slice::from_raw_parts( buffer, integer_count );
    }
}
