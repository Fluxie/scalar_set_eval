extern crate ro_scalar_set;
extern crate std;
extern crate rayon;
extern crate ocl;

use self::ocl::ProQue;
use self::ocl::Buffer;
use self::ocl::MemFlags;

use traits;
use self::rayon::prelude::*;

/// Declares a set that can be evaluated.
pub struct SetsForEvaluation<'a,T,>
where
    T: 'a + traits::FromI32 + std::clone::Clone + std::marker::Send + std::marker::Sync + ro_scalar_set::Value + WithGpu
{
    raw_data: &'a[T],
    sets: Vec<ro_scalar_set::RoScalarSet<'a,T>>,
}

impl<'a,T> SetsForEvaluation<'a,T>
where
    T: traits::FromI32 + std::clone::Clone + std::marker::Send + std::marker::Sync + ro_scalar_set::Value + WithGpu,
{
    /// Initializes new set evaluator from a collection of sets.
    pub fn new(
        raw_data: &'a[T],
        sets: Vec<ro_scalar_set::RoScalarSet<'a,T>>,
    ) -> SetsForEvaluation<'a,T>
    {
        return SetsForEvaluation { raw_data: raw_data, sets: sets };
    }

    /// Evaluates the sets with CPU.
    pub fn evaluate_with_cpu(
        &self,
        test_set: &[T],
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

    /// Evaluates the sets with GPU.
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
impl WithGpu for i32
{
        /// Evaluates the given data set with GPU.
    fn evaluate_with_gpu(
        raw_data: &[i32],
        sets: &Vec<ro_scalar_set::RoScalarSet<i32>>,
        test_set: &[i32],
    ) -> u32
    {
        panic!("Not implemented");
    }
}

/// GPU evaluation support for floats.
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

/// Evaluates a single set.
fn evaluate_set_cpu<T>(
    test_set: &[T],
    set: &ro_scalar_set::RoScalarSet<T>,
) -> u32
where
    T: traits::FromI32 + std::clone::Clone + std::marker::Send + std::marker::Sync + ro_scalar_set::Value,
{
    // Test if any of values in the set are found from the current scalar set.
    for t in test_set
    {
        // Match found?
        if set.contains( t )
        {
            return 1;
        }
    }
    return 0;
}