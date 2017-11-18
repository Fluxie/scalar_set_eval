#[macro_use]
extern crate serde_derive;
extern crate docopt;
extern crate ro_scalar_set;
extern crate rand;
extern crate memmap;
extern crate rayon;


use std::collections::HashSet;
use std::io::BufWriter;
use std::io::prelude::*;
use std::path::Path;
use std::slice;

use memmap::{Mmap, Protection};
use rand::distributions::{IndependentSample, Range};
use rayon::prelude::*;

use docopt::Docopt;

mod evaluation;
// use evaluation::WithGpu;
mod traits;

const USAGE: &'static str = "
Scalar Set Evaluator.

Usage:
  scalar_set_eval new [--floats] [--gpu] <file> <minvalue> <maxvalue> <values> <sets>
  scalar_set_eval eval [--floats] [--gpu] <file> <minvalue> <maxvalue> <values> [<sets>]
  scalar_set_eval test [--floats] [--gpu] <report> <minvalue> <maxvalue> [<values>] [<sets>]
  scalar_set_eval (-h | --help)
  scalar_set_eval --version
ro_scalar_set
Options:
  -h --help     Show this screen.
  --version     Show version.
  --mt          Multi-threaded
  --floats      Run tests using floating points
  --gpu         Run tests on GPU
";

#[derive(Debug, Deserialize)]
struct Args
{
    arg_file: String,
    arg_report: String,
    arg_minvalue: i32,
    arg_maxvalue: i32,
    arg_sets: i32,
    arg_values: i32,
    flag_version: bool,
    flag_mt: bool,
    flag_floats: bool,
    flag_gpu: bool,
    cmd_new: bool,
    cmd_eval: bool,
    cmd_test: bool,
}

/// The enegine used to evluate the set.
enum EvaluationEngine
{
    Cpu,
    Gpu,
}

/// Results of a single test.
///  # Members
/// * set_size Number of values in a set.
/// * set_count Number of sets
/// * test_set_size Number of values in the test set
/// * duration The length of the evaluation
/// * matches The number of sets that have a value matching with a value in the test set.
struct TestResult
{
    set_size: i32,
    set_count: i32,
    test_set_size: i32,

    duration: std::time::Duration,
    matches: u32,
}

fn main()
{

    // TEst
    let args: Args = Docopt::new( USAGE )
        .and_then( |d| d.deserialize() )
        .unwrap_or_else( |e| e.exit() );

    let eval_engine = if args.flag_gpu
    {
        EvaluationEngine::Gpu
    }
    else
    {
        EvaluationEngine::Cpu
    };

    // Determine action.
    let start = std::time::Instant::now();
    if args.cmd_new
    {
        // Data type
        if args.flag_floats
        {
            generate::<f32>(
                &args.arg_file,
                args.arg_sets,
                args.arg_values,
                args.arg_minvalue,
                args.arg_maxvalue,
            );
        }
        else
        {
            generate::<i32>(
                &args.arg_file,
                args.arg_sets,
                args.arg_values,
                args.arg_minvalue,
                args.arg_maxvalue,
            );
        }
    }
    else if args.cmd_eval
    {
        // Data type
        if args.flag_floats
        {
            let ( match_count, duration ) = evaluate::<f32>(
                &args.arg_file,
                args.arg_values,
                args.arg_minvalue,
                args.arg_maxvalue,
                &eval_engine,
            );
            println!(
                "Found {} matches in {}.{:06} s",
                match_count,
                duration.as_secs(),
                duration.subsec_nanos() / 1000
            );
        }
        else
        {
            let ( match_count, duration ) = evaluate::<i32>(
                &args.arg_file,
                args.arg_values,
                args.arg_minvalue,
                args.arg_maxvalue,
                &eval_engine,
            );
            println!(
                "Found {} matches in {}.{:06} s",
                match_count,
                duration.as_secs(),
                duration.subsec_nanos() / 1000
            );
        }
    }
    else if args.cmd_test
    {
        test(
            &args.arg_report,
            args.arg_minvalue,
            args.arg_maxvalue,
            args.flag_floats,
            eval_engine,
        );
    }
    else
    {
        println!( "{}", "No tests selected." );
    }
    let stop = std::time::Instant::now();
    let duration = stop.duration_since( start );
    println!(
        "Operation took {}.{:06} s.",
        duration.as_secs(),
        duration.subsec_nanos() / 1000
    );
}


fn test( 
    report: &String,
    min_value: i32,
    max_value: i32,
    floats: bool,
    eval_engine: EvaluationEngine,
)
{
    // Define test material.
    // let set_sizes: Vec<i32> = vec! { 10, 100, 1000 };
    // let set_counts: Vec<i32> = vec! { 10, 100, 1000  };
    let set_sizes: Vec<i32> = vec![10, 100, 1000, 10000];
    let set_counts: Vec<i32> = vec![10, 100, 1000, 10000, 100000];
    let test_set_sizes: Vec<i32> = vec![10, 100, 1000, 10000];

    // Generate test files.
    for set_size in &set_sizes
    {
        for set_count in &set_counts
        {

            // Reuse existing files if available.
            let file_name = get_set_file_name( set_count, set_size, &floats );
            if Path::new( &file_name ).exists()
            {
                continue;
            }

            println!( "Generating test set {}...", file_name );
            if floats
            {
                generate::<f32>( &file_name, *set_count, *set_size, min_value, max_value );
            }
            else
            {
                generate::<i32>( &file_name, *set_count, *set_size, min_value, max_value );
            }
        }
    }

    // Run the tests.
    let mut results: Vec<TestResult> = Vec::new();
    for set_size in &set_sizes
    {
        for set_count in &set_counts
        {
            for test_set_size in &test_set_sizes
            {

                // Identify the current test.
                let file_name = get_set_file_name( set_count, set_size, &floats );
                if !Path::new( &file_name ).exists()
                {
                    panic!( "Generated file not found." );
                }

                // Run and measure.
                println!( "Running test set {}...", file_name );
                let evaluation_result;
                if floats
                {
                    evaluation_result = evaluate::<f32>(
                        &file_name,
                        *test_set_size,
                        min_value,
                        max_value,
                        &eval_engine,
                    );
                }
                else
                {
                    evaluation_result = evaluate::<i32>(
                        &file_name,
                        *test_set_size,
                        min_value,
                        max_value,
                        &eval_engine,
                    );
                }
                let ( match_count, duration ) = evaluation_result;

                // Collect results for       reporting.
                let result = TestResult {
                    set_size: *set_size,
                    set_count: *set_count,
                    test_set_size: *test_set_size,
                    duration: duration,
                    matches: match_count,
                };
                results.push( result );
            }
        }
    }

    // Report the results.
    let report = std::fs::File::create( report ).expect( "Failed to open the report." );
    let mut report = BufWriter::with_capacity( 1024 * 1024, report );

    let mut current_set_size = results[0].set_size;
    let mut write_header: bool = true;
    for result in &results
    {

        // Always write header when we ancounter a new set size.
        if result.set_size != current_set_size
        {
            write_header = true;
            current_set_size = result.set_size;
        }

        // Header?
        if write_header
        {

            writeln!( &mut report, "" ).expect( "Writing report failed." );
            writeln!(
                &mut report,
                "Number of values in a set: {}",
                current_set_size
            ).expect( "Writing report failed." );;
            writeln!( &mut report, "" ).expect( "Writing report failed." );
            writeln!(
                &mut report,
                "|{:14}|{:14}|{:14}|{:14}|",
                "Sets",
                "Test set size",
                "Matching sets",
                "Duration"
            ).expect( "Writing report failed." );
            writeln!(
                &mut report,
                "|{:-<13}:|{:-<13}:|{:-<13}:|{:-<13}:|",
                "-",
                "-",
                "-",
                "-"
            ).expect( "Writing report failed." );

            write_header = false;
        }

        // Report results of a single test.
        writeln!(
            &mut report,
            "|{:14}|{:14}|{:14}|{:5}.{:06} s|",
            result.set_count,
            result.test_set_size,
            result.matches,
            result.duration.as_secs(),
            result.duration.subsec_nanos() / 1000
        ).expect( "Writing report failed." );

    }
}

fn generate<T>( 
    file: &String,
    set_count: i32,
    values_in_set: i32,
    min_value: i32,
    max_value: i32,
) where
    T: traits::FromI32 + std::clone::Clone + std::marker::Send + std::marker::Sync + ro_scalar_set::Value,
{

    println!( "Generating {} sets to {}...", set_count, file );
    let mut file = BufWriter::with_capacity(
        1024 * 1024,
        std::fs::File::create( file ).expect( "Failed to open the file." ),
    );

    // Prepare RNG.
    let between = Range::new( min_value, max_value );

    // Prepare array for holding the results.
    let sets: Vec<i32> = ( 0..set_count ).collect();
    let sets: Vec<_> = sets.par_iter()
        .map( |_| {
            let values = generate_values::<T>( values_in_set, &between );
            let result = ro_scalar_set::ro_scalar_set::RoScalarSet::new( values.as_slice() );
            return result;
        } )
        .collect();

    // Serialize the sets to a file.
    for set in sets
    {
        set.serialize( &mut file ).expect(
            "Writing scalar set to a file failed.",
        );
    }
}

/// Evaluates integer sets.
fn evaluate<'a, T>( 
    file: &String,
    values_in_set: i32,
    min_value: i32,
    max_value: i32,
    eval_engine: &EvaluationEngine,
) -> ( u32, std::time::Duration )
where
    T: traits::FromI32 + std::clone::Clone + std::marker::Send + std::marker::Sync + ro_scalar_set::Value + evaluation::WithGpu,
{
    // Construct test vector.
    let between = Range::new( min_value, max_value );
    let test_set = generate_values( values_in_set, &between );

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


fn generate_values<T>( 
    values_in_set: i32,
    between: &Range<i32>,
) -> Vec<T>
where
    T: traits::FromI32,
{

    // Collect random values.
    let mut rng = rand::thread_rng();
    let mut generated_values: HashSet<i32> = HashSet::new();
    generated_values.reserve( values_in_set as usize );
    while generated_values.len() < values_in_set as usize
    {

        let v = between.ind_sample( &mut rng );
        generated_values.insert( v );
    }

    // Convert to appropriate type.
    let mut values: Vec<T> = Vec::new();
    for v in &generated_values
    {
        values.push( T::from_i32( v ) );
    }
    return values;
}

/// Attaches the buffer into scalar sets.
fn attach_buffer<'a, T>( data: &'a [T] ) -> evaluation::SetsForEvaluation<T>
where
    T: traits::FromI32 + std::clone::Clone + std::marker::Send + std::marker::Sync + ro_scalar_set::Value + evaluation::WithGpu,
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
    return evaluation::SetsForEvaluation::new( data, buffers );
}

/// Gets file name for a set.
fn get_set_file_name( 
    set_count: &i32,
    set_size: &i32,
    floats: &bool,
) -> String
{
    let file_name;
    if *floats
    {
        file_name = format!( "f32_{}_sets_with_{}_values.bin", set_count, set_size,  );
    }
    else
    {
        file_name = format!( "i32_{}_sets_with_{}_values.bin", set_count, set_size,  );
    }

    file_name
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
