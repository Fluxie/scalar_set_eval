extern crate std;

use std::io::BufWriter;
use std::io::prelude::*;
use std::path::Path;

use evaluation::*;
use enumerations::*;
use utility::*;

/// Configurable parameters for the test.
pub struct Parameters<'a>
{
    report: &'a String,
    min_value: i32,
    max_value: i32,
    use_floats: bool,
    engine: EvaluationEngine,
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

pub fn run_tests(
    report_pattern: &String,
    min_value: i32,
    max_value: i32,
    floats: bool,
    eval_engine: EvaluationEngine,
)
{
    let params = Parameters {
        report: report_pattern,
        min_value: min_value,
        max_value: max_value,
        use_floats: floats,
        engine: eval_engine,
    };
    run_test( params );
}

/// Executes one test with the given parameters.
fn run_test( parameters: Parameters )
{
    // Define test material.
    // let set_sizes: Vec<i32> = vec! { 10, 100, 1000 };
    // let set_counts: Vec<i32> = vec! { 10, 100, 1000  };
    let set_sizes: Vec<i32> = vec![10, 100, 1000, 10000];
    let set_counts: Vec<i32> = vec![10, 100, 1000, 10000, 100000];
    let test_set_sizes: Vec<i32> = vec![10, 100, 1000, 10000];

    // Generate test files.
    generate_test_files( &set_sizes, &set_counts, &parameters );

    // Run the tests.
    let mut results: Vec<TestResult> = Vec::new();
    for set_size in &set_sizes
    {
        for set_count in &set_counts
        {
            for test_set_size in &test_set_sizes
            {

                // Identify the current test.
                let file_name = get_set_file_name( set_count, set_size, &parameters.use_floats );
                if !Path::new( &file_name ).exists()
                {
                    panic!( "Generated file not found." );
                }

                // Run and measure.
                println!( "Running test set {}...", file_name );
                let evaluation_result;
                if parameters.use_floats
                {
                    evaluation_result = evaluate::<f32>(
                        &file_name,
                        *test_set_size,
                        parameters.min_value,
                        parameters.max_value,
                        &parameters.engine,
                    );
                }
                else
                {
                    evaluation_result = evaluate::<i32>(
                        &file_name,
                        *test_set_size,
                        parameters.min_value,
                        parameters.max_value,
                        &parameters.engine,
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
    let report = std::fs::File::create( parameters.report ).expect( "Failed to open the report." );
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
                "Duration",
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

/// Generates test files for a test.
fn generate_test_files(
    set_sizes: &Vec<i32>,
    set_counts: &Vec<i32>,
    parameters: &Parameters,
)
{
    // Generate test files.
    for set_size in set_sizes
    {
        for set_count in set_counts
        {
            // Reuse existing files if available.
            let file_name = get_set_file_name( set_count, set_size, &parameters.use_floats );
            if Path::new( &file_name ).exists()
            {
                continue;
            }

            println!( "Generating test set {}...", file_name );
            if parameters.use_floats
            {
                generate::<f32>(
                    &file_name,
                    *set_count,
                    *set_size,
                    parameters.min_value,
                    parameters.max_value,
                );
            }
            else
            {
                generate::<i32>(
                    &file_name,
                    *set_count,
                    *set_size,
                    parameters.min_value,
                    parameters.max_value,
                );
            }
        }
    }
}
