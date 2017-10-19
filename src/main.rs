#[macro_use]
extern crate serde_derive;
extern crate docopt;
extern crate ro_scalar_set;
extern crate rand;
extern crate memmap;
extern crate rayon;


use std::io::BufWriter;
use std::path::Path;
use std::io::prelude::*;
use std::slice;
use std::collections::HashSet;
use std::iter::FromIterator;

use memmap::{Mmap, Protection};
use rand::distributions::{IndependentSample, Range};

use docopt::Docopt;
use rayon::prelude::*;

const USAGE: &'static str = "
Scalar Set Evaluator.

Usage:
  scalar_set_eval new <file> <minvalue> <maxvalue> <values> <sets>
  scalar_set_eval eval <file> <minvalue> <maxvalue> <values> [<sets>]
  scalar_set_eval test <report> <minvalue> <maxvalue> [<values>] [<sets>]
  scalar_set_eval (-h | --help)
  scalar_set_eval --version
ro_scalar_set
Options:
  -h --help     Show this screen.
  --version     Show version.
  --mt          Multi-threaded
";

#[derive(Debug, Deserialize)]
struct Args {
    arg_file: String,
    arg_report: String,
    arg_minvalue: i32,
    arg_maxvalue: i32,
    arg_sets: i32,
    arg_values: i32,
    flag_version: bool,
    flag_mt: bool,
    cmd_new: bool,
    cmd_eval: bool,
    cmd_test: bool,
}

/// Results of a single test.
///  # Members
/// * set_size Number of values in a set.
/// * set_count Number of sets
/// * test_set_size Number of values in the test set
/// * duration The length of the evaluation
/// * matches The number of sets that have a value matching with a value in the test set.
struct TestResult {
  set_size: i32,
  set_count: i32,
  test_set_size: i32,

  duration: std::time::Duration,
  matches: u32,
}

fn main() {
  let args: Args = Docopt::new(USAGE)
                            .and_then(|d| d.deserialize())
                            .unwrap_or_else(|e| e.exit());

  // Determine action.
   let start = std::time::Instant::now();
  if args.cmd_new {
    generate( &args.arg_file, args.arg_sets, args.arg_values, args.arg_minvalue, args.arg_maxvalue );
  }
  else if args.cmd_eval {
    let ( match_count, duration ) = evaluate( &args.arg_file, args.arg_values, args.arg_minvalue, args.arg_maxvalue );
    println!("Found {} matches in {}.{:06} s", match_count, duration.as_secs(), duration.subsec_nanos() / 1000 );
  }
  else if args.cmd_test {
    test( &args.arg_report, args.arg_minvalue, args.arg_maxvalue );
  }
  else {
    println!("{}", "No tests selected." );
  }
  let stop = std::time::Instant::now();
  let duration = stop.duration_since( start );
  println!("Operation took {}.{:06} s.",duration.as_secs(), duration.subsec_nanos() / 1000 );
}


fn test(
  report: &String,
  min_value: i32,
  max_value: i32,
)
{
  // Define test material.
  // let set_sizes: Vec<i32> = vec! { 10, 100, 1000 };
  // let set_counts: Vec<i32> = vec! { 10, 100, 1000  };
  let set_sizes: Vec<i32> = vec! { 10, 100, 1000, 10000 };
  let set_counts: Vec<i32> = vec! { 10, 100, 1000, 10000, 100000 };
  let test_set_sizes: Vec<i32> = vec! { 10, 100, 1000, 10000 };

  // Generate test files.
  for set_size in &set_sizes {
    for set_count in &set_counts {

      // Reuse existing files if available.
      let file_name = format!( "{}_sets_with_{}_values.bin", set_count, set_size,  );
      if Path::new( &file_name ).exists() {
        continue;
      }

      println!("Generating test set {}...", file_name );
      generate( &file_name, *set_count, *set_size, min_value, max_value );
    }
  }

  // Run the tests.
  let mut results: Vec<TestResult> = Vec::new();
  for set_size in &set_sizes {
    for set_count in &set_counts {
      for test_set_size in &test_set_sizes {

        // Identify the current test.
        let file_name = format!( "{}_sets_with_{}_values.bin", set_count, set_size,  );
        if ! Path::new( &file_name ).exists() {
          panic!( "Generated file not found." );
        }

        // Run and measure.
        println!("Running test set {}...", file_name );
        let ( match_count, duration ) = evaluate( &file_name, *test_set_size, min_value, max_value );

        // Collect results for reporting.
        let result = TestResult {
            set_size: *set_size, set_count: *set_count,
            test_set_size: *test_set_size,
            duration: duration, matches: match_count  };
        results.push( result );
      }
    }
  }

  // Report the results.
  let report = std::fs::File::create( report ).expect( "Failed to open the report." );
  let mut report = BufWriter::with_capacity( 1024 * 1024, report );

  let mut current_set_size = results[ 0 ].set_size;
  let mut write_header: bool = true;
  for result in &results {

    // Always write header when we ancounter a new set size.
    if result.set_size != current_set_size {
      write_header = true;
      current_set_size = result.set_size;
    }

    // Header?
    if write_header {

      writeln!( &mut report, "" ).expect( "Writing report failed." );
      writeln!( &mut report,
          "Number of values in a set: {}", current_set_size )
          .expect( "Writing report failed." );;
      writeln!( &mut report,
          "|{:14}|{:14}|{:14}|{:14}|",
          "Sets", "Test set size", "Matching sets", "Duration" ).expect( "Writing report failed." );
      writeln!( &mut report,
          "|{:-<14}|{:-<14}|{:-<14}|{:-<14}|",
          "-", "-", "-", "-" ).expect( "Writing report failed." );

      write_header = false;
    }

    // Report results of a single test.
    writeln!( &mut report,
        "|{:14}|{:14}|{:14}|{:5}.{:06} s|",
        result.set_count, result.test_set_size, result.matches,
        result.duration.as_secs(), result.duration.subsec_nanos() / 1000 )
        .expect( "Writing report failed." );

  }

}

fn generate(
  file: &String,
  set_count: i32,
  values_in_set: i32,
  min_value: i32,
  max_value: i32
) {

  println!("Generating {} sets to {}...", set_count, file );
  let mut file = BufWriter::with_capacity( 1024 * 1024, std::fs::File::create( file ).expect( "Failed to open the file." ) );

  // Prepare RNG.
  let between = Range::new( min_value, max_value );

  // Prepare array for holding the results.
  let sets: Vec<i32> = ( 0..set_count ).collect();
  let sets: Vec<_> = sets.par_iter()
      .map( |_| {
        let values = generate_values( values_in_set, &between );
        let result = ro_scalar_set::ro_scalar_set::RoScalarSet::new( values.as_slice() );
        return result;
      } )
      .collect();

  // Serialize the sets to a file.
  for set in sets {

      set.serialize( &mut file ).expect( "Writing scalar set to a file failed." );
  }
}

/// Evaluates integer sets.
fn evaluate<'a>(
  file: &String,
  values_in_set: i32,
  min_value: i32,
  max_value: i32
) -> ( u32, std::time::Duration )
{
  // Construct test vector.
  let between = Range::new( min_value, max_value );
  let test_set = generate_values( values_in_set, &between );

  // Open file for reading.
  let file = std::fs::File::open( file ).expect( "Failed to open the file." );
  let file = Mmap::open( &file, Protection::Read ).expect( "Failed to map the file");  
  {
    let integer_count = file.len() / 4;
    let buffer: *const i32 = file.ptr() as *const i32;
    let buffer = as_i32( buffer, integer_count );
    {
      // Divide
      let sets = attach_buffer( &buffer );
      
      // Run tests for each set.
      let start = std::time::Instant::now();
      let match_counter = sets.par_iter().map( |s| evaluate_set( test_set.as_slice(), &s ) ).sum();      
      let stop = std::time::Instant::now();
      let duration = stop.duration_since( start );
      return ( match_counter, duration );
    }   
  }
}

/// Evaluates a single set.
fn evaluate_set(
  test_set: &[i32],
  set: &ro_scalar_set::RoScalarSet<i32>

) -> u32 {

    // Test if any of values in the set are found from the current scalar set.
    for t in test_set
    {
      if set.contains( *t ) {
        return 1;
      }
    }
    return 0;
}

fn generate_values(
  values_in_set: i32,
  between: &Range<i32>
) -> Vec<i32> {
 
  // Collect random values.
  let mut rng = rand::thread_rng();
  let mut values: HashSet<i32> = HashSet::new();
  values.reserve( values_in_set as usize );
  while values.len() < values_in_set as usize {

      let v = between.ind_sample( &mut rng );
      values.insert( v );
  }

  let values: Vec<i32> = Vec::from_iter( values.into_iter() );
  return values;
}

/// Attaches the buffer into scalar sets.
fn attach_buffer<'a>(
  buffer: &'a[i32]
) -> Vec< ro_scalar_set::RoScalarSet<'a, i32>> {
 
  // Divide to buffers.
  let mut buffer = buffer;
  let mut buffers: Vec<ro_scalar_set::RoScalarSet<i32>> = Vec::new();
  loop {
      
    // Attach scalar set to the buffer.
    let result = match ro_scalar_set::RoScalarSet::attach( buffer ) {
      Ok( result ) => result,
      Err( _ ) => return buffers
    };
    buffer = result.1;
    buffers.push(result.0);
  }; 
}

/// Converts a slice to 32-bit integer.
fn as_i32<'a>(
  buffer: *const i32,
  integer_count: usize
) -> &'a[i32]
{
    unsafe { return slice::from_raw_parts( buffer, integer_count ); }
}
