#[macro_use]
extern crate serde_derive;
extern crate docopt;
extern crate ro_scalar_set;
extern crate rand;
extern crate memmap;
extern crate rayon;

use docopt::Docopt;

mod enumerations;
mod evaluation;
// use evaluation::WithGpu;
mod traits;
mod test;
mod utility;

use enumerations::*;

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
            utility::generate::<f32>(
                &args.arg_file,
                args.arg_sets,
                args.arg_values,
                args.arg_minvalue,
                args.arg_maxvalue,
            );
        }
        else
        {
            utility::generate::<i32>(
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
            let ( match_count, duration ) = evaluation::evaluate::<f32>(
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
            let ( match_count, duration ) = evaluation::evaluate::<i32>(
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
        test::run_tests(
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
