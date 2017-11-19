extern crate rand;
extern crate std;
extern crate ro_scalar_set;
extern crate rayon;


use std::collections::HashSet;
use std::io::BufWriter;

use rayon::prelude::*;

use rand::distributions::{IndependentSample, Range};

use traits::*;

pub fn generate<T>(
    file: &String,
    set_count: i32,
    values_in_set: i32,
    min_value: i32,
    max_value: i32,
) where
    T: FromI32 + std::clone::Clone + std::marker::Send + std::marker::Sync + ro_scalar_set::Value,
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

pub fn generate_values<T>(
    values_in_set: i32,
    between: &Range<i32>,
) -> Vec<T>
where
    T: FromI32,
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

/// Gets file name for a set.
pub fn get_set_file_name(
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