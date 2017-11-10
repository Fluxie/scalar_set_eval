
/// Tarit for converting generated i32 to target test type.
pub trait FromI32
{
    fn from_i32( value: &i32 ) -> Self;
}

/// We use floats
impl FromI32 for f32
{
    fn from_i32( value: &i32 ) -> f32
    {
        return value.clone() as f32;
    }
}

/// We use floats
impl FromI32 for i32
{
    fn from_i32( value: &i32 ) -> i32
    {
        return value.clone();
    }
}
