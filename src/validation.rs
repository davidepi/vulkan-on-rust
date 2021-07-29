use ash::version::EntryV1_0;
use std::{collections::HashSet, ffi::CStr, os::raw::c_char};

#[cfg(debug_assertions)]
pub const VALIDATION_ON: bool = true;
#[cfg(not(debug_assertions))]
pub const VALIDATION_ON: bool = false;

pub fn check_validation_layers_support(entry: &ash::Entry, required_layers: &[&str]) -> bool {
    let layer_properties = entry
        .enumerate_instance_layer_properties()
        .expect("Failed to enumerate layer properties");
    let available_layers = layer_properties
        .iter()
        .map(|x| &x.layer_name[..])
        .map(cstr_to_str)
        .collect::<HashSet<_>>();
    required_layers
        .iter()
        .fold(!layer_properties.is_empty(), |acc, &req_layer| {
            acc && available_layers.contains(req_layer)
        })
}

fn cstr_to_str(cstr: &[c_char]) -> String {
    let raw_string = unsafe {
        let pointer = cstr.as_ptr();
        CStr::from_ptr(pointer)
    };

    raw_string
        .to_str()
        .expect("Failed to c string to rust String.")
        .to_owned()
}
