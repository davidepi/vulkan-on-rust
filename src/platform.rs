use ash;

pub fn required_extension_names() -> Vec<*const i8> {
    let retval;
    #[cfg(target_os = "macos")]
    {
        retval = vec![
            ash::extensions::khr::Surface::name().as_ptr(),
            ash::extensions::mvk::MacOSSurface::name().as_ptr(),
            #[cfg(debug_assertions)]
            ash::extensions::ext::DebugUtils::name().as_ptr(),
        ]
    }
    #[cfg(target_os = "windows")]
    {
        retval = vec![
            ash::extensions::khr::Surface::name().as_ptr(),
            ash::extensions::khr::Win32Surface::name().as_ptr(),
            #[cfg(debug_assertions)]
            ash::extensions::ext::DebugUtils::name().as_ptr(),
        ]
    }
    #[cfg(target_os = "linux")]
    {
        retval = vec![
            ash::extensions::khr::Surface::name().as_ptr(),
            ash::extensions::khr::XlibSurface::name().as_ptr(),
            #[cfg(debug_assertions)]
            ash::extensions::ext::DebugUtils::name().as_ptr(),
        ]
    }
    retval
}
