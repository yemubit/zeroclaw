//! Hardware discovery — USB device enumeration and introspection.
//!
//! See `docs/hardware-peripherals-design.md` for the full design.

pub mod registry;

#[cfg(feature = "hardware")]
pub mod discover;

#[cfg(feature = "hardware")]
pub mod introspect;

use crate::config::Config;
use anyhow::Result;

/// Handle `zeroclaw hardware` subcommands.
#[allow(clippy::module_name_repetitions)]
pub fn handle_command(cmd: crate::HardwareCommands, _config: &Config) -> Result<()> {
    #[cfg(not(feature = "hardware"))]
    {
        println!("Hardware discovery requires the 'hardware' feature.");
        println!("Build with: cargo build --features hardware");
        return Ok(());
    }

    #[cfg(feature = "hardware")]
    match cmd {
        crate::HardwareCommands::Discover => run_discover(),
        crate::HardwareCommands::Introspect { path } => run_introspect(&path),
        crate::HardwareCommands::Info { chip } => run_info(&chip),
    }
}

#[cfg(feature = "hardware")]
fn run_discover() -> Result<()> {
    let devices = discover::list_usb_devices()?;

    if devices.is_empty() {
        println!("No USB devices found.");
        println!();
        println!("Connect a board (e.g. Nucleo-F401RE) via USB and try again.");
        return Ok(());
    }

    println!("USB devices:");
    println!();
    for d in &devices {
        let board = d.board_name.as_deref().unwrap_or("(unknown)");
        let arch = d.architecture.as_deref().unwrap_or("—");
        let product = d.product_string.as_deref().unwrap_or("—");
        println!(
            "  {:04x}:{:04x}  {}  {}  {}",
            d.vid, d.pid, board, arch, product
        );
    }
    println!();
    println!("Known boards: nucleo-f401re, nucleo-f411re, arduino-uno, arduino-mega, cp2102");

    Ok(())
}

#[cfg(feature = "hardware")]
fn run_introspect(path: &str) -> Result<()> {
    let result = introspect::introspect_device(path)?;

    println!("Device at {}:", result.path);
    println!();
    if let (Some(vid), Some(pid)) = (result.vid, result.pid) {
        println!("  VID:PID     {:04x}:{:04x}", vid, pid);
    } else {
        println!("  VID:PID     (could not correlate with USB device)");
    }
    if let Some(name) = &result.board_name {
        println!("  Board       {}", name);
    }
    if let Some(arch) = &result.architecture {
        println!("  Architecture {}", arch);
    }
    println!("  Memory map  {}", result.memory_map_note);

    Ok(())
}

#[cfg(feature = "hardware")]
fn run_info(chip: &str) -> Result<()> {
    #[cfg(feature = "probe")]
    {
        match info_via_probe(chip) {
            Ok(()) => return Ok(()),
            Err(e) => {
                println!("probe-rs attach failed: {}", e);
                println!();
                println!("Ensure Nucleo is connected via USB. The ST-Link is built into the board.");
                println!("No firmware needs to be flashed — probe-rs reads chip info over SWD.");
                return Err(e.into());
            }
        }
    }

    #[cfg(not(feature = "probe"))]
    {
        println!("Chip info via USB requires the 'probe' feature.");
        println!();
        println!("Build with: cargo build --features hardware,probe");
        println!();
        println!("Then run: zeroclaw hardware info --chip {}", chip);
        println!();
        println!("This uses probe-rs to attach to the Nucleo's ST-Link over USB");
        println!("and read chip info (memory map, etc.) — no firmware on target needed.");
        Ok(())
    }
}

#[cfg(all(feature = "hardware", feature = "probe"))]
fn info_via_probe(chip: &str) -> anyhow::Result<()> {
    use probe_rs::config::MemoryRegion;
    use probe_rs::{Permissions, Session};

    println!("Connecting to {} via USB (ST-Link)...", chip);
    let session = Session::auto_attach(chip, Permissions::default())
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let target = session.target();
    println!();
    println!("Chip: {}", target.name);
    println!("Architecture: {:?}", session.architecture());
    println!();
    println!("Memory map:");
    for region in target.memory_map.iter() {
        match region {
            MemoryRegion::Ram(ram) => {
                let start = ram.range.start;
                let end = ram.range.end;
                let size_kb = (end - start) / 1024;
                println!(
                    "  RAM: 0x{:08X} - 0x{:08X} ({} KB)",
                    start, end, size_kb
                );
            }
            MemoryRegion::Nvm(flash) => {
                let start = flash.range.start;
                let end = flash.range.end;
                let size_kb = (end - start) / 1024;
                println!(
                    "  Flash: 0x{:08X} - 0x{:08X} ({} KB)",
                    start, end, size_kb
                );
            }
            _ => {}
        }
    }
    println!();
    println!("Info read via USB (SWD) — no firmware on target needed.");
    Ok(())
}
