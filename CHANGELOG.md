# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.1.0] - 2025-03-05
### Added
- Totals for number of parameters and FLOPS in table headings
- Args and kwargs for each operation in hover labels
- `torch.compile` as alternative to and fallback for `torch.fx.symbolic_trace`
- Dropdown menu to choose criteria for node sizes and colors
- Added overview panel to portable figure that responds to slider
### Fixed
- Overlapping of skip connections between many nodes in a straight line
- Hang or OOM for models with too many nodes. Now falling back to greedy layout algorithm
- Exceptions for some degenerate models and special cases
### Changed
- Replaced `torchprofile.profile_macs` with `torch.profiler.profile` for FLOPS calculations 
- Formatting of activations, number of parameters and FLOPS
- Moved user options to data classes (see "mavoptions.py")
### Deprecated
- Renamed `MAV.export_html` to `MAV.export_static_html`

## [1.0.4] - 2025-01-07
### Added
- Initial release of the project.
- Patch version is due to some PyPI experiments to get the readme and dependencies right
