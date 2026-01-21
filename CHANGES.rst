Revisions
---------

2026.1.20

- Fix reading ImagejRoi.props.
- Add ImagejRoi.properties property to decode and encode ImagejRoi.props.

2026.1.8

- Improve code quality.
- Drop support for Python 3.10.

2025.12.12

- Move tests to separate module.

2025.5.10

- Support Python 3.14.

2025.2.20

- Drop support for Python 3.9.

2024.9.15

- Improve typing.
- Deprecate Python 3.9, support Python 3.13.

2024.5.24

- Fix docstring examples not correctly rendered on GitHub.

2024.3.20

- Fix writing generator of ROIs (#9).

2024.1.10

- Support text rotation.
- Improve text rendering.
- Avoid array copies.
- Limit size read from files.

2023.8.30

- Fix linting issues.
- Add py.typed marker.

2023.5.12

- Improve object repr and type hints.
- Drop support for Python 3.8 and numpy < 1.21 (NEP29).

2023.2.12

- Delay import of zipfile.
- Verify shape of coordinates on write.

2022.9.19

- Fix integer coordinates to -5000..60536 conforming with ImageJ (breaking).
- Add subpixel_coordinates in frompoints for out-of-range integer coordinates.

2022.7.29

- Update metadata.

2022.3.18

- Fix creating ROIs from float coordinates exceeding int16 range (#7).
- Fix bottom-right bounds in ImagejRoi.frompoints.

2022.2.2

- Add type hints.
- Change ImagejRoi to dataclass.
- Drop support for Python 3.7 and numpy < 1.19 (NEP29).

2021.6.6

- Add enums for point types and sizes.

2020.11.28

- Support group attribute.
- Add roiread and roiwrite functions (#3).
- Use UUID as default name of ROI in ImagejRoi.frompoints (#2).

2020.8.13

- Support writing to ZIP file.
- Support os.PathLike file names.

2020.5.28

- Fix int32 to hex color conversion.
- Fix coordinates of closing path.
- Fix reading TIFF files with no overlays.

2020.5.1

- Split positions from counters.

2020.2.12

- Initial release.
