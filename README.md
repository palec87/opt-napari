# napari-opt-preprocessor

[![tests](https://github.com/palec87/opt-napari/actions/workflows/test-plugin.yml/badge.svg)](https://github.com/palec87/opt-napari/actions/workflows/test-plugin.yml)
[![codecov](https://codecov.io/github/palec87/opt-napari/graph/badge.svg?token=2RMBWECCDS)](https://codecov.io/github/palec87/opt-napari)

OPT preprocessing plugin written for napari

place for the gif
<img src="" width="700"/>

Jump to:
- [Usage](#usage)
  - [Starting point](#starting-point)
  - [Global settings](#settings)
  - [Corrections](#corrections)
  - [Other](#other)
- [Installation](#installation)
- [Troubleshooting installation](#troubleshooting-installation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## 🛀 Usage

### Starting point
1. Data streamed from ImSwitch OPT widget (see details here[LINK])
2. Loaded images as data stack
3. Other stack 3D volume data formats

### Global settings
Tracking

Inplace operations

### Corrections
Dark-field correction
Bright-field correction
Bad-pixel correction
Intensity correction

### Other
Binning
ROI
-Log

## 💻 Installation
We encourage to create a custom virtual environment for running ToMoDL with the following instructions:

`conda create --name opt python>=3.8`

or `python.exe -m venv opt`

Install required packages using pip inside `venv`:

`pip install -r requirements.txt`

## 🎄 Contributing

Contributions are very welcome. Tests can be run with [pytest], please ensure
the coverage at least stays the same before you submit a pull request.

## 🚓 License

Distributed under the terms of the [GNU-3] license,
"napari-clusters-plotter" is free and open source software

## 💳 Acknowledgements
This project was supported by the european horizon Europe project [IMAGINE](https://cordis.europa.eu/project/id/101094250).

## 🔨 Issues

If you encounter any problems, please [file an issue]() along
with a detailed description.

[GNU-3]: https://www.gnu.org/licenses/gpl-3.0.en.html
[pytest]: https://docs.pytest.org/en/7.0.x/
