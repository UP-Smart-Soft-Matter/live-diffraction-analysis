# live-diffraction-efficiency

Live diffraction-efficiency measurement with a Thorlabs camera.

The program shows the camera image, finds the diffraction orders in every frame
and displays the share of the total power in each order, continuously. It is the
live version of the single-image scripts in `diffraction-efficiency-height-retrieval`
and uses the same method.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Thorlabs camera setup](#thorlabs-camera-setup)
- [Starting the program](#starting-the-program)
- [The window](#the-window)
- [Measuring correctly](#measuring-correctly)
- [Saved files](#saved-files)
- [Using the analysis in your own scripts](#using-the-analysis-in-your-own-scripts)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)

## Requirements

- Windows, Python 3.10 or newer (developed with 3.14)
- A Thorlabs scientific camera supported by ThorCam's TSI SDK: Zelux, Kiralux,
  Quantalux, CS-series and similar
- ThorCam installed (provides the USB driver and the SDK)

Older DCC/DCU cameras use a different SDK (uc480) and are **not** supported.

No camera is needed to try the program: it includes a simulated camera.

> The Thorlabs camera backend has so far only been tested against the simulator,
> not against real hardware.

## Installation

The project folder already contains a ready-to-use virtual environment (`.venv`).
To set it up from scratch on another PC:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Thorlabs camera setup

The Thorlabs Python package `thorlabs_tsi_sdk` is not on PyPI; it ships with
ThorCam. This has to be done once per PC.

1. Install **ThorCam** from thorlabs.com.
2. Unzip
   `C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support\Scientific_Camera_Interfaces.zip`
   to any folder.
3. Install the Python package from the unzipped folder:
   ```
   .venv\Scripts\python.exe -m pip install "<unzipped>\SDK\Python Toolkit\thorlabs_tsi_camera_python_sdk_package.zip"
   ```
4. The program looks for `thorlabs_tsi_camera_sdk.dll` in the ThorCam install
   folder. If it reports that the DLL was not found, do one of these:
   - copy all DLLs from `<unzipped>\SDK\Native Toolkit\dlls\Native_64_lib` into a
     folder named `dlls` next to `main.py`, or
   - set the environment variable `THORLABS_TSI_DLL_DIR` to that folder.

## Starting the program

```
.venv\Scripts\python.exe main.py
```

If a Thorlabs camera is connected, the program connects to the first one found.
Otherwise choose a camera in the **Camera** box and click **Connect**.

| Command | What it does |
| --- | --- |
| `main.py` | Normal start with a real camera |
| `main.py --simulate` | Start with the simulated camera (a phase grating whose efficiencies drift slowly) |
| `main.py --image path\to\image.tif` | Simulated camera that replays a saved image |

Close ThorCam before connecting: only one program can use the camera at a time.

## The window

**Camera image (centre).** Each detected order has a red box (the area that is
summed) and its order number. The dashed white rectangle is the *search region*:
orders are only looked for inside it. Drag it or pull its corner to exclude stray
reflections.

**Plots (right).**
- *Diffraction efficiency* — the current efficiency of each order in percent.
- *Profile along the orders* — the intensity along the line through all orders.
- *Efficiency over time* — the last 600 measurements of each order.

**Controls (left).**

| Control | Meaning |
| --- | --- |
| Exposure, Gain | Camera settings. Gain is greyed out if the camera has none. |
| Average frames | Averages the last N frames before the analysis. Reduces noise, slows the response. |
| Auto contrast | Scales the image colours to the current frame. Display only; does not affect the result. |
| Capture dark frame / Clear | Stores the current frame and subtracts it from all following frames. |
| ROI half-size | Each order is summed over a square with a side of twice this value. |
| Min. order spacing | Minimum distance between two orders. Must be smaller than the real spacing. |
| Peak threshold | Orders weaker than this fraction of the brightest order are ignored. |
| Peak smoothing | Blur used only for finding the orders. Increase it for noisy or speckled spots. |
| Max. orders | Keep only the N brightest orders. |
| Zero order | Which order is labelled 0: the middle one or the brightest one. |
| Subtract background (median) | Subtracts the median grey level of the analysed region. |
| Lock order positions | Freezes the current order positions (see below). |
| Reset search region | Sets the search region back to the full image. |

The table at the bottom left lists the summed intensity and efficiency of each order.

The status bar shows the camera, image size, frame rate and number of orders on
the left, and the brightest pixel value (or the saturation warning) on the right.

## Measuring correctly

The efficiency of an order is the sum of the pixel values in its box divided by
the sum over all boxes. Three things distort that number:

- **Saturation.** If a pixel in a box is at full scale, the status bar turns red.
  The efficiencies are wrong then; reduce the exposure until the warning disappears.
- **Background.** A camera reports a small non-zero value even in the dark, and this
  is counted as signal in every box, which makes weak orders look stronger. Remove
  it by blocking the beam and clicking **Capture dark frame**, or by ticking
  **Subtract background (median)**. Both are off at start.
- **Box size.** The box must contain the whole spot but must not overlap its
  neighbours. Adjust **ROI half-size** while watching the red boxes.

Only orders that are detected count towards the total. Orders outside the search
region, below the threshold, or too close to the image edge for a full box are
left out, so the percentages always add up to 100 % of what is detected.

**Lock order positions.** Normally the orders are searched again in every frame.
When an order fades below the threshold it disappears, and with *Zero order =
Middle peak* the numbering of the others shifts. Tick **Lock order positions**
while all orders are visible: the positions are then fixed, and every order keeps
being measured under the same number. Untick it after moving the beam or camera.

## Saved files

**Save snapshot** writes two files with the name you choose:

- `name.tif` — the current frame as 32-bit float TIFF (after averaging and dark subtraction)
- `name.csv` — one row per order: `order, x_px, y_px, intensity, efficiency`

**Record to CSV** writes one row per order for every analysed frame until you
click **Stop recording**: `timestamp, order, x_px, y_px, intensity, efficiency`

- `timestamp` is Unix time in seconds; rows of the same frame share the same value
- `x_px`, `y_px` is the position of the order in the full camera image
- `efficiency` is a fraction between 0 and 1

## Using the analysis in your own scripts

The analysis does not depend on the GUI or the camera:

```python
import numpy as np
from PIL import Image
from lde.analysis import AnalysisParams, analyze

img = np.array(Image.open("image.tif").convert("L"))   # 2-D grayscale array
result = analyze(img, AnalysisParams(roi_half=20, min_distance=50))

for order, eff in zip(result.orders, result.efficiencies):
    print(f"{order:+d}: {eff:.1%}")
```

`result` also contains `peaks` (row, column of each order), `intensities`,
`profile` and `line`.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| "The Thorlabs Python package 'thorlabs_tsi_sdk' is not installed" | Do step 3 of the camera setup. |
| "Could not start the Thorlabs camera SDK … dll was not found" | Do step 4 of the camera setup. |
| "No Thorlabs camera found" | Check the USB cable, close ThorCam, click **Refresh**. |
| No orders are detected | Lower **Peak threshold**, increase the exposure, check the search region. |
| Noise or reflections are detected as orders | Raise **Peak threshold**, shrink the search region, or set **Max. orders**. |
| Two boxes on one spot | Increase **Min. order spacing** or **Peak smoothing**. |
| Neighbouring orders are merged or missing | Decrease **Min. order spacing**. |
| Order numbers jump | Use **Lock order positions**. |

## Project layout

```
main.py            start script
lde/analysis.py    order detection and efficiency calculation
lde/cameras.py     Thorlabs camera backend and the simulated camera
lde/app.py         the window
requirements.txt   Python packages
```
