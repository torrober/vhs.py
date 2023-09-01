## vhs.py

Analog videotape emulator, written in Python
## Example
![](ezgif-2-3451c58aac.gif)
## How it works?

Here's a simplified explanation of how the code works:

1. The chrominance and luminance signals are separated
2. Chrominance and luminance channels are compressed and processed separately.
3. The images are re-assembled

# Usage
## UI
![](https://i.imgur.com/VK4E4A8.png)
Change the parameters to your liking, then hit record to tape, select the file you want to record, and wait until the process ends.
NOTE: the output video doesn't have sound, so it needs to be added after the effect is applied.
## Code
```python
from vhs import VHS
...
vhs = new VHS(lumaCompressionRate, lumaNoiseSigma, lumaNoiseMean, chromaCompressionRate, chromaNoiseIntensity, verticalBlur,horizontalBlur, borderSize)
#optional
vhs.generation = 3
```
## Installation
1. clone this repo
2. run pip install requirements.txt
3. run python index.py to open UI interface
