# ReMatch
Official implementation of "Mind the Residual Gap: Probabilistic Downscaling under Real-World Bias"

Code coming soon! 

Upstream:
- Repository: NVIDIA/physicsnemo
- Path: examples/weather/corrdiff
- Commit: 769d2b0eb031e9fc919e8b7e7cbdcb6816694e2b
- License: Apache-2.0

# SwinIR baseline

We use a SwinIR-style deterministic super-resolution model as a mean predictor and deterministic baseline.

Upstream:
- Repository: JingyunLiang/SwinIR
- Paper: SwinIR, ICCV 2021
- License: <check upstream license>

Modifications used in this paper:
- Adapted input/output channels for multi-channel wind-field super-resolution.
- Set window size to 7 for 21x21 low-resolution inputs.
- Used PixelShuffle upsampling to map 21x21 inputs to 168x168 outputs.
