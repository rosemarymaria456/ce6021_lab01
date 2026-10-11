# Lab Report — Lab 01: Efficient Separable Gaussian Convolution

*Fill in each section below. Be specific — name the actual methods/parameters you used, since
your report is graded alongside your code.*

## Overview

_1 sentence: What was the goal of this lab._

The goal of this lab was to implement Gaussian blur convolution efficiently by exploiting the separability of the 2D Gaussian kernel.

## Implementation

_1-2 sentences: Briefly describe the steps you performed to implment efficient convolution._

I generated a 1D Gaussian kernel of size 1 and normalised it. Then I performed the convolution as two 1D passes where it was first horizontally, then vertically padding the image once with np.pad. This is so the output keeps the same shape as the input.

## Results

_1-2 sentences: How well did your method perform vs the baseline in the pytests? ._

My code ran several times faster than the naive 2D convolve2d baseline where it passed all the scaled time thresholds down to 0.1 baseline. The output matched the input shape and was correctly cast to uint8. I can see the image of the pyramid sculpter but not the graph. I struggled in trying to get it to work.
