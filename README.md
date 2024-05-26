# Computational Photography course 

## Image Deblurring for Nonuniform Motion Blur Using Uniform Defocus Map Estimation

#### Chang, C.-F., Wu, J.-L., and Tsai, T.-Y. A single image deblurring algorithm for nonuniform motion blur using uniform defocus map estimation. 2017

Not finished implementation. 

#### Run

Install dependencies: 

```pip -m install requirements.txt```

Run program: 

```cd MotionBlur```

```python3.10 main [--input filepath] [--output filepath]```

## Color-to-gray conversion based on boundary points 

#### Zhang, L., and Wan, Y. Color-to-gray conversion based on boundary points. 2022 

#### Description

The implementation is executed in three main steps. 

The first step is to compute the boundary points of the colour image in the CIE Lab
colour space. The boundary points are computed using the Sobel operator. After applying
horizontal and vertical operators, the gradient image is obtained. The means and variances
of this image are computed, and by averaging the values, the new final gradient image is
obtained. In the paper, the threshold function wasn’t specified, so I applied thresholding based on
the percentile.

In the second step, the grayscale images are created. They are created by using linear
operations and ratios between red, green, and blue components. The step of change for the
components is set to 0.1, which means there are 66 combinations of the components in total.
The boundary points are computed from each grayscale image as well.

The final step consists of comparing boundary points from a colour image and boundary
points from grayscale images. The selected grayscale image is the one with which the colour
image has the same boundary points (most of them). And finally, the image is stretched to
fit the correct range

#### Run

Install dependencies: 

```pip -m install requirements.txt```

Run program: 

```cd ColorToGray```

```python3.10 main [--input filepath] [--output filepath]```

#### Results 

![](https://github.com/DianaDrzikova/ComputationalPhotography/blob/main/ColorToGray/results.png)
