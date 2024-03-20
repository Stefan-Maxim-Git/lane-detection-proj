# lane-detection-proj

<h2>Description</h2>
A basic lane detection program written in Python using the OpenCV library.

<h2> Requirements </h2>
This program was was written using the following: <br/>

<ul>
  <li> Python 3.6.8 </li>
  <li> OpenCV version 4.8.1 </li>
  <li> NumPy version 1.19.5 </li>
  <li> PyCharm Community Edition </li>
</ul>

<h2> Functionality </h2>

Based on a test video (which is found in the repo of this project), I applied a number of image processing filters and transformations in order to draw two lines on the initial video that follow the lane of the street. Although the algorithm is basic, it can be used
for implementing a more advanced and polished self-driving artificial intelligence software for automobiles. <br/> <br/>
In order to allow the algorithm to run as close to real-time as possible, I made use of the OpenCV library and applied several spatial and punctual image processing operations (such as changing the color space from RGB to Grayscale, blurring, changing the resolution etc.) to reduce the amount of data to be processed. After keeping only the essential data from the video, I split the video in two halves, each containing white points which represent the two lane dividers. Utilizing the NumPy library, I applied a linear regression for each half of the video in order to trace the approximate real-time location of the lane dividers and thus saving the coordinates for further use.

<h2>Operations used </h2>

<ul>
  <li> Reducing the resolution of the video</li>
  <li> Changing the color space from RGB to Grayscale </li>
  <li> Polygonal fitting </li>
  <li> Perspective transformation </li>
  <li> Blurring </li>
  <li> Edge detection </li>
  <li> Binarization </li>
  <li> Linear regression </li>
</ul>

<h2> Results </h2>

https://github.com/Stefan-Maxim-Git/lane-detection-proj/assets/164219916/2e0b0325-feb9-4244-84d6-2b7ff86451bc

