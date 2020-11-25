# pyscfb
Python/Cython/C implementation of Karmaresan, Peddinti and Cariani's Synchrony Capture Filter Bank cochlear model.

Also included are tools for adaptively matching harmonic templates to a spectral representation of a signal, as illustrated below.

![Template Animation](/images/template-animation1.gif)

The basic approach to adaptive template matching is simple.  The salience of a match between template and signal is measured by taking the innerproduct of both of these.  The F0 of the template is then adjusted up or down by calculating the gradient of the salience with respect to changes of F0.  This is just the inner product of (-1 times) the derivative of the template and the signal spectrum.  In the animation, the signal spectrum is shown in grey, the template in red, and the derivative of the template (times -1) in orange.  The basic mathematical idea was drawn from William Sethares' work on adaptive oscillators and [rhythm extraction](https://www.springer.com/gp/book/9781846286391), and used in a [different (but still pitch-related) context](https://github.com/analogouscircuit/peakpitchshift) in Dahlbom and Braasch (2020).

Some of this material was presented at the 2018 meeting of the Acoustical Society of America in Minneapolis.

## References

Arora, R. and Sethares, W. A. (2007). "Adaptive Wavetable Oscillators," IEEE Transactions on Signal Processing 55, 4382-4392.

Dahlbom, D. A. and Braasch, J. (2018). "An oscillatory template pitch model," Journal of the Acoustical Society of America 143. \[Abstract\]

Dahlbom, D. A. and Braasch, J. (2020). "How to pick a peak: Pitch and peak shifting in temporal models of pitch perception," Journal of the Acoustical Society of America 147, 2713-2727.

Kumaresan, R., Peddinti, V. K., and Cariani, P. (2013). "Synchrony capture filterbank: Auditory-inspired signal processing for tracking individual frequency components in speech," Journal of the Acoustical Society of America 133, 4290-4310.
