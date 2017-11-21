# RC-Nets
## Image Restoration Using Deep Regulated Convolutional Networks

### Abstract

While the depth of convolutional neural networks has attracted substantial attention in the deep learning research, the width of these networks has recently received greater interest. The width of networks, defined as the size of the receptive fields and the density of the channels, has demonstrated crucial importance in low-level vision tasks such as image denoising and restoration. However, the limited generalization ability, due to increased width of networks, creates a bottleneck in designing wider networks. In this paper we propose Deep Regulated Convolutional Network (RC-Net), a deep network composed of regulated sub-network blocks cascaded by skip-connections, to overcome this bottleneck. Specifically, the Regulated Convolution block (RC-block), featured by a combination of large and small convolution filters, balances the effectiveness of prominent feature extraction and the  generalization ability of the network. RC-Nets have several compelling advantages: they embrace diversified features through large-small filter combinations, alleviate the hazy boundary and blurred details in image denoising and super-resolution problems, and stabilize the learning process.  Our proposed RC-Nets outperform state-of-the-art approaches with large performance gains in various image restoration tasks while demonstrating promising generalization ability.


-----------------------------------------------------------------
### Main Contents:
-----------------------------------------------------------------
**RC_Train**: training demo for Gaussian denoising.

**demos**:  Demo_test_RC-.m.

**model**: including the trained models for Gaussian denoising and Super-resolution

**testsets**: BSD100,B200 and Set14 for Gaussian denoising evaluation and Super-resolution


-----------------------------------------------------------------

### Results:
-----------------------------------------------------------------

**First image from Set12-test with noise level=10**
![First image from Set12-test with noise level=10](http://i.imgur.com/4WkiKXI.png)

**Second image from BSD200-test with noise level=10**
![Second image from BSD200-test with noise level=10](http://imgur.com/kRH8oFx.png)

**I.Comparing 7x7 filter-size WINs with 13x13 filter-size WINs for noise level=30**
![1Comparing 7x7 filter-size WINs with 13x13 level=30](http://i.imgur.com/D7OjoKw.png)

As we can see, Increasing filter size can further improve performance.

**II.Comparing 7x7 filter-size WINs with 13x13 filter-size WINs for noise level=30**
![2Comparing 7x7 filter-size WINs with 13x13 level=30](http://i.imgur.com/p1qPVuI.png)

As we can see, Increasing filter size can further improve performance.

**III.Comparing 7x7 filter-size WINs with 13x13 filter-size WINs for noise level=30**
![3Comparing 7x7 filter-size WINs with 13x13 level=30](http://i.imgur.com/legwbim.png)
