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
