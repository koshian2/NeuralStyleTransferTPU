# NeuralStyleTransferTPU
Fast neural style trasnfer using Colab TPUs by TensorFlow/Keras.

You can output an image of 256x256 resolution in 3000 epoch in about 15 minutes.

# Result
## Content Image
![](https://github.com/koshian2/NeuralStyleTransferTPU/blob/master/images/kiyomizu.jpg)

## Style Image
![](https://github.com/koshian2/NeuralStyleTransferTPU/blob/master/images/nue.jpg)

## Generated (Total Variation Loss = 0.25)
![](https://github.com/koshian2/NeuralStyleTransferTPU/blob/master/images/kiyomizu_tv_0.25.png)

## Generated (Total Variation Loss = 0.01)
![](https://github.com/koshian2/NeuralStyleTransferTPU/blob/master/images/kiyomizu_tv_0.01.png)

Other parameters are both content loss = 0.025, style loss = 1.

# Refernce
Keras neural style transfer example (GPUs version)  
[https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py](https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py)
