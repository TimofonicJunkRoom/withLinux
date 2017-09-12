Pytorch Tips
===

## Decaying learning rate

https://discuss.pytorch.org/t/adaptive-learning-rate/320

## TensorBoard?

`pip3 install visdom`

## CHW/HWC convertion without image rotation

```
Image -> Numpy : transpose((2,0,1))
HWC      CHW

Numpy -> Image : transpose((1,2,0))
CHW      HWC
```
