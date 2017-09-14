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

## CUDA Memory is not freed.

That's because the dataloader doesn't stop its workers if you kill the process
abruptly. Make sure all the related processes get killed.
See https://github.com/pytorch/pytorch/issues/1085
