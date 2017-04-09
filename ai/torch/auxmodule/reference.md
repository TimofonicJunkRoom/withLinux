# http://torch.ch/docs/developer-docs.html

For a network module
```lua
[output]    forward(input)
[gradInput] backward(input, gradOutput)
```

please override these functions instead of `forward` and `backward`
```lua
[output]    updateOutput(input)
[gradInput] updateGradInput(input, gradOutput)
            accGradparameters(input, gradOutput) -- optional, if your module ships parameter
            reset() -- optional, how trainable parameters are reset, i.e. initialized before training.
```

empty holder for a new class
```lua
local NewClass, Parent = torch.class('nn.NewClass', 'nn.Module')

function NewClass:__init()
   Parent.__init(self)
end
function NewClass:updateOutput(input)
end
function NewClass:updateGradInput(input, gradOutput)
end
function NewClass:accGradParameters(input, gradOutput)
end
function NewClass:reset()
end
```
