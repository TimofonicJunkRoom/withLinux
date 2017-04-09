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

https://groups.google.com/forum/#!topic/torch7/b21m8xJ8TUc
```
Francisco Vitor Suzano Massa 	
16/3/31
The procedure is even simpler than before. You will need:

- write the C code as standalone functions (which take as arguments THTensors), no more Lua stuff in the C files. https://github.com/torch/nn/blob/master/lib/THNN/generic/Abs.c
- add the header declaration in THNN.lua https://github.com/torch/nn/blob/master/lib/THNN/generic/THNN.h#L5-L13
- include an entry in init.c to generate the float/double types https://github.com/torch/nn/blob/master/lib/THNN/init.c#L7-L8
- add the Lua definition of the class. the C functions are present in the THNN namespace. The difference now is that you need to pass the arguments to the C function in the lua call, and you need to add :cdata() for the tensors (the other basic types can go directly as is) https://github.com/torch/nn/blob/master/Abs.lua
```
