-- RemoveLastSingletonDim
-- if the last dimension is 1, remove it

local RemoveLastSingletonDim, Parent = torch.class('jnlayers.utils.RemoveLastSingletonDim', 'nn.Module')

function RemoveLastSingletonDim:__init()
    Parent.__init(self)
    self.size = -1
    self.didRemove = false
end

function RemoveLastSingletonDim:updateOutput(input)
    self.size = input:size():totable()
    if self.size[#self.size] == 1 then
        self.size[#self.size] = nil
        self.didRemove = true
    end
    self.output = input:view(table.unpack(self.size))
    return self.output
end

function RemoveLastSingletonDim:updateGradInput(input, gradOutput)
    if self.didRemove then
        self.size[#self.size+1] = 1
    end
    self.gradInput = gradOutput:view(table.unpack(self.size))
    self.didRemove = false
    return self.gradInput
end