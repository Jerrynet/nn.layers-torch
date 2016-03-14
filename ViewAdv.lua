-- ViewAdv
-- Advanced View() layer for same #dimensions transform
--
-- Usage
-- layer = nn.ViewAdv(dim1_size, dim2_size, ...)
-- note: argument number must be the same as input dimension number
--
-- dimX_size: 0 = same as input, -1 = Unknown
--
-- Example
-- a = torch.FloatTensor({{{1,2,3},{4,5,6}},{{1,2,3},{4,5,6}}}) -- size 2x2x3
-- layer = jnlayers.utils.ViewAdv(1, -1, 0)
-- outputs size = 1x4x3

local ViewAdv, parent = torch.class('nn.ViewAdv', 'nn.Module')

function ViewAdv:__init(...)
    parent.__init(self)
    self.size = {...}
    self.input_size = {}
    self.performSize = {}
end

function ViewAdv:updateOutput(input)
    local nEle = input:nElement()
    self.input_size  = input:size():totable()
    self.performSize = input:size():totable()
    assert(#self.input_size==#self.size)

    local isOneMinusone = 0
    local minusOneLoc = 0
    local currentEle = 1

    for i=1,#self.size do
        if self.size[i] == 0 then
            self.performSize[i] = self.input_size[i]
            currentEle = currentEle*self.input_size[i]
        elseif self.size[i] == -1 then
            isOneMinusone = isOneMinusone+1
            minusOneLoc = i
            if (isOneMinusone>1) then
                error('Should have only one -1 in nn.ViewAdv arguments.')
            end
        elseif self.size[i] > 0 then
            self.performSize[i] = self.size[i]
            currentEle = currentEle*self.performSize[i]
        end
    end

    self.performSize[minusOneLoc] = nEle/currentEle
    self.output = input:view(table.unpack(self.performSize))
    return self.output
end

function ViewAdv:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput:view(table.unpack(self.input_size))
    return self.gradInput
end