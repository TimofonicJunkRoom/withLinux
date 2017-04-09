local Dropout, Parent = torch.class('nn.MyDropout', 'nn.Module')

function Dropout:__init(p)
	Parent.__init(self)
	self.p = p or 0.5
	if self.p >= 1 or self.p < 0 then
		error('<Dropout> illegal percentage, must be 0 <= p < 1')
	end
	self.noise = torch.Tensor()
	self.havenoise = false
end

--function Dropout:updateOutput(input)
--	self.output:resizeAs(input):copy(input)
--	self.noise:resizeAs(input)
--	self.noise:bernoulli(1-self.p)
--	self.output:cmul(self.noise)
--	return self.output
--end

-- above is original implementation, One slight issue with the 
-- Jacobian class is the fact that it assumes that the
-- outputs of a module are deterministic wrt to the inputs.
-- This is not the case for that particular module, so for
-- the purpose of these tests we need to freeze the noise
-- generation, i.e. do it only once:

function Dropout:updateOutput(input)
	self.output:resizeAs(input):copy(input)
	if not self.havenoise then
	--self.noise = self.noise or input.new():resizeAs(input):bernoulli(1-self.p)
	   self.noise = input.new():resizeAs(input):bernoulli(1-self.p)
	   self.havenoise = true
	end
	self.output:cmul(self.noise)
	return self.output
end

function Dropout:updateGradInput(input, gradOutput)
	self.gradInput:resizeAs(gradOutput):copy(gradOutput)
	self.gradInput:cmul(self.noise) -- simply mask the gradients with the noise vector
	return self.gradInput
end
