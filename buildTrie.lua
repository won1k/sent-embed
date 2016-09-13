--[[
Build prefix trie from trained LM
]]--

require 'torch'
require 'nn'
require 'rnn'
require 'nngraph'
require 'hdf5'
tds = require 'tds'

cmd = torch.CmdLine()

cmd:option('-datafile', 'data/ptb.hdf5', 'file to load corpus data')
cmd:option('-loadfile', 'lm_epoch30.00_75.92.t7', 'file to load trained LM from')
cmd:option('-gpu', 0, '>=0 if GPU, -1 if CPU')

cmd:option('-num_words', 5, 'number of top words to consider')
cmd:option('-sent_len', 10, 'length of sentence')

EOS = 3
Module = nn.Module

function firstWord(data, num_words)
    local firstWords = data:read('2'):all():long()[{{},{1}}]
    for i = 2, lengths:size(1) do
    	firstWords = torch.cat(firstWords, data:read(tostring(lengths[i])):all():long()[{{},{1}}], 1)
    end
    firstWords = firstWords:squeeze()
    local nsent = firstWords:size(1)
    local counts = torch.zeros(nfeatures):long()
    -- Counts
    for i = 1, nsent do
        counts[firstWords[i]] = counts[firstWords[i]] + 1
    end
    -- Get top num_words with most counts
    local maxVal, maxIdx = counts:topk(num_words, true)
    return maxIdx
end

function nextWord(trie, num_words, currWord, prevState)
	if prevState then
		storeState(prevState)
	end

	local nextWords = model:forward(torch.Tensor{currWord})[1]
	local maxVal, maxIdx = nextWords:topk(num_words, true)

	if maxIdx[1] == EOS then
		return
	else
		local currState = getState()
		trie[currWord] = tds.Hash()
		for i = 1, num_words do
			model:forget()
			nextWord(trie[currWord], num_words, maxIdx[i], currState)
		end
	end
end

function Module:getLSTMLayers()
   if self.modules then
      for i, module in ipairs(self.modules) do
         if torch.type(module) == "nn.FastLSTM" then
            lstmLayers[k] = module
            k = k + 1
         else
            module:getLSTMLayers()
         end
      end
   end
end

function getState()
	local currState = {}
	local k = 1
	for i, module in ipairs(model.modules) do
		if module.modules then
			for j, submodule in ipairs(module.modules) do
				if torch.type(submodule) == "nn.FastLSTM" then
					if module.output ~= nil then
						print(module.output)
		         	currState[k] = {module.output:clone()}
		         	if module.cell ~= nil then
		         		if currState[k] then
		         			table.insert(currState[k], module.cell:clone())
		         		end
		         	end
		         	k = k + 1
		         end
		      end
		   end
		end
   end
   return currState
end

-- Assume currState = {1: {output, cell}, 2: {output, cell}, ...}
function storeState(currState)
   for i = 1, #lstmLayers do
   	lstmLayers[i].userPrevOutput = nn.rnn.recursiveCopy(lstmLayers[i].userPrevOutput, currState[i][1])
   	lstmLayers[i].userPrevCell = nn.rnn.recursiveCopy(lstmLayers[i].userPrevCell, currState[i][2])
   end
end

function main()
    -- parse input params
   opt = cmd:parse(arg)

   if opt.gpu >= 0 then
      print('using CUDA on GPU ' .. opt.gpu .. '...')
      require 'cutorch'
      require 'cunn'
      --cutorch.setDevice(opt.gpu + 1)
   end

   -- Load model
   model = torch.load(opt.loadfile)
   k = 1
   lstmLayers = {}
   model:getLSTMLayers()
   print('model loaded!')

   -- Load data
   local data = hdf5.open(opt.datafile, 'r')
   nfeatures = data:read('nfeatures'):all():long()[1]
   lengths = data:read('sent_lens'):all()

   -- Build trie
   local trie = tds.Hash()
   local firstWords = firstWord(data, opt.num_words)
   for i = 1, opt.num_words do
   	print(firstWords[i], i)
   	nextWord(trie, opt.num_words, firstWords[i])
   end

   -- Save trie
   print('saving model...')
   torch.save("trie", trie)
end

main()
