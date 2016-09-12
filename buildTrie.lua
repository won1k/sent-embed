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

function firstWord(sentences, num_words)
    local firstWords = sentences[{{},{1}}]:squeeze()
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

function nextWord(trie, model, num_words, currWord, t)
	-- Termination
	if t == opt.sent_len then
		return
	end
	-- Else build next layer
	trie[currWord] = tds.Hash()
	-- Get top words from language model
	local topWords = model:forward(torch.Tensor{currWord})[1]
	local maxVal, maxIdx = topWords:topk(num_words, true)
	-- Recurse
	for i = 1, num_words do
		nextWord(trie, model, num_words, maxIdx[i], t + 1)
	end

	print(t)
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

   -- Load model/data
   model = torch.load(opt.loadfile)
   print('model loaded!')
   data = hdf5.open(opt.datafile, 'r')
   nfeatures = data:read('nfeatures'):all():long()[1]

   -- Build trie
   trie = tds.Hash()
   local firstWords = firstWord(data:read(tostring(opt.sent_len)):all():long(), opt.num_words)
   for i = 1, opt.num_words do
   	nextWord(trie, model, opt.num_words, firstWords[i], 1)
   end

   -- Save trie
   torch.save("trie", trie)
end

main()
