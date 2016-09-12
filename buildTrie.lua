--[[
Build prefix trie from trained LM
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'hdf5'

cmd = torch.CmdLine()

cmd:option('-loadfile', 'lm_epoch30.00_75.92.t7', 'file to load trained LM from')
cmd:option('-dictfile', 'data/ptb.dict.hdf5', 'file to load English dict from')
cmd:option('-gpu', 0, '>=0 if GPU, -1 if CPU')

function main()
    -- parse input params
   opt = cmd:parse(arg)

   if opt.gpu >= 0 then
      print('using CUDA on GPU ' .. opt.gpu .. '...')
      require 'cutorch'
      require 'cunn'
      --cutorch.setDevice(opt.gpu + 1)
   end

   -- Load model/dict
   model = torch.load(opt.loadfile)
   dict = hdf5.open()
end

main()
