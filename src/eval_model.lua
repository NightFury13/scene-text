require('cutorch')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('paths')
require('nngraph')

require('libcrnn')
require('utilities')
require('inference')
require('CtcCriterion')
require('DatasetLmdb')
require('LstmLayer')
require('BiRnnJoin')
require('SharedParallelTable')


cutorch.setDevice(1)
torch.setnumthreads(4)
torch.setdefaulttensortype('torch.FloatTensor')

print('Loading model...')
--local modelDir = '../model/crnn_eng_rendered/'
--local modelDir = '../model/crnn_demo/'
local modelDir = '../model/crnn_blurOx_iiitVal/'
paths.dofile(paths.concat(modelDir, 'config.lua'))
local modelLoadPath = paths.concat(modelDir, 'snapshot_900000.t7')
--local modelLoadPath = paths.concat(modelDir, 'crnn_demo_model.t7')
gConfig = getConfig()
gConfig.modelDir = modelDir
gConfig.maxT = 0
local model, criterion = createModel(gConfig)
local snapshot = torch.load(modelLoadPath)
loadModelState(model, snapshot)
model:evaluate()
print(string.format('Model loaded from %s', modelLoadPath))

-- Load the labels.
local lablist = '../tool/svt_manual/svt_manual_lablist.txt'
local labels = {}
for label in io.lines(lablist) do
    labels[#labels+1] = label
end

-- Load the images.
local imlist = '../tool/svt_manual/svt_manual_imlist.txt'
local predictions = {}
for imgPath in io.lines(imlist) do
    local imagePath = imgPath
    local img = loadAndResizeImage(imagePath)
    local text, raw = recognizeImageLexiconFree(model, img)
    predictions[#predictions+1] = text
    print(string.format('Recognized text: (raw: %s) %s : Target : %s', raw, text, labels[#predictions]))
end

-- Write outputs to file for comparison
outfile = io.open('../tool/blurred/evaluation/blurOx_predictions.txt', 'w')
io.output(outfile)
for i=1,#predictions do
    io.write(labels[i] .. ' ' .. predictions[i] .. '\n')
end
io.close(outfile)
