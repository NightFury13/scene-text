require('cutorch')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('paths')
require('nngraph')
require('sys')
require('xlua')

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


function get_prediction()
    -- NOTE : CHANGE THE BELOW PATHS AS PER REQUIREMENTS --
    local imagePath = '/tmp/scene_rnd/demo.png'
    --local outfile = io.open('../data/answer.txt', 'w')
    local outfile = '/tmp/scene_rnd/answer.txt'
    -- Leave out the other stuff --

    local predictions = {}
    local img = loadAndResizeImage(imagePath)
    local text, raw = recognizeImageLexiconFree(model, img)

    -- Write outputs to file for comparison
    sys.execute('echo '.. text .. ' > ' .. outfile) 

    sys.execute('rm '..imagePath)
    print("Prediction :" .. text)
end

function do_nothing()
    print("No Image, Sleeping...")
end

print('Loading model...')
local modelDir = '../model/crnn_demo/'
paths.dofile(paths.concat(modelDir, 'config.lua'))
local modelLoadPath = paths.concat(modelDir, 'crnn_demo_model.t7')
gConfig = getConfig()
gConfig.modelDir = modelDir
gConfig.maxT = 0
model, criterion = createModel(gConfig)
snapshot = torch.load(modelLoadPath)
loadModelState(model, snapshot)
model:evaluate()
print(string.format('Model loaded from %s', modelLoadPath))

-- Infinite Loop Begins
while 1>0 do
    -- Load the image.
    xlua.trycatch(get_prediction, do_nothing)
    sys.sleep(1)
end
