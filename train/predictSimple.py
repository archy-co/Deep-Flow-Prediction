import os, sys
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import TurbDataset
from DfpNet import TurbNetG
import utils as utils

modelFn = "./modelG"
outputDir = "./prediction"
expo = 5

def main(path: str):
    dataset = TurbDataset(None, mode=TurbDataset.TEST, dataDir=path, dataDirTest=path)
    testLoader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    if os.path.exists(outputDir):
        for f in os.listdir(outputDir):
            os.remove(os.path.join(outputDir, f))
    else:
        os.makedirs(outputDir)

    inputs = torch.FloatTensor(1, 3, 128, 128)
    inputs = Variable(inputs)
    inputs = inputs.cuda()

    netG = TurbNetG(channelExponent=expo)
    netG.load_state_dict(torch.load(modelFn))
    netG.cuda()
    netG.eval()

    for i, data in enumerate(testLoader, 0):
        inputs_cpu, targets_cpu = data
        inputs_cpu = inputs_cpu.float().cuda()
        targets_cpu = targets_cpu.float().cuda()
        inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)

        outputs = netG(inputs)
        outputs_cpu = outputs.data.cpu().numpy()[0]
        targets_cpu = targets_cpu.cpu().numpy()[0]
        inputs_cpu = inputs_cpu.cpu().numpy()[0]

        utils.imageOut(f"{os.path.join(outputDir, str(i))}", outputs_cpu, targets_cpu, normalize=False, saveMontage=True)
        utils.saveAsImage(f"{os.path.join(outputDir, str(i))}_input.png", inputs_cpu[0])
        utils.saveAsImage(f"{os.path.join(outputDir, str(i))}_target.png", targets_cpu[0])
        utils.saveAsImage(f"{os.path.join(outputDir, str(i))}_output.png", outputs_cpu[0])

if __name__ == "__main__":
    # get command line arguments
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path/to/NPZs>")
        sys.exit(1)
    else:
        main(sys.argv[1])