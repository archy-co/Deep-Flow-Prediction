import os, sys
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import TurbDataset
from DfpNet import TurbNetG
import utils as utils

outputDir = "./AF_images"
expo = 5

def main(path: str):
    dataset = TurbDataset(None, mode=TurbDataset.TEST, dataDir=path, dataDirTest=path)
    testLoader = DataLoader(dataset, batch_size=1, shuffle=False)

    if os.path.exists(outputDir):
        for f in os.listdir(outputDir):
            os.remove(os.path.join(outputDir, f))
    else:
        os.makedirs(outputDir)

    for i, data in enumerate(testLoader, 0):
        inputs_cpu, targets_cpu = data
        inputs_cpu = inputs_cpu.float().cuda()
        targets_cpu = targets_cpu.float().cuda()

        targets_cpu = targets_cpu.cpu().numpy()[0]
        inputs_cpu = inputs_cpu.cpu().numpy()[0]

        utils.saveAsImage(f"{os.path.join(outputDir, str(i))}_input.png", inputs_cpu[0])
        utils.saveAsImage(f"{os.path.join(outputDir, str(i))}_target.png", targets_cpu[0])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path/to/NPZs>")
        sys.exit(1)
    else:
        main(sys.argv[1])