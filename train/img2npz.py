import os, sys
import utils as utils
import PIL.Image as Image
import numpy as np

outputDir = "./NPZs"

def main(path: str):
    if os.path.exists(outputDir):
        for f in os.listdir(outputDir):
            os.remove(os.path.join(outputDir, f))
    else:
        os.makedirs(outputDir)

    for f in os.listdir(path):
        if f.endswith(".png"):
            imageO = Image.open(os.path.join(path, f))
            imageO.load()
            image = np.asarray(imageO, dtype="float32")
            # resize so that shape is 3, 128, 128
            image = np.transpose(image, (2, 0, 1))

            print(image.shape)
            np.savez_compressed(os.path.join(outputDir, "".join(f.split(".")[:-1])), a=image)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path/to/images>")
        sys.exit(1)
    else:
        main(sys.argv[1])