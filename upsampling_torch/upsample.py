import argparse
import os
# Must be set before importing torch.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from utils import Upsampler


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help='Path to input directory. See README.md for expected structure of the directory.')
    parser.add_argument("--output_dir", required=True, help='Path to non-existing output directory. This script will generate the directory.')
    parser.add_argument("--pretrained_model_path", default=None, help='Path to the pretrained model. If not provided, the default model will be used.')
    parser.add_argument("--max_bisections", type=int, default=5, help='Maximum number of bisections for adaptive upsampling. Default is 5.')

    # Parse the arguments
    args = parser.parse_args()
    return args


def main():
    flags = get_flags()

    upsampler = Upsampler(input_dir=flags.input_dir, output_dir=flags.output_dir, pretrained_model_path=flags.pretrained_model_path, max_bisections=flags.max_bisections)
    upsampler.upsample()


if __name__ == '__main__':
    main()
