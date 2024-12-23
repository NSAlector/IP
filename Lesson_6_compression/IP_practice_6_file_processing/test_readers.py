import sys
import argparse

from utils.reader import image_reader as imread
from utils.reader import csv_reader, bin_reader, txt_reader, json_reader
from utils.processor import histogram
from utils.writer import csv_writer, bin_writer, txt_writer, image_writer, json_writer

from utils.image_toner import stat_correction


def print_args_1():
    print(type(sys.argv))
    if (len(sys.argv) > 1):
        for param in sys.argv[1:]:
            print(param, type(param))
    return sys.argv[1:]

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument ('-img','--img_path', default ='', help='Path to image')

    parser.add_argument ('-p','--path', default ='', help='Input file path ')
    parser.add_argument('-t','--type', default='txt', help='Input file format ')

    parser.add_argument('-o', '--output', help='Save file path')

    return parser

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args(sys.argv[1:])

    image = None
    hist = None

    image = imread.read_data(args.img_path)
    hist = histogram.image_processing(image)

    hist_template = None

    match args.type:
        case 'img':
            img2 = imread.read_data(args.path)
            hist_template = histogram.image_processing(img2)
        case 'csv':
            hist_template = csv_reader.read_data(args.path)
        case 'bin':
            hist_template = bin_reader.read_data(args.path)
        case 'txt':
            hist_template = txt_reader.read_data(args.path)
        case 'json':
            hist_template = json_reader.read_data(args.path)
        case _:
            pass

    res_image = stat_correction.processing(hist_template, image)
    image_writer.write_data(args.output, res_image)