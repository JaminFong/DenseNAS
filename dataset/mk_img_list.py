import argparse
import os

def get_list(data_path, output_path):
    for split in os.listdir(data_path):
        split_path = os.path.join(data_path, split)
        if not os.path.isdir(split_path):
            continue
        f = open(os.path.join(output_path, split + '_datalist'), 'a+')
        for sub in os.listdir(split_path):
            sub_path = os.path.join(split_path, sub)
            if not os.path.isdir(sub_path):
                continue
            for image in os.listdir(sub_path):
                image_name = sub + '/' + image
                f.writelines(image_name + '\n')
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Params")
    parser.add_argument('--image_path', type=str, default='', help='the path of the images')
    parser.add_argument('--output_path', type=str, default='', help='the output path of the lmdb file')
    args = parser.parse_args()

    get_list(args.image_path, args.output_path)
