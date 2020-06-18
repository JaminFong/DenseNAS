import argparse
import os


def get_list(data_path, output_path):
    for split in os.listdir(data_path):
        if split == 'train':
            split_path = os.path.join(data_path, split)
            if not os.path.isdir(split_path):
                continue
            f_train = open(os.path.join(output_path, split + '_datalist'), 'w')
            f_val = open(os.path.join(output_path, 'val' + '_datalist'), 'w')
            class_list = os.listdir(split_path)
            for sub in class_list[:100]:
                sub_path = os.path.join(split_path, sub)
                if not os.path.isdir(sub_path):
                    continue
                img_list = os.listdir(sub_path)
                train_len = int(0.8*len(img_list))
                for image in img_list[:train_len]:
                    image_name = os.path.join(sub, image)
                    f_train.writelines(image_name + '\n')
                for image in img_list[train_len:]:
                    image_name = os.path.join(sub, image)
                    f_val.writelines(image_name + '\n')
            f_train.close()
            f_val.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Params")
    parser.add_argument('--image_path', type=str, default='', help='the path of the images')
    parser.add_argument('--output_path', type=str, default='.', help='the output path of the list file')
    args = parser.parse_args()

    get_list(args.image_path, args.output_path)
