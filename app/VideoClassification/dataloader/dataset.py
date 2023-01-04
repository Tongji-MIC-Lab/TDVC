import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

def get_database(data, subset, root_path, video_path_formatter):
    video_ids = []
    video_paths = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']

        if this_subset == subset:
            video_ids.append(key)
            annotations.append(value['annotations'])
            if 'video_path' in value:
                video_paths.append(Path(value['video_path']))
            else:
                label = value['annotations']['label']
                video_paths.append(video_path_formatter(root_path, label, key))

    return video_ids, video_paths, annotations


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


class ImageLoaderPIL(object):

    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with path.open('rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class VideoLoader(object):

    def __init__(self, subset, image_name_formatter, bpp_path, image_loader=None):
        self.image_name_formatter = image_name_formatter
        if image_loader is None:
            self.image_loader = ImageLoaderPIL()
        else:
            self.image_loader = image_loader
        self.bpp_path = bpp_path
        self.subset = subset

    def __call__(self, video_path, frame_indices):
        ref_video = []
        raw_video = []
        video_path_ = []
        bpp_ = []
        for i in frame_indices:
            bpp_file = self.bpp_path / Path(video_path.parts[-2]) / Path(video_path.parts[-1]) / Path('bpp.txt')
            with open(bpp_file, 'r', encoding='utf-8') as f:
                bpp = f.read().splitlines()

            ref_image_path = video_path / self.image_name_formatter(i + 1)
            raw_image_path = Path(str(ref_image_path).replace(ref_image_path.parts[-4], 'UCF101-yuv_img'))

            if ref_image_path.exists():
                ref_video.append(self.image_loader(ref_image_path))
                raw_video.append(self.image_loader(raw_image_path))
                if self.subset == 'validation':
                    try:
                        bpp_.append(float(bpp[i]))
                    except:
                        print(ref_image_path)
            else:
                print(ref_image_path)

        video = ref_video + raw_video

        return video, bpp_, str(ref_image_path)


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalRandomCrop(object):

    def __init__(self, size):
        self.size = size
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):
        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        # out = frame_indices[begin_index:end_index]

        if random.random() < 0.5:
            out = np.arange(begin_index, len(frame_indices), 1).tolist()[:self.size]
        else:
            out = np.arange(begin_index, len(frame_indices), 2).tolist()[:self.size]

        if len(out) < self.size:
            out = self.loop(out)

        return out


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, frame_indices):
        for i, t in enumerate(self.transforms):
            if isinstance(frame_indices[0], list):
                next_transforms = Compose(self.transforms[i:])
                dst_frame_indices = [
                    next_transforms(clip_frame_indices)
                    for clip_frame_indices in frame_indices
                ]

                return dst_frame_indices
            else:
                frame_indices = t(frame_indices)
        return frame_indices


class Dataset_UCF(data.Dataset):
    def __init__(self,
                 clip,
                 root_path,
                 subset,
                 trans,
                 video_path_formatter=(lambda root_path, label, video_id:
                 root_path / label / video_id),
                 target_type='label'):
        annotation_path = Path(str(root_path).replace(str(root_path.parts[-1]), 'UCF101-json/ucf101_01.json'))
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter)

        def image_name_formatter(x):
            return f'image_{x:05d}.png'

        self.loader = VideoLoader(subset, image_name_formatter, root_path)

        self.target_type = target_type

        self.trans = trans

        self.inter = clip

        self.clip = Compose([TemporalRandomCrop(self.inter)])

        self.subset = subset

    def __make_dataset(self, root_path, annotation_path, subset,
                       video_path_formatter):
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        video_ids, video_paths, annotations = get_database(
            data, subset, root_path, video_path_formatter)
        # video_ids = video_ids[:1500]

        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        n_videos = len(video_ids)
        print('n_videos', n_videos)
        # n_videos = 20
        dataset = []
        for i in range(n_videos):
            # if i % (n_videos // 5) == 0:
            #     print('dataset loading [{}/{}]'.format(i, len(video_ids)))

            if 'label' in annotations[i]:
                label = annotations[i]['label']
                label_id = class_to_idx[label]
            else:
                label_id = -1

            video_path = video_paths[i]
            video_path_ = video_path / Path('image_00001.png')
            if not video_path_.exists():
                continue

            segment = annotations[i]['segment']
            if segment[1] == 1:
                continue

            frame_indices = list(range(segment[0], segment[1]))
            sample = {
                'video': video_path,
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': label_id
            }
            dataset.append(sample)

        return dataset, idx_to_class

    def __loading(self, path, frame_indices):
        clip, bpp, path = self.loader(path, frame_indices)
        clip = [transforms.ToTensor()(img) for img in clip]
        clip = torch.stack(clip, 0).permute(0, 2, 3, 1)

        clip = self.trans(clip)

        return clip, bpp, path

    def get_data(self, ref_video, raw_video):
        ref_i = ref_video[:, 0, :, :]
        input_, ref_ = [], []
        for idx in range(raw_video.size(1) - 1):
            if idx == 0:
                refs = torch.stack([ref_i, ref_i, ref_i, ref_i])
            elif idx == 1:
                refs = torch.stack([ref_i, ref_i, ref_video[:, 1, :, :], ref_video[:, 1, :, :]])
            else:
                refs = torch.stack(
                    [ref_i, ref_video[:, idx - 2, :, :], ref_video[:, idx - 1, :, :], ref_video[:, idx, :, :]])
            ref_.append(refs)

        return raw_video[:, 1:, :, :].transpose(1, 0), torch.stack(ref_)

    def __getitem__(self, index):
        path = self.data[index]['video']
        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]
        frame_indicesa = self.data[index]['frame_indices']

        if self.subset == 'validation':
            frame_indices = np.arange(0, len(frame_indicesa), 1).tolist()[:self.inter]
        else:
            frame_indices = self.clip(frame_indicesa)

        clip, bpp, path = self.__loading(path, frame_indices)
        ref, raw = clip.split([self.inter, self.inter], dim=1)
        inputs, refs = self.get_data(ref, raw)

        bpp = torch.from_numpy(np.asarray(bpp))
        return clip, target, bpp, path, inputs, refs

    def __len__(self):
        return len(self.data)

