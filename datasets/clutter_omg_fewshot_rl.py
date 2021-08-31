"""
mimic clutter_mnist dataset, but in few shot setting
https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/5e18a5e5b369903092f683d434efb12c7c40a83c/src/omniglot_dataset.py
Created on 5/2/2021 10:25 PM

@author: Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
"""
from __future__ import print_function
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision.ops import box_iou
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from datasets.batch_sampler import PrototypicalBatchSampler
from PIL import Image
import numpy as np
import shutil, random
import errno
import torch
import os

DEBUG = False
IMG_CACHE = {}
class OmniglotDataset(data.Dataset):
    vinalys_baseurl = 'https://raw.githubusercontent.com/jakesnell/' \
                      'prototypical-networks/master/data/omniglot/splits/vinyals/'
    vinyals_split_sizes = {
        'test': vinalys_baseurl + 'test.txt',
        'train': vinalys_baseurl + 'train.txt',
        'trainval': vinalys_baseurl + 'trainval.txt',
        'val': vinalys_baseurl + 'val.txt',
    }
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    splits_folder = os.path.join('splits', 'Tingfeng5_small3')
    raw_folder = 'raw'
    processed_folder = 'data'

    def __init__(self, root, mode='train', bg_name='clutter',transform=None,
                 target_transform=None, download=True):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        - download: need to download the dataset
        '''
        super(OmniglotDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it')
        self.classes = get_current_classes(os.path.join(
            self.root, self.splits_folder, mode + '.txt'))
        self.all_items = find_items(os.path.join(
            self.root, self.processed_folder), self.classes)

        #alphabet = list(set([a.split('/')[0] for a in self.classes]))
        # print('self.classes ', len(self.classes), self.classes[0], alphabet[0])
        # print(len(alphabet))
        # myclasses = [a for a in self.classes if a.split('/')[0] in alphabet[:5]]
        # print('my classe ', len(myclasses))
        # with open(os.path.join(self.root, 'splits', 'Tingfeng5', 'train.txt'), 'w') as f:
        #     for cls in myclasses:
        #         f.write('%s\n' % cls)
        # exit()

        self.idx_classes = index_classes(self.all_items) # class to index

        self.paths, self.y = zip(*[self.get_path_label(pl)
                              for pl in range(len(self))])


        self.x = map(load_img, self.paths, range(len(self.paths)))
        self.x = list(self.x)

        if bg_name == 'clutter':
            print('generating cluttered omniglot...')
            self.new_x, y_box = self._generate_clutter_img()
        elif bg_name == 'patch':
            print('generating random patched omniglot...')
            self.new_x, y_box = self._generate_patch_img()
        else:
            print('wrong bg name')
            exit()
        self.y = torch.cat((y_box, torch.from_numpy(np.stack(self.y)).unsqueeze(1)), dim=1)

    def _generate_clutter_img(self,):
        '''
        generate translated images, put 28*28 character at random location,
        in a 84*84 black canvas with clutter, target style [x1,y1,x2,y2]
        '''
        new_x = np.zeros((len(self.x), 84, 84), dtype=float)  # (w,h)
        y_box = torch.zeros((len(self.x), 4), dtype=torch.float32)
        patch_size = 4
        for i, data in enumerate(self.x):
            # sample a location
            x_d = random.randint(0, 84 - 28)
            y_d = random.randint(0, 84 - 28)
            #new_x[i, x_d:x_d + 28, y_d:y_d + 28] = data
            new_x[i, y_d:y_d + 28, x_d:x_d + 28] = data
            y_box[i] = torch.tensor([x_d, y_d, x_d+28-1, y_d+28-1])
            # add clutter
            for _ in range(32):
                # crop from noise data
                noise_data = self.x[random.randint(0, len(self.x) - 1)]
                x0 = random.randint(0, 28 - patch_size)
                y0 = random.randint(0, 28 - patch_size)
                cropped = noise_data[y0:y0 + patch_size, x0:x0 + patch_size]
                # sample a location to put cropped noise data
                x = random.randint(0, 84 - patch_size)
                y = random.randint(0, 84 - patch_size)
                while np.sum(new_x[i, y:y + patch_size, x:x + patch_size]) != 0:
                    x = random.randint(0, 84 - patch_size)
                    y = random.randint(0, 84 - patch_size)
                # Insert digit fragment, but not on top of digits
                if np.sum(new_x[i, y:y + patch_size, x:x + patch_size]) == 0:
                    new_x[i, y:y + patch_size, x:x + patch_size] = cropped

                # Clip any over-saturated pixels
                new_x[i] = np.clip(new_x[i], 0., 1.)
        return new_x, y_box

    def _generate_patch_img(self,):
        '''
        generate translated images, 28*28 characters at random location,
        in a 84*84 black canvas with clutter, target style [x1,y1,x2,y2]
        '''
        new_x = np.zeros((len(self.x), 84, 84), dtype=float)  # (w,h)
        y_box = torch.zeros((len(self.x), 4), dtype=torch.float32)

        for i, data in enumerate(self.x):
            # sample a location for foreground character
            x_d = random.randint(0, 84 - 28)
            y_d = random.randint(0, 84 - 28)
            new_x[i, y_d:y_d + 28, x_d:x_d + 28] = data
            y_box[i] = torch.tensor([x_d, y_d, x_d+28-1, y_d+28-1])

            # add random patch
            for _ in range(4):
                '''generate random pixel patches'''
                # sample a location
                width = np.random.randint(low=int(28 * 0.7), high=int(28 * 1.3))
                x = random.randint(0, 84 - width)
                y = random.randint(0, 84 - width)
                while box_iou(torch.tensor([[x, y, x + width - 1, y + width - 1]]),
                              torch.tensor([[x_d, y_d, x_d + 28 - 1, y_d + 28 - 1]])) > 0:
                    width = np.random.randint(low=int(28 * 0.7), high=int(28 * 1.3))
                    x = random.randint(0, 84 - width)
                    y = random.randint(0, 84 - width)
                gray = np.ones((width, width), dtype=np.uint8) * \
                       np.random.uniform(0.020, 0.863)
                new_x[i, y:y + width, x:x + width] = gray

                # Clip any over-saturated pixels
                new_x[i] = np.clip(new_x[i], 0., 1.)

        return new_x, y_box

    def __getitem__(self, idx):
        x, target = self.new_x[idx], self.y[idx]
        img = Image.fromarray(x)
        if self.transform:
            x = self.transform(x)
        if DEBUG:
            print('target ', target)
            f, ax = plt.subplots()
            x1, y1, x2, y2, _ = target
            ax.imshow(self.x[idx], cmap='gray', interpolation='none')

            f, ax = plt.subplots()
            x1, y1, x2, y2, _ = target
            ax.imshow(np.asarray(img), cmap='gray', interpolation='none')
            patch = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                              edgecolor='g', facecolor='none', fill=False)
            ax.add_patch(patch)

        img = self.totensor(img)

        return img, target

    def __len__(self):
        return len(self.all_items)

    def get_path_label(self, index):
        filename = self.all_items[index][0]
        rot = self.all_items[index][-1]
        img = str.join(os.sep, [self.all_items[index][2], filename]) + rot
        target = self.idx_classes[self.all_items[index]
                                  [1] + self.all_items[index][-1]]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder))

    def download(self):
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.splits_folder))
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for k, url in self.vinyals_split_sizes.items():
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition(os.sep)[-1]
            file_path = os.path.join(self.root, self.splits_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition(os.sep)[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            orig_root = os.path.join(self.root, self.raw_folder)
            print("== Unzip from " + file_path + " to " + orig_root)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(orig_root)
            zip_ref.close()
        file_processed = os.path.join(self.root, self.processed_folder)
        for p in ['images_background', 'images_evaluation']:
            for f in os.listdir(os.path.join(orig_root, p)):
                shutil.move(os.path.join(orig_root, p, f), file_processed)
            os.rmdir(os.path.join(orig_root, p))
        print("Download finished.")

def find_items(root_dir, classes):
    retour = []
    rots = [os.sep + 'rot000', os.sep + 'rot090', os.sep + 'rot180', os.sep + 'rot270']
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            r = root.split(os.sep)
            lr = len(r)
            label = r[lr - 2] + os.sep + r[lr - 1]
            for rot in rots:
                if label + rot in classes and (f.endswith("png")):
                    retour.extend([(f, label, root, rot)])
    print("== Dataset: Found %d items " % len(retour))
    return retour

def index_classes(items):
    idx = {}
    for i in items:
        if (not i[1] + i[-1] in idx):
            idx[i[1] + i[-1]] = len(idx)
    print("== Dataset: Found %d classes" % len(idx))
    return idx

def get_current_classes(fname):
    with open(fname) as f:
        classes = f.read().replace('/', os.sep).splitlines()
    return classes

def load_img(path, idx):
    path, rot = path.split(os.sep + 'rot')
    if path in IMG_CACHE:
        x = IMG_CACHE[path]
    else:
        x = Image.open(path)
        IMG_CACHE[path] = x
    x = x.rotate(float(rot))
    x = x.resize((28, 28))

    shape = 1, x.size[0], x.size[1]
    x = np.array(x, np.float32, copy=False)
    x = 1.0 - x # character to white
    # x = x.transpose(0, 1).contiguous().view(shape)

    return x

def init_sampler(classes_per_set, samples_per_class, labels, iterations, mode):
    if 'train' in mode:
        classes_per_it = classes_per_set
        num_samples = samples_per_class
    else:
        classes_per_it = classes_per_set
        num_samples = samples_per_class

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=iterations)

if __name__ == "__main__":
    from datasets.batch_sampler import PrototypicalBatchSampler

    trainset = OmniglotDataset(mode='val',
                              root='/research/cbim/vast/tl601/Dataset/Omniglot')
    samples_per_class = 20
    classes_per_set = 5
    sampler = init_sampler(classes_per_set, samples_per_class*2,
                           trainset.y[:,-1], 10, 'train')
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_sampler=sampler)
    #batch_sampler=sampler)
    for i, (data, target) in enumerate(train_loader):
        print('data ', data.shape)
        print('target ', target.shape)
        print('target label ', target)
        support_y = target.view(-1, 2, samples_per_class, 5)[:,0,:,:]
        query_y = target.view(-1, 2, samples_per_class, 5)[:, 1, :, :]
        print('support_y ', support_y) # (classes_per_set, 5, 5)
        print('query y ', query_y)
        break
