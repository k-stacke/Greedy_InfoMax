import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

from GreedyInfoMax.vision.data.externalinputgenerator import ExternalInputIterator
from GreedyInfoMax.vision.data.externalsourcepipeline import ExternalSourcePipeline

from nvidia.dali.plugin.pytorch import DALIGenericIterator

def get_dataloader(opt):
    if opt.dataset == "stl10":
        train_loader, train_dataset, supervised_loader, supervised_dataset, test_loader, test_dataset = get_stl10_dataloader(
            opt
        )
    elif opt.dataset == 'cam17':
        train_loader, train_dataset, supervised_loader, supervised_dataset, test_loader, test_dataset = get_camelyon_dataloader(
            opt
        )
    else:
        raise Exception("Invalid option")

    return (
        train_loader,
        train_dataset,
        supervised_loader,
        supervised_dataset,
        test_loader,
        test_dataset,
    )


def get_stl10_dataloader(opt):
    base_folder = os.path.join(opt.data_input_dir, "stl10_binary")

    aug = {
        "stl10": {
            "randcrop": 64,
            "flip": True,
            "grayscale": opt.grayscale,
            "mean": [0.4313, 0.4156, 0.3663],  # values for train+unsupervised combined
            "std": [0.2683, 0.2610, 0.2687],
            "bw_mean": [0.4120],  # values for train+unsupervised combined
            "bw_std": [0.2570],
        }  # values for labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
    }
    transform_train = transforms.Compose(
        [get_transforms(eval=False, aug=aug["stl10"])]
    )
    transform_valid = transforms.Compose(
        [get_transforms(eval=True, aug=aug["stl10"])]
    )

    unsupervised_dataset = torchvision.datasets.STL10(
        base_folder,
        split="unlabeled",
        transform=transform_train,
        download=opt.download_dataset,
    ) #set download to True to get the dataset

    train_dataset = torchvision.datasets.STL10(
        base_folder, split="train", transform=transform_train, download=opt.download_dataset
    )

    test_dataset = torchvision.datasets.STL10(
        base_folder, split="test", transform=transform_valid, download=opt.download_dataset
    )

    # default dataset loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size_multiGPU, shuffle=True, num_workers=16
    )

    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=True,
        num_workers=16,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size_multiGPU, shuffle=False, num_workers=16
    )

    # create train/val split
    if opt.validate:
        print("Use train / val split")

        if opt.training_dataset == "train":
            dataset_size = len(train_dataset)
            train_sampler, valid_sampler = create_validation_sampler(dataset_size)

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=opt.batch_size_multiGPU,
                sampler=train_sampler,
                num_workers=16,
            )

        elif opt.training_dataset == "unlabeled":
            dataset_size = len(unsupervised_dataset)
            train_sampler, valid_sampler = create_validation_sampler(dataset_size)

            unsupervised_loader = torch.utils.data.DataLoader(
                unsupervised_dataset,
                batch_size=opt.batch_size_multiGPU,
                sampler=train_sampler,
                num_workers=16,
            )

        else:
            raise Exception("Invalid option")

        # overwrite test_dataset and _loader with validation set
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size_multiGPU,
            sampler=valid_sampler,
            num_workers=16,
        )

    else:
        print("Use (train+val) / test split")

    return (
        unsupervised_loader,
        unsupervised_dataset,
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )


def create_validation_sampler(dataset_size):
    # Creating data indices for training and validation splits:
    validation_split = 0.2
    shuffle_dataset = True

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler


def get_transforms(eval=False, aug=None):
    trans = []

    if aug["randcrop"] and not eval:
        trans.append(transforms.RandomCrop(aug["randcrop"]))

    if aug["randcrop"] and eval:
        trans.append(transforms.CenterCrop(aug["randcrop"]))

    if aug["flip"] and not eval:
        trans.append(transforms.RandomHorizontalFlip())

    if aug["hue"] and not eval:
        trans.append(transforms.ColorJitter(hue=0.1))

    if aug["grayscale"]:
        trans.append(transforms.Grayscale())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))
    elif aug["mean"]:
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["mean"], std=aug["std"]))
    else:
        trans.append(transforms.ToTensor())

    trans = transforms.Compose(trans)
    return trans

def get_weighted_sampler(dataset, num_samples):
    df = dataset.dataframe
    # Get number of sampler per label. Weight = 1/num sampels
    class_weights = { row.label_new: 1/row[0] for _, row in df.groupby(['label_new']).size().reset_index().iterrows()}
    print(class_weights)
    # Set weights per sample in dataset
    weights = [class_weights[row.label_new] for _, row in df.iterrows()]
    return WeightedRandomSampler(weights=weights, num_samples=num_samples)

def get_lnco_weighted_sampler(dataset, num_samples):
    df = dataset.dataframe
    # Get number of sampler per label. Weight = 1/num sampels
    class_weights = { row.label_int: 1/row[0] for _, row in df.groupby(['label_int']).size().reset_index().iterrows()}
    print(class_weights)
    # Set weights per sample in dataset
    weights = [class_weights[row.label_int] for _, row in df.iterrows()]
    return WeightedRandomSampler(weights=weights, num_samples=num_samples)

def get_camelyon_dataloader(opt):
    #base_folder = os.path.join(opt.data_input_dir, "camelyon17_imagedata")
    base_folder = opt.data_input_dir
    print('opt.data_input_dir: ', opt.data_input_dir)

    aug = {
        "cam17": {
            "randcrop": 64,
            "flip": True,
            "hue": True,
            "grayscale": opt.grayscale,
            "mean": [0.4313, 0.4156, 0.3663],  # values for train+unsupervised combined #TODO: find new mean
            "std": [0.2683, 0.2610, 0.2687],
            "bw_mean": [0.4120],  # values for train+unsupervised combined
            "bw_std": [0.2570],
        }  # values for labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
    }
    transform_train = transforms.Compose(
        [get_transforms(eval=False, aug=aug["cam17"])]
    )
    transform_valid = transforms.Compose(
        [get_transforms(eval=True, aug=aug["cam17"])]
    )
   
    if opt.training_data_csv:
        if os.path.isfile(opt.training_data_csv):
            print("reading csv file: ", opt.training_data_csv)
            train_df = pd.read_csv(opt.training_data_csv)
        else:
            raise Exception(f'Cannot find file: {opt.training_data_csv}')

        if os.path.isfile(opt.test_data_csv):
            print("reading csv file: ", opt.test_data_csv)
            val_df = pd.read_csv(opt.test_data_csv)
        else:
            raise Exception(f'Cannot find file: {opt.test_data_csv}')
    else:
        file_ = f"{opt.data_input_dir}/camelyon17_patches_unbiased.csv"
        #file_ = f"{opt.data_input_dir}/lnco_camelyon_patches.csv"
        if os.path.isfile(file_):
            print(f"reading {file_} file")
            df = pd.read_csv(file_)
        else:
            raise Expetion(f"Cannot find file {file_}")
    
        df = clean_data(opt.data_input_dir, df)

        slide_ids = df.slide_id.unique()
        random.shuffle(slide_ids)
        train_req_ids = []
        valid_req_ids = []
        # Take same number of slides from each site
        training_size = int(len(slide_ids)*0.8) # 80% training data
        validation_size = len(slide_ids) - training_size
        train_req_ids.extend([slide_id for slide_id in slide_ids[:training_size]])  # take first
        valid_req_ids.extend([
            slide_id for slide_id in slide_ids[training_size:training_size+validation_size]])  # take last
    
        print("train / valid / total")
        print(f"{len(train_req_ids)} / {len(valid_req_ids)} / {len(slide_ids)}")
    
        train_df = df[df.slide_id.isin(train_req_ids)]
        val_df = df[df.slide_id.isin(valid_req_ids)]

        print("Saving training/test set to file")
        train_df.to_csv(f'{opt.log_path}/training_patches.csv', index=False)
        val_df.to_csv(f'{opt.log_path}/test_patches.csv', index=False)
    
    print("training patches: ", train_df.groupby('label').size())
    print("test patches: ", val_df.groupby('label').size())


    ## DALI Augmentations
    #train_dataset = ExternalInputIterator(batch_size=opt.batch_size_multiGPU,
    #                                     data_frame=train_df,
    #                                     image_dir=base_folder, shuffle=True)
    #train_dataset = iter(train_dataset)

    #test_dataset = ExternalInputIterator(batch_size=opt.batch_size_multiGPU,
    #                                     data_frame=val_df,
    #                                     image_dir=base_folder,
    #                                     shuffle=False)
    #test_dataset = iter(test_dataset)

    #train_pipe = ExternalSourcePipeline(data_iterator=train_dataset, 
    #                                      batch_size=opt.batch_size_multiGPU,
    #                                      num_threads=4, device_id=0)

    #test_pipe = ExternalSourcePipeline(data_iterator=test_dataset,
    #                                     batch_size=opt.batch_size_multiGPU,
    #                                     num_threads=4, device_id=0)

    #train_pipe.build()
    #test_pipe.build()    

    #train_loader = DALIGenericIterator([train_pipe], ['images', 'labels'], len(train_dataset), fill_last_batch=False)
    #test_loader = DALIGenericIterator([test_pipe], ['images', 'labels'], len(test_dataset), fill_last_batch=False)

    #unsupervised_loader = train_loader
    #unsupervised_dataset = train_dataset


    train_dataset = ImagePatchesDataset(train_df, image_dir=base_folder, transform=transform_train)
    test_dataset = ImagePatchesDataset(val_df, image_dir=base_folder, transform=transform_valid)

    # Weighted sampler to handle class imbalance
    train_sampler = get_weighted_sampler(train_dataset, num_samples=len(train_dataset))
    #train_sampler = get_lnco_weighted_sampler(train_dataset, num_samples=len(train_dataset))

    # default dataset loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size_multiGPU, sampler=train_sampler, num_workers=16
    )

    unsupervised_dataset = train_dataset
    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=True,
        num_workers=16,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size_multiGPU, shuffle=False, num_workers=16
    )

    # create train/val split
    if opt.validate:
        print("Use train / val split")

        df = train_df

        slide_ids = df.slide_id.unique()
        random.shuffle(slide_ids)
        train_req_ids = []
        valid_req_ids = []
        # Take same number of slides from each site
        training_size = int(len(slide_ids)*0.9) # 90% of 80% training data
        validation_size = len(slide_ids) - training_size
        train_req_ids.extend([slide_id for slide_id in slide_ids[:training_size]])  # take first
        valid_req_ids.extend([
            slide_id for slide_id in slide_ids[training_size:training_size+validation_size]])  # take last
        
        print("train / valid / total")
        print(f"{len(train_req_ids)} / {len(valid_req_ids)} / {len(slide_ids)}")
        
        train_df = df[df.slide_id.isin(train_req_ids)]
        val_df = df[df.slide_id.isin(valid_req_ids)]
        
        print("training patches: ", train_df.groupby('label').size())
        print("validation patches: ", val_df.groupby('label').size())

        print("Saving training/test set to file")
        train_df.to_csv(f'{opt.log_path}/training_patches_exl_val.csv', index=False)
        val_df.to_csv(f'{opt.log_path}/validation_patches.csv', index=False)

        train_dataset = ImagePatchesDataset(train_df, image_dir=base_folder, transform=transform_train)
        test_dataset = ImagePatchesDataset(val_df, image_dir=base_folder, transform=transform_valid)

        train_loader = torch.utils.data.DataLoader(
              train_dataset,
              batch_size=opt.batch_size_multiGPU,
              shuffle=True,
              num_workers=10
        )

        # overwrite test_dataset and _loader with validation set
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size_multiGPU,
            shuffle=False,
            num_workers=10,
        )

    else:
        print("Use (train+val) / test split")

    return (
        unsupervised_loader,
        unsupervised_dataset,
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )

def clean_data(img_dir, dataframe):
    """ Clean the data """
    available_images = {f'camelyon17_imagedata/{file_}' for file_ in os.listdir(f"{img_dir}/camelyon17_imagedata")}
    for idx, row in dataframe.iterrows():
        if row.filename not in available_images:
            print(f"Removing non-existing file from dataset: {img_dir}/{row.filename}")
            dataframe = dataframe.drop(idx)
    return dataframe


def clean_lnco_data(img_dir, dataframe):
    """ Clean the data """
    available_images = {f'colon_imagedata/roi_lgl_norm/{file_}' for file_ in os.listdir(f"{img_dir}/colon_imagedata/roi_lgl_norm")}
    available_images2 = {f'colon_imagedata/tumor/{file_}' for file_ in os.listdir(f"{img_dir}/colon_imagedata/tumor")}
    for idx, row in dataframe.iterrows():
        if row.filename not in available_images:
            if row.filename not in available_images2:
                print(f"Removing non-existing file from dataset: {img_dir}/{row.filename}")
                dataframe = dataframe.drop(idx)
    return dataframe


class ImagePatchesDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

        self.patchify = self.make_patches(patch_size=16, overlap=2)
        self.color_jitter = transforms.ColorJitter(hue=0.1)

        self.mean = [0.4313, 0.4156, 0.3663]
        self.std = [0.2683, 0.2610, 0.2687]

        self.label_enum = {'TUMOR': 1, 'NONTUMOR': 0}

    def __len__(self):
        return len(self.dataframe.index)

    def make_patches(self, patch_size, overlap):
        def patchify(x, patch_size=patch_size, overlap=overlap):
            # Assumes input of shape [channel, width, height]
            x = (
                x.unfold(1, patch_size, patch_size // overlap) #create patches per row
                .unfold(2, patch_size, patch_size // overlap) # create patches per column
                .permute(1, 2, 0, 3, 4) # patchid_row, patchid_col, channel, patch_size, patch_size
            )
            x = x.reshape(
                x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]
            ) # reshape to num_patches, channel, w, h
            return x
        return patchify


    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        path = f"{self.image_dir}/{row.filename}"
        try:
            image = Image.open(path)
        except IOError:
            print(f"could not open {path}")
            return None

        if self.transform is not None:
            image = self.transform(image)
            ## Split to multiple patches
            #image = transforms.ToTensor()(image)
            #patches = self.patchify(image)
            #patches_tensor = torch.zeros_like(patches)
            #for idx in range(patches.shape[0]):
            #    patch_img = transforms.ToPILImage()(patches[idx, ...]).convert("RGB")
            #    patch_img = self.color_jitter(patch_img)
            #    patch_img = transforms.ToTensor()(patch_img)
            #    patch_img = transforms.Normalize(self.mean, self.std)(patch_img)
            #    patches_tensor[idx, ...] = patch_img
            #    
        else:
            image = transforms.ToTensor()(image)


        # label = row.label_int
        label = self.label_enum[row.label_new]
        #one_hot = np.eye(5, dtype = np.float64)[:, label]

        return image, label, row.patch_id, row.slide_id

