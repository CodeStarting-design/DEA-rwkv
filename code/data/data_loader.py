import os, random
import torch.utils.data as data
import torch
from PIL import Image
from torchvision.transforms.functional import hflip, rotate, crop
from torchvision.transforms import ToTensor, RandomCrop, Resize
from torchvision import transforms
import os
import sys
os.chdir(sys.path[0])
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

class TrainDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(TrainDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = os.listdir(hazy_path)
        self.clear_image_list = os.listdir(clear_path)

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = hazy_image_name.split('_')[0] + '.png'

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        crop_params = RandomCrop.get_params(hazy, [256, 256])
        rotate_params = random.randint(0, 3) * 90

        hazy = crop(hazy, *crop_params)
        clear = crop(clear, *crop_params)

        hazy = rotate(hazy, rotate_params)
        clear = rotate(clear, rotate_params)

        to_tensor = ToTensor()

        hazy = to_tensor(hazy)
        clear = to_tensor(clear)

        return hazy, clear

    def __len__(self):
        return len(self.hazy_image_list)


class TestDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(TestDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = os.listdir(hazy_path)
        self.clear_image_list = os.listdir(clear_path)
        self.hazy_image_list.sort()
        self.clear_image_list.sort()

    def __getitem__(self, index):
        # data shape: C*H*W

        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = hazy_image_name.split('_')[0] + '.png'

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        to_tensor = ToTensor()

        hazy = to_tensor(hazy)
        clear = to_tensor(clear)

        return hazy, clear, hazy_image_name

    def __len__(self):
        return len(self.hazy_image_list)


class ValDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(ValDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = os.listdir(hazy_path)
        self.clear_image_list = os.listdir(clear_path)
        self.hazy_image_list.sort()
        self.clear_image_list.sort()

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = hazy_image_name.split('_')[0] + '.png'

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        to_tensor = ToTensor()

        hazy = to_tensor(hazy)
        clear = to_tensor(clear)

        return {'hazy': hazy, 'clear': clear, 'filename': hazy_image_name}

    def __len__(self):
        return len(self.hazy_image_list)
    

def augment(imgs=[], width=720,height=720, edge_decay=0., only_h_flip=False): # 要对连续三帧进行处理
    _ , H, W = imgs[0].shape
    Hc, Wc = [height,width]

    # simple re-weight for the edge
    if random.random() < Hc / H * edge_decay:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H-Hc)

    if random.random() < Wc / W * edge_decay:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W-Wc)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][:,Hs:(Hs+Hc), Ws:(Ws+Wc)]

    # horizontal flip
    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = torch.flip(imgs[i], [2])

    if not only_h_flip:
        # bad data augmentations for outdoor
        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = torch.rot90(imgs[i], rot_deg, (1, 2))
            
    return imgs


def align(imgs=[], width=500,height=500):
    _ , H, W = imgs[0].shape
    Hc, Wc = [height,width]

    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][:,Hs:(Hs+Hc), Ws:(Ws+Wc)]

    return imgs

class REVIDE_Dataset(data.Dataset):
    def __init__(self, data_dir,sub_dir,mode,width=500,height=500, edge_decay=0, only_h_flip=False): 
        
        self.root_dir = os.path.join(data_dir,sub_dir)
        self.width = width
        self.height = height 
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip
        self.mode=mode

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.input_frames = [] # 保存相对路径
        self.target_frames = []
        self.scenes = sorted(os.listdir(os.path.join(self.root_dir, 'hazy'))) # 场景文件夹的名称
        for i in range(len(self.scenes)):
            scene_folder = self.scenes[i]
            hazy_scene_path = os.path.join(self.root_dir, 'hazy', scene_folder)
            gt_scene_path = os.path.join(self.root_dir, 'gt', scene_folder)
            frame_files_hazy = sorted(os.listdir(hazy_scene_path)) # 在文件夹中的每一帧
            frame_files_gt = sorted(os.listdir(gt_scene_path))
            for j in range(len(frame_files_hazy)):
                frame_hazy_path=os.path.join(hazy_scene_path,frame_files_hazy[j])
                frame_gt_path=os.path.join(gt_scene_path,frame_files_gt[j])
                self.input_frames.append(frame_hazy_path)
                self.target_frames.append(frame_gt_path)
        

    def __len__(self):
        return len(self.input_frames)

    def __getitem__(self, idx):
        """
        在该方法中，首先根据文件路径取出相应的连续三帧
        再对这三帧进行简单的预处理
        沿着第一维度进行堆叠，返回
        """
        input_frame_path = self.input_frames[idx]
        target_frame_path = self.target_frames[idx]

        frame_hazy = Image.open(input_frame_path).convert('RGB')
        frame_hazy = self.transform(frame_hazy)
        frame_hazy=(frame_hazy)*2-1
            
        frame_gt = Image.open(target_frame_path).convert('RGB')
        frame_gt = self.transform(frame_gt)
        frame_gt=(frame_gt)*2-1


        # 使用os.path模块获取文件路径的各个部分
        dir_path, file_name = os.path.split(input_frame_path)
        _, last_dir = os.path.split(dir_path)

        # 构建后两层文件路径字符串
        result_path = os.path.join(last_dir, file_name)

        if self.mode == 'train':
            # 进行数据增强
            [frame_gt,frame_hazy]=augment([frame_gt,frame_hazy],self.width,self.height,self.edge_decay,self.only_h_flip)
        
        if self.mode == 'valid':
            [frame_gt,frame_hazy] = align([frame_gt,frame_hazy], self.width,self.height)
            return frame_hazy,frame_gt
        if self.mode == 'test':
            return frame_hazy,frame_gt
        return frame_hazy,frame_gt
    
class DehazeDataset(data.Dataset):
    def __init__(self, data_dir,sub_dir,mode,width=500,height=500, edge_decay=0, only_h_flip=False): # 每次需要取出的是三帧，在数据集读取时将连续三帧的文件路径直接保存
        
        self.root_dir = os.path.join(data_dir,sub_dir)
        self.width = width
        self.height = height 
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip
        self.mode=mode

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.input_frames = [] # 保存相对路径
        self.target_frames = []
        self.scenes = sorted(os.listdir(os.path.join(self.root_dir, 'hazy'))) # 场景文件夹的名称
        for i in range(len(self.scenes)):
            scene_folder = self.scenes[i]
            hazy_scene_path = os.path.join(self.root_dir, 'hazy', scene_folder)
            gt_scene_path = os.path.join(self.root_dir, 'gt', scene_folder)
            video_folders = sorted(os.listdir(hazy_scene_path)) # 场景下的全部雾霾视视频文件夹名
            for video_folder in video_folders: # 遍历某一个文件夹
                hazy_video_path = os.path.join(hazy_scene_path, video_folder) # 某一个文件夹的名字
                gt_video_path = os.path.join(gt_scene_path, video_folder)
                frame_files_hazy = sorted(os.listdir(hazy_video_path)) # 在文件夹中的每一帧
                frame_files_gt = sorted(os.listdir(gt_video_path))
                for j in range(len(frame_files_hazy)):
                    frame_hazy_path= os.path.join(hazy_video_path,frame_files_hazy[j])
                    frame_gt_path= os.path.join(gt_video_path,frame_files_gt[j])
                    self.input_frames.append(frame_hazy_path)
                    self.target_frames.append(frame_gt_path)
        

    def __len__(self):
        return len(self.input_frames)

    def __getitem__(self, idx):
        """
        在该方法中，首先根据文件路径取出相应的连续三帧
        再对这三帧进行简单的预处理
        沿着第一维度进行堆叠，返回
        """
        input_frame_path = self.input_frames[idx]
        target_frame_path = self.target_frames[idx]

        frame_hazy = Image.open(input_frame_path).convert('RGB')
        frame_hazy = self.transform(frame_hazy)
        frame_hazy=(frame_hazy)*2-1
            
        frame_gt = Image.open(target_frame_path).convert('RGB')
        frame_gt = self.transform(frame_gt)
        frame_gt=(frame_gt)*2-1


        # 使用os.path模块获取文件路径的各个部分
        dir_path, file_name = os.path.split(input_frame_path)
        _, last_dir = os.path.split(dir_path)

        # 构建后两层文件路径字符串
        result_path = os.path.join(last_dir, file_name)

        if self.mode == 'train':
            # 进行数据增强
            [frame_gt,frame_hazy]=augment([frame_gt,frame_hazy],self.width,self.height,self.edge_decay,self.only_h_flip)
        
        if self.mode == 'valid':
            [frame_gt,frame_hazy] = align([frame_gt,frame_hazy], self.width,self.height)
            return frame_hazy,frame_gt
        if self.mode == 'test':
            return frame_hazy,frame_gt
        return frame_hazy,frame_gt