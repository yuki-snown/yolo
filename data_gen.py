import torch
import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

class Dataset(torch.utils.data.Dataset):
    def __init__(self, datas, anchors, model_size=(320, 320), cls_num=1):
        self.datas = datas
        self.data_num = len(self.datas)
        self.model_size = model_size # width, height
        self.cls_num = cls_num
        self.anchors = anchors
        self.seed = 0.25*(np.random.rand()+1)

    def __len__(self):
        return self.data_num

    def rand(self):
      return 1.0+.5*np.random.rand()

    def __getitem__(self, idx):
        data = self.datas[idx].split()
        bboxes = []
        classes = [] 
        img_path = data.pop(0)
        img = Image.open(img_path).convert('RGB') # HWC
        w1, h1 = img.size
        rnd = self.seed + (1-self.seed)*np.random.rand()
        w2 = int(self.model_size[0]*rnd)
        h2 = int(self.model_size[1]*rnd)
        img = img.resize((w2, h2))
        x = np.random.randint(0, self.model_size[0] - w2)
        y = np.random.randint(0, self.model_size[1] - h2)
        image = Image.new('RGB', self.model_size, (0, 0, 0))
        image.paste(img, (x, y))
        img = image
        flip = np.random.rand()<.5
        if flip:
          img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = np.asarray(img, dtype=np.uint8)/255.
        '''
        img = rgb_to_hsv(img)
        hue = np.random.rand()*0.2-0.1
        sat = self.rand() if np.random.rand()<.5 else 1/self.rand()
        val = self.rand() if np.random.rand()<.5 else 1/self.rand()
        img[..., 0] += hue
        img[..., 1] *= sat
        img[..., 2] *= val
        img[img>1] = 1
        img[img<0] = 0
        img = hsv_to_rgb(img)
        '''
        img = np.asarray(img, dtype=np.float32)        
        img = torch.from_numpy(img)
        img = img.transpose(0,1).transpose(2,0) # CHW
        channels = 1+4+self.cls_num
        small_scale = np.zeros([channels*3, self.model_size[1]//8, self.model_size[0]//8], dtype='float32')
        midium_scale = np.zeros([channels*3, self.model_size[1]//16, self.model_size[0]//16], dtype='float32')
        large_scale = np.zeros([channels*3, self.model_size[1]//32, self.model_size[0]//32], dtype='float32')
        for d in data:
          x0,y0,x1,y1,cls = list(map(lambda x: int(x), d.split(',')))
          x0 = int(x0*(w2/w1))+x
          x1 = int(x1*(w2/w1))+x
          y0 = int(y0*(h2/h1))+y
          y1 = int(y1*(h2/h1))+y
          center_x, center_y = (x0+x1)/2, (y0+y1)/2
          width, height = x1-x0, y1-y0
          if flip:
            center_x = self.model_size[0] - center_x

          i, j = int(center_x // 8), int(center_y // 8)
          for anchor, ind in enumerate(range(0,channels*3,channels)):
            small_scale[ind:4+ind,j,i] = [(center_x%8)/8,
                                        (center_y%8)/8,
                                        np.log(width/anchors[anchor][0]),
                                        np.log(height/anchors[anchor][1])]
            small_scale[4+ind,j,i] = 1.0
            small_scale[5+ind+cls,j,i] = 1.0

          i, j = int(center_x // 16), int(center_y // 16)
          for anchor, ind in enumerate(range(0,channels*3,channels)):
            midium_scale[ind:4+ind,j,i] = [(center_x%16)/16,
                                          (center_y%16)/16,
                                          np.log(width/anchors[anchor+3][0]),
                                          np.log(height/anchors[anchor+3][1])]
            midium_scale[4+ind,j,i] = 1.0
            midium_scale[5+ind+cls,j,i] = 1.0

          i, j = int(center_x // 32), int(center_y // 32)
          for anchor, ind in enumerate(range(0,channels*3,channels)):
            large_scale[ind:4+ind,j,i] = [(center_x%32)/32,
                                        (center_y%32)/32,
                                        np.log(width/anchors[anchor+6][0]),
                                        np.log(height/anchors[anchor+6][1])]
            large_scale[4+ind,j,i] = 1.0
            large_scale[5+ind+cls,j,i] = 1.0

        return img, [torch.from_numpy(small_scale), torch.from_numpy(midium_scale), torch.from_numpy(large_scale)]


class val_Dataset(torch.utils.data.Dataset):
    def __init__(self, datas, anchors, model_size=(320, 320), cls_num=1):
        self.datas = datas
        self.data_num = len(self.datas)
        self.model_size = model_size # width, height
        self.cls_num = cls_num
        self.anchors = anchors

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        data = self.datas[idx].split()
        bboxes = []
        classes = [] 
        img_path = data.pop(0)
        img = Image.open(img_path).convert('RGB') # HWC
        w1, h1 = img.size
        rnd = 1.0
        w2 = int(self.model_size[0]*rnd)
        h2 = int(self.model_size[1]*rnd)
        img = img.resize((w2, h2))
        x, y = 0, 0
        image = Image.new('RGB', self.model_size, (128, 128, 128))
        image.paste(img, (x, y))
        img = image
        img = np.asarray(img, dtype=np.uint8)
        img = np.asarray(img/255., dtype=np.float32)        
        img = torch.from_numpy(img)
        img = img.transpose(0,1).transpose(2,0) # CHW
        channels = 1+4+self.cls_num
        small_scale = np.zeros([channels*3, self.model_size[1]//8, self.model_size[0]//8], dtype='float32')
        midium_scale = np.zeros([channels*3, self.model_size[1]//16, self.model_size[0]//16], dtype='float32')
        large_scale = np.zeros([channels*3, self.model_size[1]//32, self.model_size[0]//32], dtype='float32')
        for d in data:
          x0,y0,x1,y1,cls = list(map(lambda x: int(x), d.split(',')))
          x0 = int(x0*(w2/w1))+x
          x1 = int(x1*(w2/w1))+x
          y0 = int(y0*(h2/h1))+y
          y1 = int(y1*(h2/h1))+y
          center_x, center_y = (x0+x1)/2, (y0+y1)/2
          width, height = x1-x0, y1-y0

          i, j = int(center_x // 8), int(center_y // 8)
          for anchor, ind in enumerate(range(0,channels*3,channels)):
            small_scale[ind:4+ind,j,i] = [(center_x%8)/8,
                                        (center_y%8)/8,
                                        np.log(width/anchors[anchor][0]),
                                        np.log(height/anchors[anchor][1])]
            small_scale[4+ind,j,i] = 1.0
            small_scale[5+ind+cls,j,i] = 1.0

          i, j = int(center_x // 16), int(center_y // 16)
          for anchor, ind in enumerate(range(0,channels*3,channels)):
            midium_scale[ind:4+ind,j,i] = [(center_x%16)/16,
                                          (center_y%16)/16,
                                          np.log(width/anchors[anchor+3][0]),
                                          np.log(height/anchors[anchor+3][1])]
            midium_scale[4+ind,j,i] = 1.0
            midium_scale[5+ind+cls,j,i] = 1.0

          i, j = int(center_x // 32), int(center_y // 32)
          for anchor, ind in enumerate(range(0,channels*3,channels)):
            large_scale[ind:4+ind,j,i] = [(center_x%32)/32,
                                        (center_y%32)/32,
                                        np.log(width/anchors[anchor+6][0]),
                                        np.log(height/anchors[anchor+6][1])]
            large_scale[4+ind,j,i] = 1.0
            large_scale[5+ind+cls,j,i] = 1.0

        return img, [torch.from_numpy(small_scale), torch.from_numpy(midium_scale), torch.from_numpy(large_scale)]