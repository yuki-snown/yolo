import torch
import torch.nn as nn
import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import collect_boxes
from model import Detector
from data_gen import val_Dataset

def main():

    with open('anchor.txt', 'r') as f:
        lines = f.readlines()
    f.close()

    anchors = []
    for line in lines:
        *a, = list(map(int, line.split(',')))
        anchors.append(a)
    anchors = np.array(anchors, dtype='int')

    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    cls_num = len(classes)
    channels = 1+4+cls_num
    threshold = 0.40
    iou = 0.40
    model_size=(416, 416)
    img_path = './test.jpeg'

    hsv_tuples = [(x / cls_num, 1., 1.) for x in range(cls_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Detector(cls_num=cls_num)
    model.load_state_dict(torch.load('ep100_loss_48.6804.pth', torch.device(device)))
    model.to(device)

    model.eval()
    with torch.no_grad():
        img = Image.open(img_path).convert('RGB')
        img = img.resize(model_size)
        img = np.asarray(img, dtype=np.uint8)/255.
        img = torch.from_numpy(img).float()
        img = img.transpose(0,1).transpose(2,0) # CHW
        img = torch.unsqueeze(img, 0)
        image = img.to(device)
        output = model(image)
        sigmoid = nn.Sigmoid()
        a, b = 0, 0
        _, _, h, w = image.shape
        img = image[0].to('cpu').detach().numpy().copy()
        img = img.transpose((1,2,0)) * 255
        img = np.asarray(img, dtype=np.uint8)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        anchor = torch.tensor(anchors).to(device)
        bbox = np.array([])
        for c in range(3):
            for area in range(0,channels*3, channels):
                for i in range(output[c].shape[2]):
                    for j in range(output[c].shape[3]):
                        o = sigmoid(output[c][0,area+4:area+channels,i,j])
                        xy = sigmoid(output[c][0,area:area+2,i,j])*(8*(2**c)) + torch.Tensor([j*(8*(2**c)), i*(8*(2**c))]).to(device)
                        wh = torch.exp(output[c][0,area+2:area+4,i,j])* torch.Tensor(anchors[area//channels+c*3]).to(device)
                        if o[0] > threshold:
                            box = torch.cat([xy,wh], dim=0).to('cpu').detach().numpy()
                            box = np.append(box, o[0].item())
                            box = np.append(box, o[1:].argmax().item())
                            pos = np.append(box[:2]-box[2:4]/2, box[:2]+box[2:4]/2)
                            pos[0] = max(pos[0], 0)
                            pos[1] = max(pos[1], 0)
                            pos[2] = min(pos[2], model_size[0])
                            pos[3] = min(pos[3], model_size[1])
                            box = np.append(pos, box[4:])
                            bbox = np.append(bbox, box, axis=0)
        bbox = bbox.reshape(-1,6)
        ids = collect_boxes(bbox, iou)
        for b in bbox[ids]:
            draw.rectangle((int(b[0]), int(b[1]), int(b[2]), int(b[3])), outline=colors[int(b[5])])      
            draw.text((int(b[0]), int(b[1])), "{}: {:.2f}".format(classes[int(b[5])], b[4]), font=ImageFont.truetype("FiraMono-Medium.otf", size=model_size[0]//32))
        img.show()

if __name__ == '__main__':
    main()