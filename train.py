import sys
import torch
import numpy as np
from tqdm import tqdm
from model import Detector
from utils import YoloLoss
from data_gen import Dataset, val_Dataset

def main():

    with open('anchor.txt', 'r') as f:
        lines = f.readlines()
    f.close()

    anchors = []
    for line in lines:
        *a, = list(map(int, line.split(',')))
        anchors.append(a)
    anchors = np.array(anchors, dtype='int')

    with open('2012_train.txt', 'r') as f:
        train_data = f.readlines()
    f.close()

    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    batch_size = 32
    initial_epochs = 0
    epochs = 100
    cls_num = len(classes)
    val_th_loss = sys.float_info.max
    weight_decay = 5e-4
    model_size = (416, 416)
    cv = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Detector(cls_num=cls_num)
    #model.load_state_dict(torch.load('checkpoint/ep103_loss:149.3047.pth'))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.2, verbose=1)
    yolo_loss = YoloLoss(cls_num=cls_num)

    for e in range(initial_epochs, epochs):
        cv_loss = 0.0
        losses = []
        val_losses = []
        sp = len(train_data)//cv
        for c in range(cv):
            tdata = train_data[:sp*c] + train_data[sp*(c+1):]
            vdata = train_data[sp*c:sp*(c+1)]
            train_data_set = Dataset(tdata, anchors, model_size=model_size, cls_num=cls_num)
            train_dataloader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
            val_data_set = val_Dataset(vdata, anchors, model_size=model_size, cls_num=cls_num)
            val_dataloader = torch.utils.data.DataLoader(val_data_set, batch_size=batch_size, shuffle=True)

            with tqdm(train_dataloader) as pbar:
                model.train()
                pbar.set_description("[Epoch %d - cv %d]" % (e + 1, c + 1))
                train_total_loss = 0.0
                train_total_size = 0
                for img, target in pbar:
                    images = img.to(device)
                    targets = [target[0].to(device), target[1].to(device), target[2].to(device)]
                    outputs = model(images)
                    loss = yolo_loss(outputs, targets)
                    l2_reg = torch.tensor(0.).to(device)
                    for name, param in model.named_parameters():
                        if param.requires_grad and ('.conv.weight' in name):    
                            l2_reg += torch.norm(param, p=2)
                        loss += l2_reg*weight_decay
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(loss=loss.item()/images.shape[0])
                    train_total_loss += loss.item()
                    train_total_size += images.shape[0]
                losses.append(train_total_loss/train_total_size)

            model.eval()
            with torch.no_grad():
                val_total_loss = 0.0
                val_total_size = 0
                for img, tgt in val_dataloader:
                    image = img.to(device)
                    tgt = [tgt[0].to(device), tgt[1].to(device), tgt[2].to(device)]
                    output = model(image)
                    loss = yolo_loss(output, tgt)
                    val_total_loss += loss.item()
                    val_total_size += img.shape[0]
                val_losses.append(val_total_loss/val_total_size)

        cv_loss = sum(losses) / cv
        lr_scheduler.step(cv_loss)
        print('val_loss: {:.5f}'.format(cv_loss),flush=True)
        print('losses: {}'.format(losses))
        print('val_losses: {}'.format(val_losses))
        if val_th_loss > (cv_loss):
            val_th_loss = cv_loss
            torch.save(model.state_dict(), 'checkpoint/ep{:03d}_loss:{:.4f}.pth'.format(e+1, cv_loss))

if __name__ == '__main__':
    main()