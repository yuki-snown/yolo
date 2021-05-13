import torch
import torch.nn as nn

def cul_iou(box1, box2):
  x0, y0, x1, y1 = box1[:4]
  x2, y2, x3, y3 = box2[:4]
  area1 = (x1-x0)*(y1-y0)
  area2 = (x3-x2)*(y3-y2)
  area3 = (max(x0, x2)-min(x1,x3))*(max(y0, y2)-min(y1,y3))
  return area3/(area1+area2-area3)

def collect_boxes(bbox, iou):
  idxs = set(list(range(len(bbox))))
  rm_ids = set()
  for i in range(len(bbox)):
    for j in range(i+1, len(bbox)):
      if (bbox[i, -1]==bbox[j, -1])and(iou<cul_iou(bbox[i], bbox[j])):
        val = i if min(bbox[i, -2], bbox[j, -2]) == bbox[i, -2] else j
        rm_ids.add(val)
  return list(idxs - rm_ids)

class YoloLoss(nn.Module):
    def __init__(self, cls_num):
      super().__init__()
      self.cls_num = cls_num
      self.channels = 1+4+self.cls_num
      self.sigmoid = nn.Sigmoid()
      self.bce_loss = nn.BCELoss(size_average=False, reduction='sum')
      self.mse_loss = nn.MSELoss(size_average=False, reduction='sum')

    def forward(self, outputs, targets):
      loss = 0.0
      for c in range(3):
        out_xy = torch.cat([self.sigmoid(outputs[c][:,::self.channels])*targets[c][:,4::self.channels], self.sigmoid(outputs[c][:,1::self.channels])*targets[c][:,4::self.channels]], 1)
        tgt_xy = torch.cat([targets[c][:,::self.channels], targets[c][:,1::self.channels]], 1)        

        out_wh = torch.cat([outputs[c][:,2::self.channels]*targets[c][:,4::self.channels], outputs[c][:,3::self.channels]*targets[c][:,4::self.channels]], 1)
        tgt_wh = torch.cat([targets[c][:,2::self.channels], targets[c][:,3::self.channels]], 1)
        
        b, _, h, w = targets[c].size()
        scale1 = torch.repeat_interleave(targets[c][:, 4].view(b, 1, h, w), self.cls_num, dim=1)
        scale2 = torch.repeat_interleave(targets[c][:,4+self.channels].view(b, 1, h, w), self.cls_num, dim=1)
        scale3 = torch.repeat_interleave(targets[c][:,4+self.channels*2].view(b, 1, h, w), self.cls_num, dim=1)
        out_obj = torch.cat([self.sigmoid(outputs[c][:,5:self.channels])*scale1,
                             self.sigmoid(outputs[c][:,self.channels+5:self.channels*2])*scale2, 
                             self.sigmoid(outputs[c][:,self.channels*2+5:])*scale3], 1)
        tgt_obj = torch.cat([targets[c][:,5:self.channels], targets[c][:,self.channels+5:self.channels*2], targets[c][:,self.channels*2+5:]], 1)

        xy_loss = self.bce_loss(out_xy, tgt_xy)
        wh_loss = self.mse_loss(out_wh, tgt_wh) / 2
        obj_loss = self.bce_loss(out_obj, tgt_obj)
        conf_loss = self.bce_loss(self.sigmoid(outputs[c][:,4::self.channels]), targets[c][:,4::self.channels])
        loss += (xy_loss + wh_loss + conf_loss + obj_loss)
      return loss