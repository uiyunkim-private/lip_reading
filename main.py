from pytorch.utils.dataset_recorder import Dataset_recorder
import torch
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
import cv2
import os
import torchvision
from facenet_pytorch import MTCNN



def video_file_loader(path):
    print(path)
    device = torch.device('cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=1,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device)

    images = []
    cap = cv2.VideoCapture(path)

    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        mtcnn = mtcnn(frame)
        print(mtcnn)
        gray = cv2.cvtColor(mtcnn, cv2.COLOR_BGR2GRAY)
        images.append(gray)

    print(len(images))
    big_window = []
    for i in range(len(images)-4):
        small_window = []
        for j in range(5):
            small_window.append((images[i+j]))
        big_window.append(small_window)

    big_window = torch.Tensor(big_window)
    return big_window




#trainset = torchvision.datasets.DatasetFolder(root=os.path.join(os.getcwd(),'unprocessed'),loader=video_file_loader,extensions='mp4')

#print(trainset[0][0].shape)

#print(trainset)
#
#
# class TimeDistributed(torch.nn.Module):
#     def __init__(self, module, batch_first=False):
#         super(TimeDistributed, self).__init__()
#         self.module = module
#         self.batch_first = batch_first
#
#     def forward(self, x):
#
#         if len(x.size()) <= 2:
#             return self.module(x)
#
#         # Squash samples and timesteps into a single axis
#         x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
#
#         y = self.module(x_reshape)
#
#         # We have to reshape Y
#         if self.batch_first:
#             y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
#         else:
#             y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
#
#         return y
#
#
# t1 = torch.Tensor([2,3])
# t2 = torch.Tensor([0,1])
#
# tensorsum = t1 + t2
#
# print(tensorsum)
#
# model = torch.nn.Sequential(
#     TimeDistributed(torchvision.models.resnet18()),
#     torch.nn.LSTM(1000,512),
#     torch.nn.LSTM(1000,512),
#     torch.nn.Softmax(26)
#
# )
#
# model.train()
# print(model)




if __name__ == "__main__":

    Dataset_recorder(class_name='alfa',type='train',shape=(120,120),save_original=True)