import torch.nn as nn
from .bi_lstm import BiLSTM
from .mobilenet import MobileNetV1
from .resnet import resnet50


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.mobilenet_v1 = MobileNetV1()
        self.mobilenet_v1.apply(self.mobilenet_v1.weight_init)

        self.bilstm = BiLSTM(args)
        self.bilstm.apply(self.bilstm.weight_init)

        self.conv_images = nn.Conv2d(1024, args.feature_size, 1)
        self.conv_text = nn.Conv2d(1024, args.feature_size, 1)
    
    def forward(self, images, text, text_length):
        image_features = self.mobilenet_v1(images)
        text_features = self.bilstm(text, text_length)
        image_embeddings, text_embeddings= self.build_joint_embeddings(image_features, text_features)

        return image_embeddings, text_embeddings

    def build_joint_embeddings(self, images_features, text_features):
        
        #images_features = images_features.permute(0,2,3,1)
        #text_features = text_features.permute(0,3,1,2)
         
        image_embeddings = self.conv_images(images_features).squeeze()
        text_embeddings = self.conv_text(text_features).squeeze()

        return image_embeddings, text_embeddings
