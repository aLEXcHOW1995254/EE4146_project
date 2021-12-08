import argparse
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms as transforms
from dataloaders import dataset
from misc import progress_bar
from networks import *

CLASSES = ('buildings', 'forest', 'glacier', 'mountain', 'sea', 'street')


def main():
    parser = argparse.ArgumentParser(description="Classification with PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=2, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=4, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=4, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    parser.add_argument('--resume', type=str, default=None, help='the path of pretrained model to resume')

    args = parser.parse_args()

    torch.cuda.empty_cache()
    solver = Solver(args)
    solver.run()


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.resume = config.resume
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = dataset.Classification_Dataset(root_dir='./dataloaders/train10500',
                                                   csv_file='./raw_images/train_labels.csv', transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size,
                                                        shuffle=True)
        test_set = dataset.Classification_Dataset(root_dir='./raw_images/test', csv_file='./dataloaders/naive_baseline.csv',
                                                  transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # self.model = alexnet(pretrained=True).to(self.device)
        # self.model = vgg11(pretrained=True).to(self.device)
        # self.model = vgg13(pretrained=True).to(self.device)
        # self.model = vgg16(pretrained=True).to(self.device)
        # self.model = vgg19(pretrained=True).to(self.device)
        # self.model = resnet18(pretrained=True).to(self.device)
        # self.model = resnet34(pretrained=True).to(self.device)
        # self.model = resnet50(pretrained=True).to(self.device)
        # self.model = resnet101(pretrained=True).to(self.device)
        # self.model = resnet152(pretrained=True).to(self.device)
        # self.model = densenet121(pretrained=True, pretrain_model_path=self.resume, num_classes=len(CLASSES)).to(self.device)
        # self.model = densenet161(pretrained=True, pretrain_model_path=self.resume, num_classes=len(CLASSES)).to(self.device)
        # self.model = densenet169(pretrained=True, pretrain_model_path=self.resume, num_classes=len(CLASSES)).to(self.device)
        self.model = densenet201(pretrained=True, pretrain_model_path=self.resume, num_classes=len(CLASSES)).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            target = torch.argmax(target, dim=1)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                target = torch.argmax(target, dim=1)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def save(self):
        # model_out_path = "./checkpoints/alexnet-owt-7be5be79.pth"
        # model_out_path = "./checkpoints/vgg11-8a719046.pth"
        # model_out_path = "./checkpoints/vgg13-19584684.pth"
        # model_out_path = "./checkpoints/vgg16-397923af.pth"
        # model_out_path = "./checkpoints/vgg19-dcbb9e9d.pth"
        # model_out_path = "./checkpoints/resnet18-5c106cde.pth"
        # model_out_path = "./checkpoints/resnet34-333f7ec4.pth"
        # model_out_path = "./checkpoints/resnet50-19c8e357.pth"
        # model_out_path = "./checkpoints/resnet101-5d3b4d8f.pth"
        # model_out_path = "./checkpoints/resnet152-b121ed2d.pth"
        # model_out_path = "./checkpoints/model.pth"
        # model_out_path = "./checkpoints/densenet121-a639ec97.pth"
        # model_out_path = "./checkpoints/densenet161-8d451a50.pth"
        # model_out_path = "./checkpoints/densenet169-b2777c0a.pth"
        model_out_path = "./checkpoints/densenet201-c1103571.pth"
        torch.save(self.model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        for epoch in range(1, self.epochs + 1):
            print("\n===> epoch: %d/%d" % (epoch, self.epochs))
            train_result = self.train()
            print(train_result)
            test_result = self.test()
            self.optimizer.step()
            self.scheduler.step()
            accuracy = max(accuracy, test_result[1])
            if epoch == self.epochs:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                self.save()


if __name__ == '__main__':
    main()
