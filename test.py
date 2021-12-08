import argparse
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
import csv
from torchvision import transforms as transforms
from dataloaders import dataset
from misc import progress_bar
from networks import *

CLASSES = ('buildings', 'forest', 'glacier', 'mountain', 'sea', 'street')


def main():
    parser = argparse.ArgumentParser(description="Classification with PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--testBatchSize', default=8, type=int, help='testing batch size')
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
        self.test_batch_size = config.testBatchSize
        self.resume = config.resume
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.test_loader = None

    def load_data(self):
        test_transform = transforms.Compose([transforms.ToTensor()])
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

        state_dict = torch.load(self.resume)
        self.model.load_state_dict(state_dict, strict=False)

        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        f = open('C:/Users/IOTLAB/Documents/output.csv', 'w')
        writer = csv.writer(f)
        fields = ["ID", "label"]
        writer.writerow(fields)
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                target = torch.argmax(target, dim=1)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                print(prediction[1].cpu().numpy())
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
                test_results = prediction[1].cpu().numpy()
                for elements in range(len(test_results)):
                    row = [total + elements, CLASSES[test_results[elements]]]
                    writer.writerow(row)
                total += target.size(0)
                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))
        f.close()
        return test_loss, test_correct / total

    def run(self):
        self.load_data()
        self.load_model()
        test_result = self.test()
        print("===> ACC. PERFORMANCE: %.3f%%" % (test_result[1] * 100))


if __name__ == '__main__':
    main()
