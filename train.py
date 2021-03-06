import torch
from torch.autograd import Variable
import cv2
from torchvision import transforms
from utils import getGaussPyr_img

def retrain(trainloader, model, use_cuda, epoch, criterion, optimizer):
    model.train()
    correct, total = 0, 0
    acc_sum, loss_sum = 0, 0
    i = 0
    for batch_idx, (data, target) in enumerate(trainloader):
        if use_cuda:
            target=target.cuda()
            data=data.cuda()


        optimizer.zero_grad()
        output = model(data)

        # calculate accuracy
        correct += (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
        total += trainloader.batch_size
        train_acc = 100. * correct / total
        acc_sum += train_acc
        i += 1

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tTraining Accuracy: {:.3f}%'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item(), train_acc))

    acc_avg = acc_sum / i
    loss_avg = loss_sum / len(trainloader.dataset)
    print()
    print('Train Epoch: {}\tAverage Loss: {:.3f}\tAverage Accuracy: {:.3f}%'.format(epoch, loss_avg, acc_avg))

    with open('/content/drive/MyDrive/cam/train_acc.txt', 'a') as f:
        f.write(str(acc_avg))
    f.close()
    with open('/content/drive/MyDrive/cam/train_loss.txt', 'a') as f:
        f.write(str(loss_avg))
    f.close()



def retrain_aug(trainloader, model, use_cuda, epoch, criterion, optimizer):
    model.train()
    correct, total = 0, 0
    acc_sum, loss_sum = 0, 0
    i = 0
    for batch_idx, (data, target) in enumerate(trainloader):
        if use_cuda:
            target=target.cuda()
        #print(type(data),type(target))
            for i in data:
                i=i.cuda()

        for data_i in data:
            optimizer.zero_grad()
            output = model(data_i)

            # calculate accuracy
            correct += (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
            total += trainloader.batch_size
            train_acc = 100. * correct / total
            acc_sum += train_acc
            i += 1

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tTraining Accuracy: {:.3f}%'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item(), train_acc))

    acc_avg = acc_sum / i
    loss_avg = loss_sum / len(trainloader.dataset)
    print()
    print('Train Epoch: {}\tAverage Loss: {:.3f}\tAverage Accuracy: {:.3f}%'.format(epoch, loss_avg, acc_avg))

    with open('/content/drive/MyDrive/cam/train_acc.txt', 'a') as f:
        f.write(str(acc_avg))
    f.close()
    with open('/content/drive/MyDrive/cam/train_loss.txt', 'a') as f:
        f.write(str(loss_avg))
    f.close()

def retest(testloader, model, use_cuda, criterion, epoch, RESUME):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(testloader.dataset)
    test_acc = 100. * correct / len(testloader.dataset)
    result = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(testloader.dataset), test_acc)
    print(result)

    # Save checkpoint.
    if epoch % 2 == 0:
        torch.save(model.state_dict(), '/content/drive/MyDrive/checkpoint/' + 'vgg16_bn' +str(RESUME + int(epoch / 10)) + '.pth')
        
