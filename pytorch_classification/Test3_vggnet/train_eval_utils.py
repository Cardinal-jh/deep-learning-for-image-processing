import sys

from tqdm import tqdm
import torch


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    #mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()
        #mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 打印平均loss
        #data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
    #return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    # 用于存储预测正确的样本个数
    #sum_num = torch.zeros(1).to(device)
    # 统计验证集样本总数目
    num_samples = len(data_loader.dataset)

    sample_num = 0

    # 打印验证进度
    data_loader = tqdm(data_loader, desc="validation...", file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        #sum_num += torch.eq(pred, labels.to(device)).sum()

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    # 计算预测正确的比例
    #acc = sum_num.item() / num_samples

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
    #return acc






