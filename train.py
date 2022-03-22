import os
import torch
from torch.autograd import Variable
from torch import optim
import model
import dataloader
import copy,random
import argparse
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default='../method6/data/train256/WVtrain/')
parser.add_argument('--val_dir', type=str, default='../method6/data/val256/WVval2/')
parser.add_argument('--outputs_dir',help='output model dir',default='./WVmodel/')
parser.add_argument('--epoch', type=int, default=2000)
parser.add_argument('--trainBatchSize',default=1)
parser.add_argument('--testBatchSize', default=2)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_decay_step', type=int, default=250)
parser.add_argument('--lr_decay_rate', type=float, default=0.95)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--gamma',type=float,default=0.01,help='scheduler gamma')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

config = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():

    seed = random.randint(1, 10000)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

    train_set = dataloader.get_training_set(config.train_dir)
    val_set = dataloader.get_val_set(config.val_dir)

    train_loader = DataLoader(dataset=train_set, num_workers=config.threads, batch_size=config.trainBatchSize, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=config.threads, batch_size=config.testBatchSize, shuffle=False)

    net = model.model()
    loss = torch.nn.MSELoss()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        net = net.cuda()
        loss = loss.cuda()

    optimizer = optim.Adam(net.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[500], gamma=config.lr_decay_rate)

    best_epoch = 0
    best_SAM = 1

    for i in range(config.epoch):
        net.train()
        epoch_gtfuse_loss = dataloader.AverageMeter()

        for batch_idx, (gtBatch, msBatch, panBatch) in enumerate(train_loader):

            if torch.cuda.is_available():
                msBatch = torch.nn.functional.interpolate(msBatch, size=(256, 256), mode='bilinear')
                msBatch, panBatch, gtBatch = msBatch.cuda(), panBatch.cuda(), gtBatch.cuda()
                msBatch = Variable(msBatch.to(torch.float32))
                panBatch = Variable(panBatch.to(torch.float32))
                gtBatch = Variable(gtBatch.to(torch.float32))
            N = len(train_loader)

            gt_fuse = net(msBatch, panBatch)
            optimizer.zero_grad()
            outLoss = loss(gtBatch, gt_fuse)
            outLoss.backward(retain_graph=True)

            optimizer.step()
            epoch_gtfuse_loss.update(outLoss.item(), msBatch.shape[0])

            if (batch_idx + 1) % 100 == 0:
                training_state = '  '.join(
                    ['Epoch: {}', '[{} / {}]', 'outLoss: {:.6f}']
                )
                training_state = training_state.format(
                    i, batch_idx, N, outLoss
                )
                print(training_state)

        net.eval()
        all = dataloader.AverageMeter()

        with torch.no_grad():
            for j, (gt_val, ms_val, pan_val) in enumerate(val_loader):
                if torch.cuda.is_available():
                    ms_val = torch.nn.functional.interpolate(ms_val, size=(256, 256), mode='bilinear')
                    ms_val, pan_val, gt_val = ms_val.cuda(), pan_val.cuda(), gt_val.cuda()
                    ms_val = Variable(ms_val.to(torch.float32))
                    pan_val = Variable(pan_val.to(torch.float32))
                    gt_val = Variable(gt_val.to(torch.float32))

                mp = net(ms_val, pan_val)
                test_loss = loss(gt_val, mp)
                all.update(test_loss.item(), ms_val.shape[0])

            print('eval allloss: {:.6f}'.format(all.avg))

        if all.avg < best_SAM:
            best_epoch = i
            best_SAM = all.avg
            best_weights = copy.deepcopy(net.state_dict())
            torch.save(net.state_dict(), os.path.join(config.outputs_dir, 'epoch-best-{}.pth'.format(i)))
        print('best epoch:{:.0f}'.format(best_epoch))

        scheduler.step()

    print('best epoch: {}, epoch_SAM: {:.6f}'.format(best_epoch, best_SAM))
    torch.save(best_weights, os.path.join(config.outputs_dir, 'best.pth'))

if __name__ == "__main__":
    main()