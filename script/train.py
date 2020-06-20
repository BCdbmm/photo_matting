# _*_ coding: utf-8 _*_
# @Time : 2020/6/20 下午8:23
# @Author : sunyulong
# @File : train.py

from tensorboardX import SummaryWriter
import argparse
import json
import os
import torch
from models import model
from data import datasets

parser=argparse.ArgumentParser('训练模型')

parser.add_argument('--gpu_indices', default=[0], type=int, nargs='+',
                    help='The indices of gpus to be used.')
parser.add_argument('--num_epochs', default=300, type=int,
                    help='Number of epochs')
parser.add_argument('--batch_size_per_gpu', default=1, type=int,
                    help='Batch size of one gpu.')
parser.add_argument('--learning_rate', default=1e-4, type=float,
                    help='Initial learning rate.')
parser.add_argument('--end_learning_rate', default=1e-6, type=float,
                    help='End learning rate.')
parser.add_argument('--decay_epochs', default=20, type=int,
                    help='Decay learning rate every decay_step.')
parser.add_argument('--lr_decay_factor', default=0.9, type=float,
                    help='Learning rate decay factor.')
parser.add_argument('--annotation_path', default='../data/train.txt', type=str,
                    help='Path to the annotation file.')
parser.add_argument('--root_dir', default=None, type=str,
                    help='Path to the images folder: xxx/Matting_Human_Half.')
parser.add_argument('--model_dir', default='../check_point_dir', type=str,
                    help='Where the trained model file is stored.')

FLAGS = parser.parse_args()

def config_lr(optim,decay=0.9):
    lr=FLAGS.learning_rate*decay
    if lr<FLAGS.end_learning_rate:
        return FLAGS.end_learning_rate
    for param_group in optim.param_groups:
        param_group['lr']=lr
    return lr

def train():
    gpu_indices = FLAGS.gpu_indices
    num_epochs = FLAGS.num_epochs
    learning_rate = FLAGS.learning_rate
    lr_decay = FLAGS.lr_decay_factor
    batch_size = FLAGS.batch_size_per_gpu * len(gpu_indices)
    num_steps_to_save_checkpoint = 128000 // batch_size
    annotation_path = FLAGS.annotation_path
    root_dir = FLAGS.root_dir
    model_dir = FLAGS.model_dir
    gpu_ids_str = ','.join([str(index) for index in gpu_indices])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str

    # Device configuration
    cuda_ = 'cuda:{}'.format(gpu_indices[0])
    device = torch.device(cuda_ if torch.cuda.is_available() else 'cpu')
    matting_dataset = datasets.MattingDataset(annotation_path=annotation_path,
                                             root_dir=root_dir)
    train_loader = torch.utils.data.DataLoader(matting_dataset,
                                               batch_size=batch_size,
                                               num_workers=2,
                                               shuffle=True,
                                               drop_last=True)
    fte=model.featureExtract(model.VGG16_BN_CONFIGS.get('13conv')).to(device)
    mnet=model.Mnet(fte).to(device)

    start_epoch, start_step = 0, 0
    last_ckpt_path= None
    json_path = os.path.join(model_dir, 'checkpoint.json')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        if os.path.exists((json_path)):
            with open(json_path,'r')as reader:
                ckpt_dict=json.load(reader)
                start_epoch=ckpt_dict.get('epoch',0)+1
                start_step=ckpt_dict.get('step',0)+1
            mnet_name='model-{}-{}.ckpt'.format(start_epoch,start_step)
            if os.path.exists(os.path.join(model_dir,mnet_name)):
                last_ckpt_path=os.path.join(model_dir,mnet_name)
            if os.path.exists(os.path.join(model_dir,'model.ckpt')):
                last_ckpt_path=os.path.join(model_dir,'model.ckpt')
    if last_ckpt_path and os.path.exists(last_ckpt_path):
        pre_trained_params=torch.load(last_ckpt_path).items()
        state_dict={k.replace('module.',''):v for k,v in pre_trained_params}
        mnet.load_state_dict(state_dict)
        print('load pretrained parameters, Done')
    mnet=torch.nn.DataParallel(mnet,device_ids=gpu_indices)
    optim=torch.optim.Adam(mnet.parameters(),lr=learning_rate)
    log_dir=os.path.join(model_dir,'logs')
    log=SummaryWriter(log_dir=log_dir)
    total_step=len(train_loader)
    for epoch in range(start_epoch,num_epochs):
        for i,(images,alphas,alphas_noise,masks) in enumerate(train_loader):
            images = images.to(device)
            alphas = alphas.to(device)
            alphas_noise = alphas_noise.to(device)
            masks = masks.to(device)
            outputs=mnet(images)
            # print(outputs)
            loss=model.loss(outputs,alphas,masks)
            optim.zero_grad()
            loss.backward()
            optim.step()
            step=i+epoch*total_step
            if (i+1)%50==0:
                # print(loss.item())
                log.add_scalar('loss',loss.item(),step+1)
                info={'alphas':
                            alphas.cpu().numpy()[:2],
                        'alphas_noise':
                            alphas_noise.cpu().numpy()[:2],
                        'alphas_pred':
                            outputs.data.cpu().numpy()[:2]}
                for tag,imgs in info.items():
                    log.add_images(tag,imgs,step+1,dataformats='NCHW')

            if (step+1)%num_steps_to_save_checkpoint==0:
                model_name='model-{}-{}.ckpt'.format(epoch+1,i+1)
                model_path=os.path.join(model_dir,model_name)
                torch.save(mnet.state_dict(),model_path)
                ckpt_dict={'epoch':epoch,'step':i,'global_step':step}
                with open(json_path,'w')as writer:
                    json.dump(ckpt_dict,writer)
        if epoch%FLAGS.decay_epochs==0:
            num_decays=epoch//FLAGS.decay_epochs
            lr=config_lr(optim,decay=lr_decay**num_decays)
            log.add_scalar('learning_rate',lr,step+1)
        log.close()
        model_path=os.path.join(model_dir,'model.ckpt')
        torch.save(mnet.state_dict(),model_path)
        ckpt_dict={'epoch':num_epochs-1,'step':total_step-1,'global_step':num_epochs*total_step-1}
        with open(json_path,'w')as writer:
            json.dump(ckpt_dict,writer)
if __name__=='__main__':
    train()