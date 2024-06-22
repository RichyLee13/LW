import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument("--model_names", default=['ABCNet'], nargs='+',
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")
parser.add_argument("--pth_dirs", default=['Dataset_mask/ABCNet_400.pth.tar'], nargs='+',  help="checkpoint dir, default=None or ['NUDT-SIRST/ACM_400.pth.tar','NUAA-SIRST/ACM_400.pth.tar']")
parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default=['Dataset_mask'], nargs='+',
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                    help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                    help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./log_615/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)

global opt
opt = parser.parse_args()
## Set img_norm_cfg
if opt.img_norm_cfg_mean != None and opt.img_norm_cfg_std != None:
  opt.img_norm_cfg = dict()
  opt.img_norm_cfg['mean'] = opt.img_norm_cfg_mean
  opt.img_norm_cfg['std'] = opt.img_norm_cfg_std

def img2patchs(img, patch_size=(256, 256), overlap_size=(20, 20)):
    """
    将图像分割成多个补丁，并返回这些补丁以及它们在原始图像中的位置。

    参数:
    img: 输入图像，Numpy数组。
    patch_size: 每个补丁的大小，形如(width, height)的元组。
    overlap_size: 补丁之间的重叠大小，形如(width, height)的元组。

    返回值:
    patchs_all: 包含所有补丁的列表。
    target_size: 原始图像的大小，形如(height, width)的元组。
    remain_size: 最后一行和最后一列的剩余大小，形如(height, width)的元组。
    """
    # 获取图像尺寸
    h, w, c = img.shape
    ph, pw = patch_size
    oh, ow = overlap_size

    # 计算调整后的目标大小，以确保补丁可以完全覆盖图像
    r_h = (h - ph) % (ph - oh)
    r_w = (w - pw) % (pw - ow)

    target_w, target_h = w, h

    # 如果图像大小小于补丁大小减去重叠大小，则直接返回整个图像
    if not (h >= ph > oh and w >= pw > ow):
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
        img = img.squeeze(0).permute(1, 2, 0)
        return [[img]], (target_h, target_w), (0, 0)

    # 计算需要分割的行数和列数
    N = math.ceil((target_h - ph) / (ph - oh)) + 1
    M = math.ceil((target_w - pw) / (pw - ow)) + 1

    # 存储所有补丁的列表
    patchs_all = []
    for n in range(N):
        patchs_row = []
        for m in range(M):

            # 根据当前行和列计算补丁的起始位置
            if n == N - 1:
                ph_start = target_h - ph
            else:
                ph_start = n * (ph - oh)

            if m == M - 1:
                pw_start = target_w - pw
            else:
                pw_start = m * (pw - ow)
            patch = img[ph_start:(ph_start + ph), pw_start:(pw_start + pw), :]
            patchs_row.append(patch)
        patchs_all.append(patchs_row)

    return patchs_all, (target_h, target_w), (r_h, r_w)

def check_type(var):
    if isinstance(var, np.ndarray):
        return "变量是 NumPy 数组"
    elif isinstance(var, torch.Tensor):
        return "变量是 PyTorch 张量"
    else:
        return "变量既不是 NumPy 数组，也不是 PyTorch 张量"
def patchs2img(patchs, target_h,target_w,r_size, overlap_size=(20, 20)):
    """
    将补丁重新拼接成原始图像。

    参数:
    patchs: 补丁列表，每个补丁是一个Numpy数组。
    r_size: 原始图像的剩余大小，形如(height, width)的元组。
    overlap_size: 补丁之间的重叠大小，形如(width, height)的元组。

    返回值:
    拼接后的图像，Numpy数组。
    """
    N = len(patchs)
    M = len(patchs[0])

    oh, ow = overlap_size

    patch_shape = patchs[0][0].shape
    ph, pw = patch_shape[:2]
    r_h, r_w = r_size

    c = 1
    # 如果只有一个补丁，则直接返回该补丁
    if N == 1 and M == 1:
        return_img = patchs[0][0]
        return_img = return_img.permute(2, 0, 1).unsqueeze(0)
        return_img = F.interpolate(return_img, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return_img = return_img.squeeze(0).permute(1, 2, 0)
        return return_img
    row_imgs = []
    for n in range(N):
        row_img = patchs[n][0]
        for m in range(1, M):
            # 根据当前列计算重叠宽度
            if m == M - 1 and r_w != 0:
                ow_new = pw - r_w
            else:
                ow_new = ow
            patch = patchs[n][m]
            h, w = row_img.shape[:2]
            new_w = w + pw - ow_new
            big_row_img = np.zeros((h, new_w, c), dtype=np.float32)
            # 将 GPU Tensor 转换为 numpy 数组之前，先转移到 CPU 并使用 .detach()
            if isinstance(row_img, torch.Tensor):
                big_row_img[:, :w - ow_new, :] = row_img[:, :w - ow_new, :].detach().cpu().numpy()
            else:
                big_row_img[:, :w - ow_new, :] = row_img[:, :w - ow_new, :]
            big_row_img[:, w:, :] = patch[:, ow_new:, :].detach().cpu().numpy()
            # 处理重叠区域
            if isinstance(row_img, torch.Tensor):
                overlap_row_01 = row_img[:, w - ow_new:, :].detach().cpu().numpy()
            else:
                overlap_row_01 = row_img[:, w - ow_new:, :]
            overlap_row_02 = patch[:, :ow_new, :].detach().cpu().numpy()

            # 计算重叠区域的权重
            weight = 0.5
            overlap_row = (overlap_row_01 * (1 - weight))+ (overlap_row_02 * weight)
            big_row_img[:, w - ow_new:w, :] = overlap_row
            row_img = big_row_img

        row_imgs.append(row_img)

    column_img = row_imgs[0]
    for i in range(1, N):
        # 根据当前行计算重叠高度
        if i == N - 1 and r_h != 0:
            oh_new = ph - r_h
        else:
            oh_new = oh

        row_img = row_imgs[i]
        h, w = column_img.shape[:2]
        new_h = h + ph - oh_new

        big_column_img = np.zeros((new_h, w, c), dtype=np.float32)
        big_column_img[:h - oh_new, :, :] = column_img[:h - oh_new, :, :]
        big_column_img[h:, :, :] = row_img[oh_new:, :, :]
        # 处理重叠区域
        overlap_column_01 = column_img[h - oh_new:, :, :]
        overlap_column_02 = row_img[:oh_new, :, :]

        # 计算重叠区域的权重
        weight = 0.5
        overlap_column = (overlap_column_01 * (1 - weight))+ (overlap_column_02 * weight)
        big_column_img[h - oh_new:h, :, :] = overlap_column

        column_img = big_column_img

    return column_img


def test(): 
    test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    net = Net(model_name=opt.model_name, mode='test').cuda()
    try:
        net.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    except:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.load_state_dict(torch.load(opt.pth_dir, map_location=device)['state_dict'])
    net.eval()
    
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    with torch.no_grad():
        for idx_iter, (img, gt_mask, size, img_dir) in enumerate(test_loader):
            img = Variable(img).cuda()
            # pred = net.forward(img)
            img = img.squeeze(0)
            # (channels, height, width)——>h, w, c
            img = img.permute(1, 2, 0)
            patchs, target_size, r_size = img2patchs(img)
            pre_patchs = []
            for i in range(len(patchs)):
                # print('i',i)
                pre_patch = []
                for j in range(len(patchs[i])):
                    patchs[i][j] = patchs[i][j].permute(2, 0, 1)
                    tensor = torch.unsqueeze(patchs[i][j], 0).cuda()
                    # print(img.shape)
                    pre = net.forward(tensor)
                    pre_np = pre.squeeze(0)
                    pre_np = pre_np.permute(1, 2, 0)
                    pre_patch.append(pre_np)

                pre_patchs.append(pre_patch)

            pred = patchs2img(pre_patchs, target_size[0], target_size[1], r_size)
            # print("2",pred.shape)
            pred = torch.tensor(pred).permute(2, 0, 1)
            pred = pred.unsqueeze(0).cuda()

            pred = pred[:,:,:size[0],:size[1]]
            gt_mask = gt_mask[:,:,:size[0],:size[1]]
            eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
            eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)   
            
            ### save img
            if opt.save_img == True:
                img_save = transforms.ToPILImage()((pred[0,0,:,:]).cpu())
                if not os.path.exists(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name):
                    os.makedirs(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name)
                img_save.save(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + img_dir[0] + '.png')  
    
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')

if __name__ == '__main__':
    opt.f = open(opt.save_log + 'test_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
    if opt.pth_dirs == None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            print(opt.model_name)
            opt.f.write(opt.model_name + '_400.pth.tar' + '\n')
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(dataset_name)
                opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = opt.save_log + opt.dataset_name + '/' + opt.model_name + '_400.pth.tar'
                test()
            print('\n')
            opt.f.write('\n')
        opt.f.close()
    else:
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    if dataset_name in pth_dir and model_name in pth_dir:
                        opt.test_dataset_name = dataset_name
                        opt.model_name = model_name
                        opt.train_dataset_name = pth_dir.split('/')[0]
                        print(pth_dir)
                        opt.f.write(pth_dir)
                        print(opt.test_dataset_name)
                        opt.f.write(opt.test_dataset_name + '\n')
                        opt.pth_dir = opt.save_log + pth_dir
                        test()
                        print('\n')
                        opt.f.write('\n')
        opt.f.close()
        
