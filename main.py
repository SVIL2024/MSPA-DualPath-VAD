import torch
from data import *
from torch.autograd import Variable
import scipy.stats
from utils import *
from reconstruction_model import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from ObjectMask import *
import copy
import os
import glob
import sys
import warnings
from contextlib import contextmanager

torch.autograd.set_detect_anomaly(True)

# 设置随机种子函数
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
@contextmanager
def suppress_output():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        null_device = open('nul' if os.name == 'nt' else '/dev/null', 'w')
        sys.stdout = null_device
        sys.stderr = null_device
        yield
    finally:
        null_device.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

def train_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release")
    channel_in = 1
    if args.dataset_type == 'ped2':
        channel_in = 1
        train_folder = os.path.join('UCSDped2', 'Train')
        test_folder = os.path.join('UCSDped2', 'Test')
        learning_rate = args.learning_rate_ped2
    if args.dataset_type == 'avenue':
        channel_in = 3
        train_folder = os.path.join('Avenue', 'Train')
        test_folder = os.path.join('Avenue', 'Test')
        learning_rate = args.learning_rate_avenue
    if args.dataset_type == 'shanghai':
        channel_in = 3
        learning_rate = 1e-4
        train_folder = os.path.join('shanghaitech', 'training', 'frames')
        test_folder = os.path.join('shanghaitech', 'testing', 'frames')
        args.epochs = 10
    img_extension = '.tif' if args.dataset_type == 'ped2' else '.jpg'
    exp_dir = args.exp_dir + '_lr' + str(learning_rate) + '_bs' + str(args.batch_size)
    log_dir = os.path.join('./', args.path + str(args.path_num), args.dataset_type, exp_dir)
    print(log_dir)

    log_dir_writer = log_dir + "/" + "writer" + "/"
    writer = SummaryWriter(log_dir=log_dir_writer)
    if not os.path.exists(log_dir_writer):
        os.makedirs(log_dir_writer)

    # init model
    if args.start_epoch < args.epochs:
        En = Encoder(num_in_ch=channel_in, features_root=32)
        Den = Decoder(features_root=32, num_out_ch=channel_in, skip_ops=args.skip_ops)
        Dep = Decoder0(features_root=32, num_out_ch=channel_in, skip_ops=args.skip_ops)
        mem = Mem(2000, 256)
        mem_p = Mem(2000, 256)
        Dep.load_state_dict(Den.state_dict())
        En = nn.DataParallel(En).cuda()
        mem = nn.DataParallel(mem).cuda()
        mem_p = nn.DataParallel(mem_p).cuda()
        Den = nn.DataParallel(Den).cuda()
        Dep = nn.DataParallel(Dep).cuda()

        out_p3 = torch.zeros(1, 1).cuda()
        out_n3 = torch.zeros(1, 1).cuda()
        kl_out3 = torch.zeros(1, 1).cuda()
        kl_fea = torch.zeros(1, 1).cuda()
        mem_out_p = {"x3_out": torch.zeros(1, 1).cuda(), "att_weight": torch.zeros(1, 1).cuda()}
        mem_out_n = {"x3_out": torch.zeros(1, 1).cuda(), "att_weight": torch.zeros(1, 1).cuda()}
        kl_label = 0

        # init dataloader
        trans_compose = transforms.Compose([transforms.ToTensor()])
        train_dataset = Reconstruction3DDataLoader(train_folder, trans_compose,
                                                   resize_height=args.h, resize_width=args.w,
                                                   dataset=args.dataset_type,
                                                   img_extension=img_extension)
        train_dataset_jump = Reconstruction3DDataLoaderJump(train_folder, transforms.Compose([transforms.ToTensor()]),
                                                            resize_height=args.h, resize_width=args.w,
                                                            dataset=args.dataset_type, jump=args.jump,
                                                            img_extension=img_extension)

        train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.num_workers, drop_last=True)

        train_batch_jump = data.DataLoader(train_dataset_jump, batch_size=args.batch_size,
                                           shuffle=True, num_workers=args.num_workers, drop_last=True)

        test_dataset = Reconstruction3DDataLoader(test_folder, trans_compose,
                                                  resize_height=args.h, resize_width=args.w, dataset=args.dataset_type,
                                                  img_extension=img_extension, train=False)

        test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                     shuffle=False, num_workers=args.num_workers_test, drop_last=False)

        params_En = list(En.parameters())
        params_Den = list(Den.parameters())
        params_Dep = list(Dep.parameters())
        params_mem = list(mem.parameters())
        params_mem_p = list(mem_p.parameters())
        params = params_En + params_Den + params_Dep + params_mem + params_mem_p
        optimizer = torch.optim.Adam(params, lr=learning_rate)

        loss_func_mse = nn.MSELoss(reduction='none')
        auroc_img_best=0
        img_step=0

        sigma = args.sigma_noise ** 2
        with suppress_output():
            yolo_model, cifar_loader, cifar_iter = init_dependencies(
                cifar_path='./dataset/cifar100',
                yolo_cfg='Yolov3/yolov3/cfg/yolov3-spp.cfg',
                yolo_weights='Yolov3/yolov3/weights/yolov3-spp-ultralytics.pt',
                device=device,
                channel_in=channel_in
            )
        tic = time.time()
        for step in tqdm(range(args.start_epoch, args.epochs), ascii=True):
            tic_stage = time.time()
            En.train()
            Den.train()
            Dep.train()
            mem_p.train()
            mem.train()

            print('Training is start..')
            for j, (imgs, imgsjump) in enumerate(zip(train_batch, train_batch_jump)):
                net_in = copy.deepcopy(imgs)
                net_in = net_in.cuda()
                net_in_jump_orig = imgsjump.to(device)

                anomaly_gaus = gaussian(net_in_jump_orig, 1, 0, sigma)
                anomaly_jump = net_in_jump_orig
                anomaly_shuffle = genMotionAnoSmps(net_in)
                with suppress_output():
                    with torch.no_grad():
                        anomaly_ObjectMask, cifar_iter = generate_pseudo_anomalies(
                            net_in=net_in,
                            yolo_model=yolo_model,
                            cifar_loader=cifar_loader,
                            cifar_iter=cifar_iter,
                            device=device,
                            channel_in=channel_in
                        )

                anomaly_candidates = [anomaly_gaus, anomaly_jump, anomaly_ObjectMask, anomaly_shuffle]
                net_in_pseudo = random.choice(anomaly_candidates)

                pseudo_stat = []
                for b in range(args.batch_size):
                    total_pseudo_prob = 0
                    rand_number = np.random.rand()

                    pseudo_anomaly = total_pseudo_prob <= rand_number < total_pseudo_prob + args.pseudo_anomaly
                    total_pseudo_prob += args.pseudo_anomaly
                    if pseudo_anomaly:
                        net_in[b] = net_in_pseudo[b][0]
                        pseudo_stat.append(True)
                    else:
                        pseudo_stat.append(False)

                # train
                for b in range(args.batch_size):
                    if pseudo_stat[b]:  # 异常
                        fea_p, x2, x1, x0 = En(net_in)
                        mem_out_p = mem_p(fea_p)
                        att_weight_p = mem_out_p["att_weight"]
                        out, out_p1, out_p2, out_p3 = Dep(mem_out_p["x3_out"],  x2, x1, x0)
                        kl_label = 1
                    else:  # 正常
                        fea_n,  x2, x1, x0 = En(net_in)
                        mem_out_n = mem(fea_n)
                        att_weight_n = mem_out_n["att_weight"]
                        out, out_n1, out_n2, out_n3 = Den(mem_out_n["x3_out"], x2, x1, x0)

                loss_mse = loss_func_mse(out, net_in)

                loss_feas = -1.0 * torch.abs(mem_out_p["x3_out"].detach().cuda() - mem_out_n["x3_out"].detach().cuda())

                modified_loss_mse = []
                for b in range(args.batch_size):
                    if  pseudo_stat[b]:
                        modified_loss_mse.append(torch.mean(-loss_mse[b]))
                        loss_sparsity_p = torch.mean(torch.sum(-att_weight_p.detach() * torch.log(att_weight_p.detach() + 1e-12), dim=1))
                        kl_fea = -1.0 * KL_divergence(mem_out_n["x3_out"].detach().cpu(), mem_out_p["x3_out"].detach().cpu())
                        kl_fea = torch.tensor(kl_fea).cuda()
                        kl_out3 = -1.0 * KL_divergence(out_n3.detach().cpu(), out_p3.detach().cpu())  # Skip Connection
                        kl_out3 = torch.tensor(kl_out3).cuda()
                        kl_label = 1

                    else:  # no pseudo anomaly
                        modified_loss_mse.append(torch.mean(loss_mse[b]))
                        loss_sparsity_n = torch.mean(torch.sum(-att_weight_n.detach() * torch.log(att_weight_n.detach() + 1e-12), dim=1))

                assert len(modified_loss_mse) == loss_mse.size(0)
                stacked_loss_mse = torch.stack(modified_loss_mse)

                if kl_label:
                    loss = torch.mean(stacked_loss_mse)\
                           + (loss_feas.sum()) * 0.0002 \
                           + kl_fea * 0.0002 \
                           + kl_out3.sum() * 0.0002 \
                           + loss_sparsity_p * args.loss_m_weight \

                else:
                    loss = torch.mean(stacked_loss_mse)\
                           + (loss_feas.sum()) * 0.0002 \
                           + loss_sparsity_n * args.loss_m_weight

                optimizer.zero_grad()
                loss.sum().backward(retain_graph=True)
                optimizer.step()

            model_dict = {
                    'En': En,
                    'mem': mem,
                    'mem_p': mem_p,
                    'Den': Den,
                    'Dep': Dep
            }
            torch.save(model_dict, os.path.join(log_dir, 'model_{:02d}.pth'.format(step)))

            train_loss = 0
            train_loss += loss.mean().item()
            writer.add_scalar("train_loss", train_loss, int(step))

            # test
            print(f"Evaluating model from epoch {step}...")
            labels = np.load('./frame_labels_' + args.dataset_type + '.npy', allow_pickle=True)
            videos = OrderedDict()
            videos_list = sorted(glob.glob(os.path.join(test_folder, '*/')))
            for video in videos_list:
                    video_name = video.split('\\')[-2]
                    videos[video_name] = {}
                    videos[video_name]['path'] = video
                    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + img_extension))
                    videos[video_name]['frame'].sort()
                    videos[video_name]['length'] = len(videos[video_name]['frame'])

            labels_list = []
            label_length = 0
            psnr_list = {}

            for video in sorted(videos_list):
                video_name = video.split('\\')[-2]
                labels_list = np.append(labels_list,
                                                labels[0][8 + label_length:videos[video_name]['length'] + label_length - 7])
                label_length += videos[video_name]['length']
                psnr_list[video_name] = []

            label_length = 0
            video_num = 0
            label_length += videos[videos_list[video_num].split('\\')[-2]]['length']

            En.eval()
            Den.eval()
            with torch.no_grad():
                for k, (imgs) in enumerate(test_batch):
                    if k == label_length - 15 * (video_num + 1):
                        video_num += 1
                        label_length += videos[videos_list[video_num].split('\\')[-2]]['length']

                    imgs = Variable(imgs).cuda()
                    fea, x2, x1, x0 = En(imgs)
                    fea_mem = mem(fea)
                    out, out1, out2, out3 = Den(fea_mem["x3_out"], x2, x1, x0)
                    loss_mse = loss_func_mse(out[0, :, 8], imgs[0, :, 8])
                    loss_pixel = torch.mean(loss_mse)
                    mse_imgs = loss_pixel.item()
                    psnr_list[videos_list[video_num].split('\\')[-2]].append(psnr(mse_imgs))

            anomaly_score_total_list = []
            for vi, video in enumerate(sorted(videos_list)):
                video_name = video.split('\\')[-2]
                score = anomaly_score_list(psnr_list[video_name])

                anomaly_score_total_list += score
            anomaly_score_total_list = np.asarray(anomaly_score_total_list)
            accuracy = AUC(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))

            # 记录日志
            toc_stage = time.time()
            write2txt(log_dir,
                      f'model_{step}_AUC: {accuracy * 100:.2f}% | train loss: {train_loss:.4f} | ' f'test loss: {mse_imgs:.4f} | psnr: {psnr(mse_imgs):.4f}  |  time: {(toc_stage - tic_stage) / 60:.2f} min')
            writer.add_scalar("auroc_img", accuracy, step)
            writer.add_scalar("test_loss", mse_imgs, step)

            if accuracy > auroc_img_best or step == 0:
                auroc_img_best = accuracy
                img_step = int(step)
                best_En_weights = En.state_dict()
                best_Den_weights = Den.state_dict()
                best_optimizer = optimizer.state_dict()

                best_ckp_path = str(log_dir + "/epoch" + str(img_step) + ".pth")

        toc = time.time()
        print('total time:' + str((toc - tic) / 3600) + "h")
        print('mean time:' + str((toc - tic) / 60 / (args.epochs - args.start_epoch)) + "min")
        print('best_ckp_path: ' + best_ckp_path)

        torch.save({'En': best_En_weights, 'Den': best_Den_weights, 'epoch': img_step, 'optimizer': best_optimizer}, best_ckp_path)
        write2txt(log_dir, '-----------------------------------------')
        write2txt(log_dir, f'best_model_{str(img_step)}_AUC: ' + ' ' + f'{auroc_img_best * 100}%' + ' ' + f'total time: {(toc - tic) / 60} min')
        return auroc_img_best, img_step


def write2txt(filename, content):
    f = open(os.path.join(filename, 'log.txt'), 'a')
    f.write(str(content) + "\n")
    f.close()


parser = argparse.ArgumentParser(description="VAD")
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs for training')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--learning_rate_ped2', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--learning_rate_avenue', default=0.0000001, type=float, help='initial learning_rate')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=0, help='number of workers for the test loader')
parser.add_argument('--loss_m_weight', help='loss_m_weight', type=float, default=0.0002)

parser.add_argument('--dataset_type', type=str, default='ped2', choices=['ped2', 'avenue', 'shanghai'],
                    help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--path', type=str, default='exp_log', help='directory of data')
parser.add_argument('--path_num', type=int, default=11, help='number of path')
parser.add_argument('--mem_dim', type=int, default=2000, help='size of mem')
parser.add_argument('--sigma_noise', default=0.9, type=float, help='sigma of noise added to the iamges')
parser.add_argument('--exp_dir', type=str, default='log_double', help='basename of folder to save weights')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                    help='adam or sgd with momentum and cosine annealing lr')
parser.add_argument('--mem_usage', default=[False, False, False, True], type=str)
parser.add_argument('--skip_ops', default=["none", "none", "concat"], type=str)
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch. usually number in filename + 1')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

# Device options
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--pseudo_anomaly', type=float, default=0.01,
                    help='pseudo anomaly jump frame (skip frame) probability. 0 no pseudo anomaly')
parser.add_argument('--jump', nargs='+', type=int, default=[2],
                    help='Jump for pseudo anomaly (hyperparameter s)')
parser.add_argument('--model_dir', type=str, default='exp_log11/ped2/log_double_lr0.0001_bs2',
                    help='directory of model')
parser.add_argument('--model_start', type=int, default=0, help='Start model index')
parser.add_argument('--model_end', type=int, default=9, help='End model index ')
parser.add_argument('--print_all', action='store_true', help='print all reconstruction loss')
args = parser.parse_args()

if __name__ == '__main__':
    setup_seed(42)
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    auroc_img_best, img_step = train_test(args)
    print("auroc_img:", str(auroc_img_best), "epoch:", str(img_step))