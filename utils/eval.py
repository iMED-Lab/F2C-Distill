import os
import torch


def save_model(args, epoch, model, optimizer, if_best=True):
    output_dir = args.output_dir
    epoch_name = str(epoch)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # 取出原模型
    if if_best:
        checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
    else:
        checkpoint_path = os.path.join(output_dir, 'checkpoint_last.pth')
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer,
        'epoch': epoch,
        'args': args,
    }
    torch.save(to_save, checkpoint_path)
    

def load_model(args, checkpoint, device, if_best=True, weights_only=False):
    output_dir = checkpoint
    if if_best:
        print('Loading best model from %s' % checkpoint)
        checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
    else:
        checkpoint_path = os.path.join(output_dir, 'checkpoint_last.pth')
    if args.device == 'cuda':
        model_state_dict = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)['model']
    else:
        model_state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))['model']
    return model_state_dict
