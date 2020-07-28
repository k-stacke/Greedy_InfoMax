import torch
# import torchvision.transforms as transforms
import time
import numpy as np

#### own modules
from GreedyInfoMax.utils import logger
from GreedyInfoMax.vision.arg_parser import arg_parser
from GreedyInfoMax.vision.models import load_vision_model
from GreedyInfoMax.vision.data import get_dataloader

import neptune

torch.backends.cudnn.benchmark=True

def validate(opt, model, test_loader):
    model.eval()
    total_step = len(test_loader)
    #total_step = int(test_loader._size/opt.batch_size_multiGPU)

    loss_epoch = [0 for i in range(opt.model_splits)]
    starttime = time.time()

    for step, batch_data in enumerate(test_loader):
        print("\r Validation Step: {}/{}".format(step, total_step), end="")

        img = batch_data[0]
        label = batch_data[1]

        model_input = img.to(opt.device)
        label = label.to(opt.device)

        with torch.no_grad():
            loss, _, _, _ = model(model_input, label, n=opt.train_module)
        loss = torch.mean(loss, 0)

        loss_epoch += loss.data.cpu().numpy()

    for i in range(opt.model_splits):
        print(
            "Validation Loss Model {}: Time (s): {:.1f} --- {:.4f}".format(
                i, time.time() - starttime, loss_epoch[i] / total_step
            )
        )

    validation_loss = [x/total_step for x in loss_epoch]

    return validation_loss


def train(opt, model, exp):
    total_step = len(train_loader)
    model.module.switch_calc_loss(True)

    print_idx = 100

    starttime = time.time()
    cur_train_module = opt.train_module

    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch):
        model.train()

        loss_epoch = [0 for i in range(opt.model_splits)]
        loss_updates = [1 for i in range(opt.model_splits)]


        for step, batch_data in enumerate(train_loader):
            if step % print_idx == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Training Block: {}, Time (s): {:.1f}".format(
                        epoch + 1,
                        opt.num_epochs + opt.start_epoch,
                        step,
                        total_step,
                        cur_train_module,
                        time.time() - starttime,
                    )
                )

            starttime = time.time()

            img = batch_data[0]
            label = batch_data[1]

            model_input = img.to(opt.device)
            label = label.to(opt.device)#.float()

            loss, _, _, accuracy = model(model_input, label, n=cur_train_module)
            loss = torch.mean(loss, 0) # take mean over outputs of different GPUs
            accuracy = torch.mean(accuracy, 0)

            if cur_train_module != opt.model_splits and opt.model_splits > 1:
                loss = loss[cur_train_module].unsqueeze(0)


            # loop through the losses of the modules and do gradient descent
            for idx, cur_losses in enumerate(loss):
                if len(loss) == 1 and opt.model_splits != 1:
                    idx = cur_train_module

                model.zero_grad()

                if idx == len(loss) - 1:
                    cur_losses.backward()
                else:
                    cur_losses.backward(retain_graph=True)
                optimizer[idx].step()

                print_loss = cur_losses.item()
                if opt.infoloss_acc:
                    print_acc = [a.item() for a in accuracy[idx, :]]
                else:
                    print_acc = accuracy[idx].item()
                if step % print_idx == 0:
                    print("\t \t Loss: \t \t {:.4f}".format(print_loss))
                    if opt.loss == 1 or opt.infoloss_acc:
                        print(f"\t \t Accuracy: {print_acc}")
                        #print("\t \t Accuracy: \t \t {:.4f}".format(print_acc))

                loss_epoch[idx] += print_loss
                loss_updates[idx] += 1

                exp.send_metric(f'loss_{idx}', print_loss)
                if opt.infoloss_acc:
                    for i, acc in enumerate(print_acc):
                        exp.send_metric(f'infoloss_acc_{idx}_{i}', acc)


        if opt.validate:
            validation_loss = validate(opt, model, test_loader) #test_loader corresponds to validation set here
            for idx, val_loss in enumerate(validation_loss):
                exp.send_metric(f'val_loss_{idx}', val_loss)
            logs.append_val_loss(validation_loss)

        logs.append_train_loss([x / loss_updates[idx] for idx, x in enumerate(loss_epoch)])
        logs.create_log(model, epoch=epoch, optimizer=optimizer)


if __name__ == "__main__":

    neptune.init('k-stacke/cpc-greedyinfomax')

    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt)
    opt.training_dataset = "unlabeled"

    exp = neptune.create_experiment(name='CPC Cam17 unbiased', params=opt.__dict__,
            tags=['cpc'])

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    if opt.device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    # load model
    model, optimizer = load_vision_model.load_model_and_optimizer(opt)

    logs = logger.Logger(opt)

    train_loader, _, supervised_loader, _, test_loader, _ = get_dataloader.get_dataloader(
        opt
    )

    if opt.loss == 1:
        train_loader = supervised_loader

    try:
        # Train the model
        train(opt, model, exp)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")

    logs.create_log(model)
    exp.stop()
