import torch
import time
import numpy as np

#### own modules
from GreedyInfoMax.utils import logger
from GreedyInfoMax.vision.arg_parser import arg_parser
from GreedyInfoMax.vision.models import load_vision_model
from GreedyInfoMax.vision.data import get_dataloader


def validate(opt, model, test_loader):
    total_step = len(test_loader)
    #total_step = int(test_loader._size/opt.batch_size_multiGPU)

    loss_epoch = [0 for i in range(opt.model_splits)]
    starttime = time.time()

    for step, batch_data in enumerate(test_loader):
        print("\r Validation Step: {}/{}".format(step, total_step), end="")

        img = batch_data[0]
        label = batch_data[1]
        
        # RESHAPE INPUT ARRAY
        # img = img.reshape(img.shape[0] * img.shape[1], img.shape[2], img.shape[3], img.shape[4])

        model_input = img.to(opt.device)
        label = label.to(opt.device)
        #label = label.squeeze().long()

        loss, _, _, _, _ = model(model_input, label, n=opt.train_module)
        loss = torch.mean(loss, 0)

        loss_epoch += loss.data.cpu().numpy()

    for i in range(opt.model_splits):
        print(
            "Validation Loss Model {}: Time (s): {:.1f} --- {:.4f}".format(
                i, time.time() - starttime, loss_epoch[i] / total_step
            )
        )

    validation_loss = [x/total_step for x in loss_epoch]
    
    #DALI
    #test_loader.reset()
    return validation_loss


def train(opt, model):
    total_step = int(train_loader._size/opt.batch_size_multiGPU)
    #total_step = len(train_loader)
    model.module.switch_calc_loss(True)

    print_idx = 100

    starttime = time.time()
    cur_train_module = opt.train_module

    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch):

        loss_epoch = [0 for i in range(opt.model_splits)]
        loss_updates = [1 for i in range(opt.model_splits)]

        domain_loss_reg = 0.1
        #if epoch >= 1:
        #    domain_loss_reg = 0.1
        #if epoch >= 5:
        #    domain_loss_reg = 0.02
        #if epoch >= 5:
        #    domain_loss_reg = 0.04


        for step, batch_data in enumerate(train_loader):
        #for step, (img, label) in enumerate(train_loader):

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
            # RESHAPE INPUT ARRAY
            # img = img.reshape(img.shape[0] * img.shape[1], img.shape[2], img.shape[3], img.shape[4])
 
            model_input = img.to(opt.device)
            label = label.to(opt.device)#.float()
            #label = label.squeeze().long()

            #if step > 300:
            #     domain_loss_reg = 0.1

            loss, _, _, accuracy, domain_loss = model(model_input, label, n=cur_train_module, domain_reg=domain_loss_reg)
            loss = torch.mean(loss, 0) # take mean over outputs of different GPUs
            domain_loss = torch.mean(domain_loss, 0) # take mean over outputs of different GPUs
            accuracy = torch.mean(accuracy, 0)

            if cur_train_module != opt.model_splits and opt.model_splits > 1:
                loss = loss[cur_train_module].unsqueeze(0) 
                domain_loss = domain_loss[cur_train_module].unsqueeze(0)



            # loop through the losses of the modules and do gradient descent
            for idx, (cur_losses, domain_losses) in enumerate(zip(loss, domain_loss)):
                if len(loss) == 1 and opt.model_splits != 1:
                    idx = cur_train_module
                model.zero_grad()

                if idx == len(loss) - 1:
                    domain_losses.backward()
                    cur_losses.backward()
                else:
                    domain_losses.backward(retain_graph=True)
                    cur_losses.backward(retain_graph=True)

                optimizer[idx].step()
                
                if len(domain_opt) > 0:
                   domain_opt[idx].step()

                print_loss = cur_losses.item()
                print_acc = accuracy[idx].item()
                if step % print_idx == 0:
                    print("\t \t Loss: \t \t {:.4f}".format(print_loss))
                    if opt.loss == 1:
                        print("\t \t Accuracy: \t \t {:.4f}".format(print_acc))

                loss_epoch[idx] += print_loss
                loss_updates[idx] += 1
 

        # DALI reset loader 
        # train_loader.reset()

        # DALI reset loader 
        train_loader.reset()

        if opt.validate:
            validation_loss = validate(opt, model, test_loader) #test_loader corresponds to validation set here
            logs.append_val_loss(validation_loss)

        logs.append_train_loss([x / loss_updates[idx] for idx, x in enumerate(loss_epoch)])
        logs.create_log(model, epoch=epoch, optimizer=optimizer, domain_optimizer=domain_opt)


if __name__ == "__main__":

    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt)
    opt.training_dataset = "unlabeled"

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    if opt.device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    # load model
    model, optimizer, domain_opt = load_vision_model.load_model_and_optimizer(opt)

    logs = logger.Logger(opt)

    train_loader, _, supervised_loader, _, test_loader, _ = get_dataloader.get_dataloader(
        opt
    )

    if opt.loss == 1:
        train_loader = supervised_loader

    try:
        # Train the model
        train(opt, model)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")

    logs.create_log(model)
