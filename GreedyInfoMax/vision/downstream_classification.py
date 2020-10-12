import torch
import numpy as np
import time
import os
import pandas as pd

from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

## own modules
from GreedyInfoMax.vision.data import get_dataloader
from GreedyInfoMax.vision.arg_parser import arg_parser
from GreedyInfoMax.vision.models import load_vision_model
from GreedyInfoMax.utils import logger, utils

import neptune
#from neptune import HostedNeptuneBackend

class StoreOutput():
    def __init__(self, name):
        self.outputs = []
        self.name = name
    
    def __call__(self, module, module_in, module_out):
        #print(module_out.shape)
        out = F.adaptive_avg_pool2d(module_out, 1)
        out = out.reshape(-1, 7, 7, out.shape[1])
        out = out.permute(0, 3, 1, 2).contiguous() # batch size, repr, n_patches_x, n_patches_y
        self.outputs.append(out)
        
    def clear(self):
        self.outputs = []

def train_logistic_regression(opt, context_model, predict_model, train_loader, criterion, optimizer, scheduler, exp):
    store_output = None
    hook_handle = None
    if opt.layer_out > -1:
        # Add Forward hook
        for idx, md in context_model.named_modules():
            if idx == f'module.model.0.0.model.layer {opt.layer_out}':
                print('Adding forward hook at layer: ', idx)
                store_output = StoreOutput('train')
                hook_handle = md.register_forward_hook(store_output)

    total_step = len(train_loader)
    predict_model.train()

    starttime = time.time()
    for epoch in range(opt.num_epochs):
        if store_output and len(store_output.outputs) > 0:
            print('store output', torch.cat(store_output.outputs).shape)
            store_output.clear()
            print('cleared: ', len(store_output.outputs))

        epoch_acc1 = 0
        epoch_acc5 = 0

        loss_epoch = 0
        if scheduler:
            print('Learning rate: ', scheduler.get_lr())

        for step, (img, target, patch_id, slide_id) in enumerate(train_loader):
        # for step, (img, target) in enumerate(train_loader):
            predict_model.zero_grad()

            model_input = img.to(opt.device)

            if opt.model_type == 2:  ## fully supervised training
                _, _, z, _ = context_model(model_input, target)
            else:
                with torch.no_grad():
                    _, _, z, _ = context_model(model_input, target)
                z = z.detach() #double security that no gradients go to representation learning part of model

            if store_output:
                #print('using layer output from: ', store_output.name)
                #print(torch.cat(store_output.outputs).shape)
                z = torch.cat(store_output.outputs).detach()
                store_output.clear()

            prediction = predict_model(z)

            target = target.to(opt.device)
            loss = criterion(prediction, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy
            #acc1, acc5 = utils.accuracy(prediction.data, target, topk=(1, 5))
            #epoch_acc1 += acc1
            #epoch_acc5 += acc5
            acc1 =  utils.accuracy(prediction.data, target, topk=(1,))[0]
            epoch_acc1 += acc1

            sample_loss = loss.item()
            loss_epoch += sample_loss

            exp.send_metric(f'loss_0', sample_loss)

            if step % 100 == 0:
                print(
                    '\rEpoch [{}/{}], Step [{}/{}], Time (s): {:.1f}, Acc1: {:.4f}, Acc5: {:.4f}, Loss: {:.4f}'.format(
                        epoch + 1,
                        opt.num_epochs,
                        step,
                        total_step,
                        time.time() - starttime,
                        acc1,
                        0.0, #acc5,
                        sample_loss), end="")
                starttime = time.time()
        if scheduler:
            scheduler.step()

        if opt.validate:
            # validate the model - in this case, test_loader loads validation data
            val_acc1, _ , val_loss, _ = test_logistic_regression(
                opt, context_model, predict_model, test_loader, criterion, exp
            )
            exp.send_metric('val_loss_0', val_loss)
            exp.send_metric('val_acc', val_acc1)
            logs.append_val_loss([val_loss])

        print("\nOverall accuracy for this epoch: ", epoch_acc1 / total_step)
        logs.append_train_loss([loss_epoch / total_step])
        logs.create_log(
            context_model,
            epoch=epoch,
            #predict_model=predict_model,
            accuracy=epoch_acc1 / total_step,
            #acc5=epoch_acc5 / total_step,
        )

    if hook_handle:
        hook_handle.remove()


def test_logistic_regression(opt, context_model, predict_model, test_loader, criterion, exp):
    store_output = None
    hook_handle = None
    if opt.layer_out > -1:
        # Add Forward hook
        for idx, md in context_model.named_modules():
            if idx == f'module.model.0.0.model.layer {opt.layer_out}':
                print('Adding forward hook at layer: ', idx)
                store_output = StoreOutput('test')
                hook_handle = md.register_forward_hook(store_output)

    total_step = len(test_loader)
    context_model.eval()
    predict_model.eval()

    starttime = time.time()

    loss_epoch = 0
    epoch_acc1 = 0
    epoch_acc5 = 0

    all_preds = []
    all_labels = []
    all_slides = []
    all_outputs0 = []
    all_outputs1 = []
    all_patches = []

    for step, (img, target, patch_id, slide_id) in enumerate(test_loader):
    # for step, (img, target) in enumerate(test_loader):
        model_input = img.to(opt.device)

        with torch.no_grad():
            _, _, z, _ = context_model(model_input, target)

        z = z.detach()

        if store_output:
            #print(torch.cat(store_output.outputs).shape)
            z = torch.cat(store_output.outputs).detach()
            store_output.clear()

        prediction = predict_model(z)

        target = target.to(opt.device)
        loss = criterion(prediction, target)

        _, preds = torch.max(prediction.data, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().data.numpy())
        all_patches.extend(patch_id)
        all_slides.extend(slide_id)


        probs = torch.nn.functional.softmax(prediction.data, dim=1).cpu().numpy()
        all_outputs0.extend(probs[:, 0])
        all_outputs1.extend(probs[:, 1])


        # calculate accuracy
        #acc1, acc5 = utils.accuracy(prediction.data, target, topk=(1, 5))
        acc1 = utils.accuracy(prediction.data, target, topk=(1,))[0]
        epoch_acc1 += acc1
        #epoch_acc5 += acc5

        sample_loss = loss.item()
        loss_epoch += sample_loss

        if step % 100 == 0:
            print(
                "\rStep [{}/{}], Time (s): {:.1f}, Acc1: {:.4f}, Acc5: {:.4f}, Loss: {:.4f}".format(
                    step,
                    total_step,
                    time.time() - starttime,
                    acc1,
                    0.0, #acc5,
                    sample_loss), end=""
            )
            starttime = time.time()

        del img
        del target

    if hook_handle:
        hook_handle.remove()
    del store_output 

    print("\nTesting Accuracy: ", epoch_acc1 / total_step)
    df = pd.DataFrame({'label': all_labels,
                       'prediction': all_preds,
                       'patch_id': all_patches,
                       'slide_id': all_slides,
                       'probabilities_0': all_outputs0,
                       'probabilities_1': all_outputs1,
                      })

    for label in df.label.unique():
        label_acc = len(df[(df.label == label) & (df.prediction == df.label)])/len(df[df.label == label])
        exp.send_metric(f'acc_{label}', label_acc)

    return epoch_acc1 / total_step, epoch_acc5 / total_step, loss_epoch / total_step, df


if __name__ == "__main__":

    proxies = {'http': 'http://127.0.0.1:8118', 'https': 'http://127.0.0.1:8118'}

    neptune.init('k-stacke/cpc-greedyinfomax',)# backend=HostedNeptuneBackend(proxies=proxies))

    opt = arg_parser.parse_args()

    add_path_var = "linear_model"

    arg_parser.create_log_path(opt, add_path_var=add_path_var)
    opt.training_dataset = "train"

    exp = neptune.create_experiment(name='CPC linear classification', params=opt.__dict__,
                                    tags=['cpc', 'linear'])

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # load pretrained model
    context_model, _ = load_vision_model.load_model_and_optimizer(
        opt, reload_model=True, calc_loss=False
    )
    context_model.module.switch_calc_loss(False)

    ## model_type=2 is supervised model which trains entire architecture; otherwise just extract features
    if opt.model_type != 2:
        context_model.eval()

    _, _, train_loader, _, test_loader, _ = get_dataloader.get_dataloader(opt)

    classification_model = load_vision_model.load_classification_model(opt)

    if opt.model_type == 2:
        params = list(context_model.parameters()) + list(classification_model.parameters())
    else:
        params = classification_model.parameters()

    optimizer = torch.optim.Adam(params, lr=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    #scheduler = MultiStepLR(optimizer, milestones=[5, 10, 20, 30], gamma=0.1)
    scheduler = None

    logs = logger.Logger(opt)

    try:
        # Train the model
        train_logistic_regression(opt, context_model, classification_model, train_loader, criterion, optimizer, scheduler, exp)

        # Test the model
        acc1, acc5, _, df = test_logistic_regression(
            opt, context_model, classification_model, test_loader, criterion, exp
        )

    except KeyboardInterrupt:
        print("Training got interrupted")

    df.to_csv(
       f"{opt.log_path}/inference_result_model.csv")

    logs.create_log(
        context_model,
        classification_model=classification_model,
        accuracy=acc1,
        #acc5=acc5,
        final_test=True,
    )
    torch.save(
        context_model.state_dict(), os.path.join(opt.log_path, "context_model.ckpt")
    )
