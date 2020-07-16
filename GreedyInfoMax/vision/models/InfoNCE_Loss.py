import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

from GreedyInfoMax.utils import model_utils


class InfoNCE_Loss(nn.Module):
    def __init__(self, opt, in_channels, out_channels):
        super().__init__()
        self.opt = opt
        self.negative_samples = self.opt.negative_samples
        self.k_predictions = self.opt.prediction_step

        self.W_k = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            for _ in range(self.k_predictions)
        )

        self.contrast_loss = ExpNLLLoss()

        if self.opt.weight_init:
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                if m in self.W_k:
                    # nn.init.kaiming_normal_(
                    #     m.weight, mode="fan_in", nonlinearity="tanh"
                    # )
                    model_utils.makeDeltaOrthogonal(
                        m.weight,
                        nn.init.calculate_gain(
                            "Sigmoid"
                        ),
                    )

    def forward(self, z, c, skip_step=1):
        # z == c when training unsupervised
        # z is [batch_size, repr, x_coord, y_coord]
        batch_size = z.shape[0]

        total_loss = 0
        total_accuracy = torch.zeros(self.k_predictions)

        if self.opt.device.type != "cpu":
            cur_device = z.get_device()
        else:
            cur_device = self.opt.device

        # For each element in c, contrast with elements below
        for k in range(1, self.k_predictions + 1):
            ### compute log f(c_t, x_{t+k}) = z^T_{t+k} W_k c_t
            # compute z^T_{t+k} W_k:
            ztwk = (
                self.W_k[k - 1]
                .forward(z[:, :, (k + skip_step) :, :])  # Bx, C , H , W
                .permute(2, 3, 0, 1)  # H, W, Bx, C
                .contiguous()
            )  # y, x, b, c

            # Shuffle all possible samples that will be used for contrastive loss
            ztwk_shuf = ztwk.view(
                ztwk.shape[0] * ztwk.shape[1] * ztwk.shape[2], ztwk.shape[3]
            )  # y * x * batch, c
            rand_index = torch.randint(
                ztwk_shuf.shape[0],  # y *  x * batch
                (ztwk_shuf.shape[0] * self.negative_samples, 1),
                dtype=torch.long,
                device=cur_device,
            )
            # Sample more
            rand_index = rand_index.repeat(1, ztwk_shuf.shape[1])

            ztwk_shuf = torch.gather(
                ztwk_shuf, dim=0, index=rand_index, out=None
            )  # y * x * b * n, c

            ztwk_shuf = ztwk_shuf.view(
                ztwk.shape[0],
                ztwk.shape[1],
                ztwk.shape[2],
                self.negative_samples,
                ztwk.shape[3],
            ).permute(
                0, 1, 2, 4, 3
            )  # y, x, b, c, n

            #### Compute  x_W1 . c_t:
            context = (
                c[:, :, : -(k + skip_step), :].permute(2, 3, 0, 1).unsqueeze(-2)
            )  # y, x, b, 1, c
            # Compute for positive
            log_fk_main = torch.matmul(context, ztwk.unsqueeze(-1)).squeeze(
                -2
            )  # y, x, b, 1

            # Compute for negative
            log_fk_shuf = torch.matmul(context, ztwk_shuf).squeeze(-2)  # y, x, b, n

            log_fk = torch.cat((log_fk_main, log_fk_shuf), 3)  # y, x, b, 1+n
            log_fk = log_fk.permute(2, 3, 0, 1)  # b, 1+n, y, x

            # Softmax for all
            log_fk = torch.softmax(log_fk, dim=1)

            # Each example will be considered a class
            # Class 0 is the correct class for all outputs
            true_f = torch.zeros(
                (batch_size, log_fk.shape[-2], log_fk.shape[-1]),
                dtype=torch.long,
                device=cur_device,
            )  # b, y, x

            # Now log_fk_main should be the one closest to zero...
            # count all correct (i.e. zero) predicitons
            predict = torch.argmin(log_fk, dim=1).reshape(-1)
            # print('predicted', predict)
            # print('zero count', (predict == 0).sum(dim=0))

            accuracy = torch.true_divide((predict == 0).sum(dim=0), predict.shape[0])

            # print(f'accuracy {(predict == 0).sum(dim=0)}/{predict.shape[0]} = {accuracy}')
            total_accuracy[k-1] += accuracy

            total_loss += self.contrast_loss(input=log_fk, target=true_f)

        total_loss /= self.k_predictions

        return total_loss, total_accuracy


class ExpNLLLoss(_WeightedLoss):

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(ExpNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        x = torch.log(input + 1e-11)
        return F.nll_loss(x, target, weight=self.weight, ignore_index=self.ignore_index,
                          reduction=self.reduction)
