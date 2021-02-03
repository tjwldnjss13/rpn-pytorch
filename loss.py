import torch


def custon_rpn_loss(predict_reg, predict_cls, target_reg, target_cls, alpha=10):
    assert predict_reg.shape == target_reg.shape
    assert predict_cls.shape == target_cls.shape

    positive_mask = (target_cls == 1)

    cls_loss = custom_cross_entropy_loss(predict_cls, target_cls)
    reg_loss = custom_smooth_l1_loss(predict_reg, target_reg, positive_mask)

    return cls_loss.sum() + alpha * reg_loss.sum()


def custom_cross_entropy_loss(predict, target):
    assert predict.shape == target.shape

    loss_tensor = -(target * torch.log2(predict + 1e-20) + (1 - target) * torch.log2(1 - predict + 1e-20))

    return loss_tensor.sum() / predict.shape[0]


def custom_smooth_l1_loss(predict, target, external_mask=None):
    assert predict.shape == target.shape

    loss_tensor_temp = predict - target

    loss_tensor_mask_1 = (loss_tensor_temp.abs() < 1)
    loss_tensor_mask_2 = (loss_tensor_temp.abs() >= 1)

    loss_tensor_1 = .5 * loss_tensor_temp ** 2 * loss_tensor_mask_1
    loss_tensor_2 = (loss_tensor_temp.abs() - .5) * loss_tensor_mask_2

    if external_mask is not None:
        loss_tensor_1 *= external_mask
        loss_tensor_2 *= external_mask

    return (loss_tensor_1.sum() + loss_tensor_2.sum()) / predict.shape[0]

