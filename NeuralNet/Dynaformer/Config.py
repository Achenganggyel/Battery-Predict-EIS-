#Use Config.py instead of .yaml to simplify the process
class Config():
    def __init__(self):
        self.model_name == "dynaformer"
        self.model_lr = 0.0001
        self.model_instance_norm =  False #图像处理领域的归一化方法
        self.model_batch_size = 64
        self.model_loss = 'rmse'
        self.model_patience_lr_plateau = 100 #`patience_lr_plateau` 是指在学习率衰减时，
                                            # 等待多少次 epoch、 模型性能没有提升就降低学习率。
                                            # 通常用于避免在训练中过早停止学习率衰减，以便在训练后期仍能获得更好的性能。
                                            # 这个参数通常在 `torch.optim.lr_scheduler.ReduceLROnPlateau` 中使用。
        