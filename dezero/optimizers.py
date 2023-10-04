class Optimizer:
    def __init__(self):
        self.target = None
        # 在更新参数前对所有参数进行预处理，如权重衰减、梯度裁剪
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks():
            f(params)

        for param in params:
            self.update_one(param)

    def updata_one(self, params):
        raise NotImplementedError()
    
    def add_hook(self, f):
        self.hooks.append(f)