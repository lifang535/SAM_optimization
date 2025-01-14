import torch
import random

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w) 

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  

        self.base_optimizer.step()  

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure!!!"
        closure = torch.enable_grad()(closure) 

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device 
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

class ESAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05,beta=1.0,gamma=1.0,adaptive=False,**kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.beta = beta
        self.gamma = gamma

        defaults = dict(rho=rho,adaptive=adaptive, **kwargs)
        super(ESAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7) / self.beta
            for p in group["params"]:
                p.requires_grad = True 
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w * 1)  
                self.state[p]["e_w"] = e_w



        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]: continue
                p.sub_(self.state[p]["e_w"])  
                self.state[p]["e_w"] = 0

                if random.random() > self.beta:
                    p.requires_grad = False

        self.base_optimizer.step()  

        if zero_grad: self.zero_grad()

    def step(self):
        inputs, targets, loss_fct, model, defined_backward = self.paras
        assert defined_backward is not None, "Sharpness Aware Minimization requires defined_backward, but it was not provided"

        model.require_backward_grad_sync = False
        model.require_forward_param_sync = True


        logits = model(inputs)
        loss = loss_fct(logits,targets)

        l_before = loss.clone().detach()
        predictions = logits
        return_loss = loss.clone().detach()
        loss = loss.mean()
        defined_backward(loss)

        self.first_step(True)


        with torch.no_grad():
            l_after = loss_fct(model(inputs), targets)
            instance_sharpness = l_after - l_before

            prob = self.gamma
            if prob >= 0.99:
                indices = range(len(targets))
            else:
                position = int(len(targets) * prob)
                cutoff,_ = torch.topk(instance_sharpness,position)
                cutoff = cutoff[-1]

                indices = [instance_sharpness > cutoff] 

        model.require_backward_grad_sync = True
        model.require_forward_param_sync = False



        loss = loss_fct(model(inputs[indices]), targets[indices])
        loss = loss.mean()
        defined_backward(loss)
        self.second_step(True)

        self.returnthings = (predictions,return_loss)
 

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

class LookSAM(torch.optim.Optimizer):

    def __init__(self, k, alpha, model, base_optimizer, criterion, rho=0.05, **kwargs):

        defaults = dict(alpha=alpha, rho=rho, **kwargs)
        self.model = model
        super(LookSAM, self).__init__(self.model.parameters(), defaults)

        self.k = k
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.criterion = criterion

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.criterion = criterion
        

    @staticmethod
    def normalized(g):
        return g / g.norm(p=2)

    def step(self, t, samples, targets, zero_grad=False):
        if not t % self.k:
            group = self.param_groups[0]
            scale = group['rho'] / (self._grad_norm() + 1e-7)

            for index_p, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                self.state[p]['old_p'] = p.data.clone()
                self.state[f'old_grad_p_{index_p}']['old_grad_p'] = p.grad.clone()

                with torch.no_grad():
                    e_w = p.grad * scale.to(p)
                    p.add_(e_w)

            self.criterion(self.model(samples), targets).backward()

        group = self.param_groups[0]
        for index_p, p in enumerate(group['params']):
            if p.grad is None:
                continue
            if not t % self.k:
                old_grad_p = self.state[f'old_grad_p_{index_p}']['old_grad_p']
                g_grad_norm = LookSAM.normalized(old_grad_p)
                g_s_grad_norm = LookSAM.normalized(p.grad)
                self.state[f'gv_{index_p}']['gv'] = torch.sub(p.grad, p.grad.norm(p=2) * torch.sum(
                    g_grad_norm * g_s_grad_norm) * g_grad_norm)

            else:
                with torch.no_grad():
                    gv = self.state[f'gv_{index_p}']['gv']
                    p.grad.add_(self.alpha.to(p) * (p.grad.norm(p=2) / (gv.norm(p=2) + 1e-8) * gv))

            p.data = self.state[p]['old_p']

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device) for group in self.param_groups for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )

        return norm

class DynamicSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.00, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(DynamicSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        
        self.rho = rho

    def _dynamic_rho(self, loss):
        """
        根据 loss 动态调整 rho 的值
        
        当 loss 较高时，rho 逐渐较小，以加快收敛速度；
        当 loss 较低时，rho 逐渐较大，以增强模型泛化能力。
        """
        if loss > 1.0 and self.rho != 0.0:
            self.rho = 0.0
        elif loss > 0.5 and self.rho != 0.05:
            self.rho = 0.05
        elif loss > 0.1 and self.rho != 0.1:
            self.rho = 0.1
        elif loss > 0.01 and self.rho != 0.2:
            self.rho = 0.2
        else:
            self.rho = 0.3
            
        for group in self.param_groups:
            group["rho"] = self.rho

    @torch.no_grad()
    def first_step(self, zero_grad=False, loss=None):
        if loss is not None:
            self._dynamic_rho(loss)
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w) 

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  

        self.base_optimizer.step()  

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure!!!"
        closure = torch.enable_grad()(closure) 

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device 
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# sgd adam sam_sgd sam_adam esam_sgd esam_adam looksam_sgd looksam_adam dynamicsam_sgd dynamicsam_adam
def get_optimizer(optimizer_name, 
                  model, 
                  lr=0.001, 
                  weight_decay=0,
                  momentum=0.9):
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=lr, 
                                     weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr, 
                                    momentum=momentum, 
                                    weight_decay=weight_decay)
    elif optimizer_name == 'sam_sgd':
        optimizer = SAM(model.parameters(), 
                        base_optimizer=torch.optim.SGD, 
                        lr=lr, 
                        momentum=momentum)
                        # weight_decay=weight_decay)
    elif optimizer_name == 'sam_adam':
        optimizer = SAM(model.parameters(), 
                        base_optimizer=torch.optim.Adam, 
                        lr=lr)
    elif optimizer_name == 'esam_sgd':
        optimizer = ESAM(model.parameters(), 
                         base_optimizer=torch.optim.SGD, 
                         lr=lr, 
                         momentum=momentum)
    elif optimizer_name == 'esam_adam':
        optimizer = ESAM(model.parameters(), 
                         base_optimizer=torch.optim.Adam, 
                         lr=lr)
    elif optimizer_name == 'looksam_sgd':
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = LookSAM(k=10, 
                            alpha=0.7, 
                            model=model, 
                            base_optimizer=torch.optim.SGD, 
                            criterion=criterion, 
                            rho=0.05, 
                            lr=lr, 
                            momentum=momentum)
    elif optimizer_name == 'looksam_adam':
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = LookSAM(k=10, 
                            alpha=0.7, 
                            model=model, 
                            base_optimizer=torch.optim.Adam, 
                            criterion=criterion, 
                            rho=0.05, 
                            lr=lr)
    elif optimizer_name == 'dynamicsam_sgd':
        optimizer = DynamicSAM(model.parameters(), 
                               base_optimizer=torch.optim.SGD, 
                               lr=lr, 
                               rho=0.2,
                               momentum=momentum)
    elif optimizer_name == 'dynamicsam_adam':
        optimizer = DynamicSAM(model.parameters(), 
                               base_optimizer=torch.optim.Adam, 
                               lr=lr,
                               rho=0.2)
    else:
        raise ValueError('Not supported optimizer: {}'.format(optimizer_name))
    return optimizer
