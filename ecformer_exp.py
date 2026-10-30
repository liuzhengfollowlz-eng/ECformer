import torch

class MCAM:
    def __init__(self, model, target_attention_layer,target_grad_layer, detach=True):
        self.model = model.eval()
        self.gradients = []
        self.attentions = []
        self.before_softmax_attentions = []
        self.detach = detach
        self.handles = []
        self.device = next(self.model.parameters()).device

        for _, (name, module) in enumerate(self.model.named_modules()):
            if target_attention_layer in name:
                self.handles.append(module.register_forward_hook(self.get_attention))
            if 'before_softmax' in name:
                self.handles.append(module.register_forward_hook(self.get_before_softmax_attention))
            if target_grad_layer in name:
                self.handles.append(module.register_forward_hook(self.get_gradient))

    def get_attention(self, module, input, output):
        attention = output
        if self.detach:
            self.attentions.append(attention.cpu().detach())
        else:
            self.attentions.append(attention)

    def get_before_softmax_attention(self, module, input, output):
        attention = output
        if self.detach:
            self.before_softmax_attentions.append(attention.cpu().detach())
        else:
            self.before_softmax_attentions.append(attention)

    def get_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.detach:
                self.gradients = [grad.cpu().detach()] + self.gradients
            else:
                self.gradients = [grad] + self.gradients

        output.register_hook(_store_grad)
    
    def __call__(self, x,coords,method="transformer_attribution",start_layer=0,cls_idx=None):
        
        self.attentions = []
        self.before_softmax_attentions=[]
        self.gradients = []
        #x = x.to(self.device)

        self.model.zero_grad()
        output = self.model(x,coords,inf=True)  #B,C
        _, prediction = torch.max(output, 1) #B

        output=output.squeeze(0) #C
        prediction=prediction.squeeze(0)#标量 索引

        if cls_idx is None:
            cls_idx=prediction                              

        loss = output[cls_idx]
        loss.backward()

        num_cls_tokens=1 if not hasattr(self.model, "num_cls_tokens") else self.model.num_cls_tokens
        cams = None
        attentions=torch.stack(self.before_softmax_attentions) # L,B,H,N,N
        gradients=torch.stack(self.gradients) # L,B,H,N,N
        attentions = torch.sigmoid(attentions)
        #attentions = torch.rrelu(attentions)
        cams = attentions * gradients
        #cams = gradients2 
        cams=cams.clamp(min=0).mean(dim=0).mean(dim=1) # B,N,N

        #cams = cams[:, 0:num_cls_tokens, num_cls_tokens:].mean(dim=1) #B,N
        cams = cams[:, 0:num_cls_tokens, :].mean(dim=1) #B,N

        return prediction,cams

    def release(self):
        for handle in self.handles:
            handle.remove()
    