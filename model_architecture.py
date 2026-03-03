import torch
import torch.nn as nn
import timm

class HexFormerLorentzHead(nn.Module):
    def __init__(self, in_features, num_classes, curvature=1.0):
        super(HexFormerLorentzHead, self).__init__()
        self.c = curvature
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        self.bias = nn.Parameter(torch.Tensor(num_classes))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def expmap0(self, v):
        norm_v = torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=1e-10)
        time_coord = torch.cosh(norm_v)
        space_coord = torch.sinh(norm_v) * (v / norm_v)
        return torch.cat([time_coord, space_coord], dim=-1)

    def forward(self, x):
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-5)
        x = torch.minimum(torch.ones_like(x_norm), 10.0 / x_norm) * x
        return torch.hub.K_linear(x, self.weight, self.bias) if hasattr(torch.hub, 'K_linear') else torch.nn.functional.linear(x, self.weight, self.bias)

def get_medsight_hex_model(num_classes=4):
    # Initializing the base ViT as per your notebook
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.head = nn.Sequential(
    nn.Linear(model.num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes) )
    return model
