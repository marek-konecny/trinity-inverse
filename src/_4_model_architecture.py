if __name__ == "__main__": raise RuntimeError("Don't run this directly as the main entry point.")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as ttransforms
from typing import List, Tuple

## I.I.D. Submodules ##
class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int) -> None:
        super().__init__()
        channels: List[int] = []
        for i in range(num_layers + 1):
            # exponential interpolation from in_channels to out_channels
            t = i / num_layers
            channel = int(in_channels ** (1 - t) * out_channels ** t)
            channels.append(channel)
        
        layers: List[nn.Module] = []
        for i in range(num_layers):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            dilation = min(2 ** i, 4)
            padding = dilation
            
            layers.append(nn.Conv2d(in_ch, out_ch, 3, 1, padding=padding, dilation=dilation))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class ASPPBlock(nn.Module):
    def __init__(self, channels: int, dilations: List[int]) -> None:
        super().__init__()
        self.aspp_branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        ])
        
        for dilation in dilations:
            self.aspp_branches.append(
                nn.Sequential(nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False), nn.LeakyReLU(0.1, inplace=True))
            )
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False), nn.LeakyReLU(0.1, inplace=True)
        )
        
        num_branches = len(self.aspp_branches) + 1  # +1 for global_pool
        self.project = nn.Sequential(nn.Conv2d(channels * num_branches, channels, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.aspp_branches]
        
        x_global = self.global_pool(x)
        x_global = x_global.expand(-1, -1, x.shape[2], x.shape[3])
        branch_outputs.append(x_global)
        
        out = torch.cat(branch_outputs, dim=1)
        return self.project(out)

class UNetBlock(nn.Module):
    class UNetDoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, inplace=True)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.double_conv(x)
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.inc = self.UNetDoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), self.UNetDoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), self.UNetDoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), self.UNetDoubleConv(128, 256))
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = self.UNetDoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = self.UNetDoubleConv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = self.UNetDoubleConv(64, 32)
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4)
        diffY = x3.size()[2] - x.size()[2]
        diffX = x3.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x3, x], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        diffY = x2.size()[2] - x.size()[2]
        diffX = x2.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)
        return self.outc(x)

##   I.I.D. Module   ##
class IntrinsicImageDecompositionModule(nn.Module):
    def __init__(self, iid_submodules: List[nn.Module], latent_channels: int, r_high_skip_connection: bool) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.r_high_skip_connection = r_high_skip_connection
        
        self.iid_submodules = nn.ModuleList(iid_submodules)

        self.head_surface = nn.Conv2d(latent_channels, 4, 1)
        self.head_r_low = nn.Sequential(
            nn.Conv2d(latent_channels, 32, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 3, 1),
            ttransforms.GaussianBlur(kernel_size=13, sigma=3.0)
        )
        self.head_r_high = nn.Sequential(
            nn.Conv2d(latent_channels + (3 if r_high_skip_connection else 0), 64, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 3, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = x
        for submodule in self.iid_submodules:
            feat = submodule(feat)

        feat_global = feat

        phys = self.head_surface(feat_global)
        log_c = phys[:, 0:1]
        
        # compositional contrast
        c = torch.exp(log_c)

        # surface normals
        n_raw = phys[:, 1:4]
        n_raw_enforced = torch.cat([n_raw[:, 0:2], torch.abs(n_raw[:, 2:3])], dim=1)
        n_norm = torch.norm(n_raw_enforced, p=2, dim=1, keepdim=True)
        n = n_raw_enforced / (n_norm + 1e-8)

        # per-detector low-frequency residuals
        r_low = self.head_r_low(feat_global)
        
        # underlying white noise map
        if self.r_high_skip_connection:
            f_with_input_skip = torch.cat([feat_global, x], dim=1)
        else:
            f_with_input_skip = feat_global
        eta = self.head_r_high(f_with_input_skip)

        return c, n, r_low, eta


##    D.R. Module   ##
class DifferentiableRendererModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # the defaults for these detector parameters below were experimentally derived
        # from initial pre-training runs. these specific values are not critical, as
        # the model should dynamically adjust and optimize them during training.

        # learned BSE detector projection vector
        self.p = nn.Parameter(torch.tensor([
            [0.2, 0.2], # ETD
            [0.0, 0.0], # T1
            [-0.1, 0.1] # T2
        ]))

        # learned detector SE/BSE mixing coefficient
        self.alpha_logits = nn.Parameter(torch.tensor([
            -1.0, # ETD (~0.27 SE)
            -2.0, # T1 (~0.12 SE)
            2.0   # T2 (~0.88 SE) 
        ]).view(1,3,1,1))

        # learned SE shading coefficients
        self.beta = nn.Parameter(torch.tensor([0.5]).view(1,1,1,1))
        self.w = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]).view(1,3,1,1))
        self.b = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]).view(1,3,1,1))

    def forward(self, x: torch.Tensor, c: torch.Tensor, n: torch.Tensor, r_low: torch.Tensor, eta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nx = n[:, 0:1]
        ny = n[:, 1:2]
        nz = n[:, 2:3]

        px = self.p[:, 0].view(1, 3, 1, 1)
        py = self.p[:, 1].view(1, 3, 1, 1)
        dot_prod = nx*px + ny*py + nz

        # predicted image gain
        g = x.mean(dim=(2, 3), keepdim=True) + 1e-6

        # BSE shading term
        s_bse = c * F.relu(dot_prod)
        
        # SE shading term
        s_se = (self.w * c + self.b) * torch.exp(self.beta * (1.0 - nz))
        
        # intrinsic ideal shading term
        alpha = torch.sigmoid(self.alpha_logits)
        S = (1.0 - alpha) * s_bse + alpha * s_se

        # per-detector high-frequency residuals
        clean_est = g * (S + r_low)
        r_high = eta * torch.sqrt(torch.clamp(clean_est, min=1e-6)).detach()

        # renderer formula
        f_img = clean_est + r_high
        return f_img, r_high, g


class TrinityInverseModel(nn.Module):
    def __init__(self, latent_channels: int = 96, r_high_skip_connection: bool = True, arch_config: dict = None) -> None:
        super().__init__()
        if arch_config is None:
            arch_config = {"type": "encoder_aspp", "encoder_layers": 4, "aspp_dilations": [6, 12, 18]}

        arch_type = arch_config["type"]
        if arch_type == "encoder_aspp":
            iid_submodules: List[nn.Module] = [
                EncoderBlock(in_channels=3, out_channels=latent_channels, num_layers=arch_config["encoder_layers"]),
                ASPPBlock(channels=latent_channels, dilations=arch_config.get("aspp_dilations", [6, 12, 18]))
            ]
        elif arch_type == "unet":
            iid_submodules = [UNetBlock(in_channels=3, out_channels=latent_channels)]
        elif arch_type == "encoder_unet":
            iid_submodules = [
                EncoderBlock(in_channels=3, out_channels=latent_channels, num_layers=arch_config["encoder_layers"]),
                UNetBlock(in_channels=latent_channels, out_channels=latent_channels)
            ]
        else:
            raise ValueError(f"Unknown architecture type: {arch_type}")

        self.arch_config = arch_config
        self.iid_module = IntrinsicImageDecompositionModule(
            iid_submodules=iid_submodules,
            latent_channels=latent_channels,
            r_high_skip_connection=r_high_skip_connection
        )
        self.dr_module = DifferentiableRendererModule()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c, n, r_low, eta = self.iid_module(x)
        f_img, r_high, g = self.dr_module(x, c, n, r_low, eta)
        nx, ny, nz = n[:, 0:1], n[:, 1:2], n[:, 2:3]
        return f_img, c, nx, ny, nz, r_low, r_high, eta, g


# Alias for backward compatibility
ProposedModelArchitecture = TrinityInverseModel
