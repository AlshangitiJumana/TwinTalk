import torch
import torch.nn as nn
import torch.nn.functional as F

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        t_kernel, v_kernel = kernel_size
        self.tcn_padding = (t_kernel // 2, 0)

        # Graph Convolution
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1, v_kernel), padding=(0, v_kernel // 2))

        # Temporal Convolution
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(t_kernel, 1), stride=(stride, 1), padding=(0, 0)),
            nn.BatchNorm2d(out_channels)
        )

        # Residual Connection Fix
        if not residual:
            self.residual = lambda x: torch.zeros_like(x)
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)

        # Padding before temporal conv
        x = F.pad(x, (0, 0, self.tcn_padding[0], self.tcn_padding[0]))

        x = self.gcn(x)
        x = self.tcn(x)

        # Handle shape mismatch if needed
        if x.shape != res.shape:
            min_c = min(x.shape[1], res.shape[1])
            min_t = min(x.shape[2], res.shape[2])
            min_v = min(x.shape[3], res.shape[3])
            x = x[:, :min_c, :min_t, :min_v]
            res = res[:, :min_c, :min_t, :min_v]

        x += res
        return self.relu(x)




class Model(nn.Module):
    def __init__(self, in_channels, num_class, graph_args, edge_import=False, **kwargs):
        super().__init__()

        self.st_gcn_networks = nn.ModuleList((
             STGCNBlock(in_channels, 64, kernel_size=(9, 3), residual=True),
            STGCNBlock(64, 64, kernel_size=(9, 3)),
            STGCNBlock(64, 64, kernel_size=(9, 3)),
            STGCNBlock(64, 128, kernel_size=(9, 3), stride=2),
            STGCNBlock(128, 128, kernel_size=(9, 3)),
            STGCNBlock(128, 256, kernel_size=(9, 3), stride=2),
            STGCNBlock(256, 256, kernel_size=(9, 3)),
        ))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        # x shape: [N, C, T, V]
        for gcn in self.st_gcn_networks:
            x = gcn(x)
        x = self.pool(x)  # [N, 256, 1, 1]
        x = x.view(x.size(0), -1)
        return self.fc(x)
