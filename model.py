
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from CT import CT

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, isrelu=True):
        super(BasicConv2d, self).__init__()
        if kernel_size == 1:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                          dilation=dilation, bias=False)
        else:
            self.conv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
            )
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.isrelu = isrelu

    def forward(self, x):
        x = self.conv(x)
        if self.isrelu:
            x = self.relu(x)
        return x

class FoldAttention(nn.Module):
    def __init__(self, indim=64, patch_size=8, image_size=64, num_heads=32,qkv_bias=False):
        super(FoldAttention, self).__init__()
        self.dim = patch_size ** 2 * indim
        self.norm_q, self.norm_k_pan, self.norm_v_pan = nn.LayerNorm(self.dim*2), nn.LayerNorm(self.dim), nn.LayerNorm(self.dim)
        self.norm_k_ms, self.norm_v_ms = nn.LayerNorm(self.dim), nn.LayerNorm(self.dim)
        self.to_q = nn.Linear(self.dim*2, self.dim//4, bias=qkv_bias)
        self.to_k_pan = nn.Linear(self.dim, self.dim//4, bias=qkv_bias)
        self.to_v_pan = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.to_k_ms = nn.Linear(self.dim, self.dim//4, bias=qkv_bias)
        self.to_v_ms = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.feat2patch = torch.nn.Unfold(kernel_size=patch_size, padding=0, stride=patch_size)
        self.patch2feat = torch.nn.Fold(output_size=(image_size, image_size), kernel_size=patch_size, padding=0, stride=patch_size)
        self.scale = (self.dim / num_heads) ** (-0.5)
        self.image_size = image_size
        self.heads = num_heads
        self.proj1 = nn.Linear(self.dim, self.dim)
        self.proj2 = nn.Linear(self.dim, self.dim)

    def get_qkv(self, pan, ms):
        q = torch.cat([pan,ms],1)
        unfold_q = self.feat2patch(q)
        unfold_q = rearrange(unfold_q, "b c n -> b n c")
        unfold_q = self.to_q(self.norm_q(unfold_q))
        q = rearrange(unfold_q, "b n (g c) -> b g n c", g=self.heads)

        unfold_k_pan = self.feat2patch(pan)
        unfold_k_pan = rearrange(unfold_k_pan, "b c n -> b n c")
        unfold_k_pan = self.to_k_pan(self.norm_k_pan(unfold_k_pan))
        k_pan = rearrange(unfold_k_pan, "b n (g c) -> b g n c", g=self.heads)

        unfold_v_pan = self.feat2patch(pan)
        unfold_v_pan = rearrange(unfold_v_pan, "b c n -> b n c")
        unfold_v_pan = self.to_v_pan(self.norm_v_pan(unfold_v_pan))
        v_pan = rearrange(unfold_v_pan, "b n (g c) -> b g n c", g=self.heads)

        unfold_k_ms = self.feat2patch(ms)
        unfold_k_ms = rearrange(unfold_k_ms, "b c n -> b n c")
        unfold_k_ms = self.to_k_ms(self.norm_k_ms(unfold_k_ms))
        k_ms = rearrange(unfold_k_ms, "b n (g c) -> b g n c", g=self.heads)

        unfold_v_ms = self.feat2patch(ms)
        unfold_v_ms = rearrange(unfold_v_ms, "b c n -> b n c")
        unfold_v_ms = self.to_v_ms(self.norm_v_ms(unfold_v_ms))
        v_ms = rearrange(unfold_v_ms, "b n (g c) -> b g n c", g=self.heads)

        return q, k_pan, v_pan, k_ms, v_ms

    def forward(self, pan, ms):
        q, k_pan, v_pan, k_ms, v_ms = self.get_qkv(pan, ms)

        attn_pan = (q @ k_pan.transpose(-2, -1)) * self.scale
        attn_ms = (q @ k_ms.transpose(-2, -1)) * self.scale

        attn_pan = F.softmax(attn_pan, dim=-1)
        attn_ms = F.softmax(attn_ms, dim=-1)

        out_pan = (attn_pan @ v_pan).transpose(1, 2)
        out_ms = (attn_ms @ v_ms).transpose(1, 2)

        out_pan = rearrange(out_pan, "b q g c -> b (g c) q")
        out_ms = rearrange(out_ms, "b q g c -> b (g c) q")

        out_pan = self.patch2feat(out_pan)
        out_ms = self.patch2feat(out_ms)

        out = torch.cat([out_pan+pan, out_ms+ms], 1)

        return out

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class FusionBlock(torch.nn.Module):
    def __init__(self, channels, patch_size, image_size):
        super(FusionBlock, self).__init__()
        self.fold_attn = FoldAttention(channels, patch_size, image_size)
        self.conv = BasicConv2d(2 * channels, channels, 1, 1, isrelu=False)

    def forward(self, x_pan, x_ms):

        a_cat = self.fold_attn(x_pan, x_ms)
        out = self.conv(a_cat)
        return out

class Fusion_network(nn.Module):
    def __init__(self, nC):
        super(Fusion_network, self).__init__()
        img_size = [64, 32, 16]
        patch_size = [4, 4, 1]

        self.fusion_block1 = FusionBlock(nC[0], patch_size[0], img_size[0])
        self.fusion_block2 = FusionBlock(nC[1], patch_size[1], img_size[1])
        self.fusion_block3 = FusionBlock(nC[2], patch_size[2], img_size[2])

    def forward(self, en_ir, en_vi):
        f1_0 = self.fusion_block1(en_ir[0], en_vi[0])
        f2_0 = self.fusion_block2(en_ir[1], en_vi[1])
        f3_0 = self.fusion_block3(en_ir[2], en_vi[2])

        return [f1_0, f2_0, f3_0]

class tail(nn.Module):
    def __init__(self, nb_filter, output_nc=4, deepsupervision=True):
        super(tail, self).__init__()
        self.deepsupervision = deepsupervision
        self.nb_filter = nb_filter
        block = DenseBlock_light
        self.DB1_1 = block(nb_filter[0], 32)
        self.DB2_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.DB3_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.DB2_2 = block(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv_out = BasicConv2d(32, output_nc, 1, isrelu=False)

        self.up4_1 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(self.nb_filter[2], self.nb_filter[2]*4, 3, 1, 0),
                nn.PixelShuffle(2)
            )
        self.up3_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.nb_filter[1], self.nb_filter[1]*4, 3, 1, 0),
            nn.PixelShuffle(2)
        )
        self.up3_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.nb_filter[1], self.nb_filter[1]*4, 3, 1, 0),
            nn.PixelShuffle(2)
        )
        self.up2_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.nb_filter[0], self.nb_filter[0] * 4, 3, 1, 0),
            nn.PixelShuffle(4)
        )
        self.up2_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.nb_filter[0], self.nb_filter[0] * 4, 3, 1, 0),
            nn.PixelShuffle(4)
        )

    def forward(self, x4_t, x3_t, x2_t):

        x2_1 = self.DB2_1(torch.cat([x2_t, self.up3_1(x3_t)], 1))
        x3_1 = self.DB3_1(torch.cat([x3_t, self.up4_1(x4_t)], 1))
        x2_2 = self.DB2_2(torch.cat([x2_1, self.up3_2(x3_1)], 1))
        out = self.conv_out(torch.cat([self.up2_1(x2_1), self.up2_2(x2_2)], 1))
        return out

class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseBlock_light, self).__init__()
        out_channels_def = int(in_channels / 2)
        denseblock = []
        denseblock += [BasicConv2d(in_channels, out_channels_def),
                       BasicConv2d(out_channels_def, out_channels, 1)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out

class model(nn.Module):
    def __init__(self, nb_filter=[64, 128, 256, 512]):
        super(model, self).__init__()

        self.msbackbone = CT(in_chans=4)
        self.panbackbone = CT(in_chans=1)
        self.fusion_model = Fusion_network(nb_filter)
        self.decoder = tail(nb_filter)

    def forward(self, ms, pan):

        ms_pvt = self.msbackbone(ms)
        pan_pvt = self.panbackbone(pan)
        f = self.fusion_model(ms_pvt, pan_pvt)
        x4_t = f[2]
        x3_t = f[1]
        x2_t = f[0]
        out = self.decoder(x4_t, x3_t, x2_t)

        return out


if __name__ == '__main__':
    model = model()
    ms = torch.randn(1, 4, 256, 256)
    pan = torch.randn(1, 1, 256, 256)

    prediction = model(ms, pan)
    print(prediction.size())
