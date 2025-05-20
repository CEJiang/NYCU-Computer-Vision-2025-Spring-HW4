"""PromptIR: A Transformer-based Image Restoration Model with Prompt Injection."""

import torch
from torch import nn
from timm.models.swin_transformer import SwinTransformerBlock


class PromptGenerationModule(nn.Module):
    """Generate task-aware prompts based on input features."""

    def __init__(self, num_prompts, channels):
        super().__init__()
        self.prompt_components = nn.Parameter(
            torch.randn(num_prompts, channels, 1, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.selector = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, num_prompts, kernel_size=1),
            nn.Flatten(start_dim=1),
            nn.Softmax(dim=1)
        )

    def forward(self, features):
        """Forward pass to generate the prompt tensor."""
        pooled = self.gap(features)
        scores = self.selector(pooled)
        prompt = torch.einsum("bn,nchw->bchw", scores, self.prompt_components)
        return prompt


class PromptInteractionModule(nn.Module):
    """Inject prompts and apply transformer + convolutional fusion."""

    def __init__(self, channels, transformer_block):
        super().__init__()
        self.concat = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.transformer_block = transformer_block
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, features, prompt):
        """Forward pass to integrate prompt into features."""
        prompt = prompt.expand_as(features)
        combined = torch.cat([features, prompt], dim=1)
        fused = self.concat(combined)
        transformed = self.transformer_block(fused)
        return self.fuse(transformed)


class PromptBlock(nn.Module):
    """Combine Prompt Generation and Interaction modules."""

    def __init__(self, num_prompts, channels, transformer_block):
        super().__init__()
        self.pgm = PromptGenerationModule(num_prompts, channels)
        self.pim = PromptInteractionModule(channels, transformer_block)

    def forward(self, features):
        """Forward pass through PGM + PIM."""
        prompt = self.pgm(features)
        return self.pim(features, prompt)


class TransformerBlock(nn.Module):
    """Wrap multiple SwinTransformerBlocks into a module."""

    def __init__(self, dim, input_resolution,
                 depth=4, num_heads=3, window_size=7):
        super().__init__()
        self.blocks = nn.Sequential(*[
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_path=0.0,
                norm_layer=nn.LayerNorm
            ) for i in range(depth)
        ])

    def forward(self, x):
        """Forward pass through Swin Transformer blocks."""
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.blocks(x)
        return x.permute(0, 3, 1, 2).contiguous()


class Downsample(nn.Module):
    """Downsample block using conv + pixel unshuffle."""

    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        """Forward downsampling."""
        return self.body(x)


class Upsample(nn.Module):
    """Upsample block using conv + pixel shuffle."""

    def __init__(self, in_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        """Forward upsampling."""
        return self.body(x)


class PromptIR(nn.Module):
    """Main PromptIR architecture with Swin Transformer and prompt blocks."""

    def __init__(self, num_prompts=2):
        super().__init__()
        self.input_proj = nn.Conv2d(3, 96, 3, padding=1)

        # Encoder
        self.encoder1 = TransformerBlock(96, (256, 256))
        self.encoder2 = TransformerBlock(192, (128, 128))
        self.encoder3 = TransformerBlock(384, (64, 64))
        self.encoder4 = TransformerBlock(768, (32, 32))

        self.downsample1_2 = Downsample(96)
        self.downsample2_3 = Downsample(192)
        self.downsample3_4 = Downsample(384)

        # Decoder
        self.upsample4_3 = Upsample(768)
        self.upsample3_2 = Upsample(384)
        self.upsample2_1 = Upsample(192)

        # Prompt blocks
        self.prompt4 = PromptBlock(
            num_prompts, 768, TransformerBlock(768, (32, 32), depth=2))
        self.prompt3 = PromptBlock(
            num_prompts, 384, TransformerBlock(384, (64, 64), depth=2))
        self.prompt2 = PromptBlock(
            num_prompts, 192, TransformerBlock(192, (128, 128), depth=2))
        self.prompt1 = PromptBlock(
            num_prompts, 96, TransformerBlock(96, (256, 256), depth=2))

        # Fusion + decoder
        self.fuse3 = nn.Conv2d(384 + 384, 384, 1)
        self.fuse2 = nn.Conv2d(192 + 192, 192, 1)
        self.fuse1 = nn.Conv2d(96 + 96, 96, 1)

        self.decoder3 = TransformerBlock(384, (64, 64))
        self.decoder2 = TransformerBlock(192, (128, 128))
        self.decoder1 = TransformerBlock(96, (256, 256))

        self.refinement = nn.Sequential(
            TransformerBlock(96, (256, 256)),
            nn.GroupNorm(8, 96)
        )

        self.final = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 3, 1)
        )

    def forward(self, x):
        """Forward pass through PromptIR network."""
        residual = x
        x = self.input_proj(x)

        # Encode
        feat_1 = self.encoder1(x)
        feat_2 = self.encoder2(self.downsample1_2(feat_1))
        feat_3 = self.encoder3(self.downsample2_3(feat_2))
        feat_4 = self.encoder4(self.downsample3_4(feat_3))

        # Prompt injection
        prompt_4 = self.prompt4(feat_4)

        fusion_3 = torch.cat([feat_3, self.upsample4_3(prompt_4)], dim=1)
        fusion_3 = self.fuse3(fusion_3)
        fusion_3 = self.decoder3(fusion_3)
        fusion_3 = self.prompt3(fusion_3) + fusion_3

        fusion_2 = torch.cat([feat_2, self.upsample3_2(fusion_3)], dim=1)
        fusion_2 = self.fuse2(fusion_2)
        fusion_2 = self.decoder2(fusion_2)
        fusion_2 = self.prompt2(fusion_2) + fusion_2

        fusion_1 = torch.cat([feat_1, self.upsample2_1(fusion_2)], dim=1)
        fusion_1 = self.fuse1(fusion_1)
        fusion_1 = self.decoder1(fusion_1) + fusion_1

        # Final refinement
        refined = self.refinement(fusion_1) + fusion_1
        output = self.final(refined)

        return torch.sigmoid(output + residual)
