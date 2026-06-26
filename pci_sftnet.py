import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# --- Variable Name Mapping Dictionary ---
NAME_MAPPING = {
    'WindSpeed': 'Wind',
    'WINDSPEED': 'Wind',
    'TEMP': 'T2m',
    'Temperature': 'T2m',
    'SoilMoisture': 'SM',
    'DEM': 'DEM',
    'NDVI': 'NDVI',
    'LST': 'LST',
    'Albedo': 'Albedo',
    'Slope': 'Slope',
    'Aspect': 'Aspect',
    'T2m': 'T2m'
}


class SFTLayer(nn.Module):
    """
    Spatial Feature Transform (SFT) Layer
    F_out = F_in * (1 + gamma) + beta
    """
    def __init__(self, in_channels, cond_channels):
        super().__init__()
        self.predict_scale = nn.Conv2d(cond_channels, in_channels, 1)
        self.predict_shift = nn.Conv2d(cond_channels, in_channels, 1)

        # Initialize as identity mapping
        nn.init.constant_(self.predict_scale.weight, 0)
        nn.init.constant_(self.predict_scale.bias, 0)
        nn.init.constant_(self.predict_shift.weight, 0)
        nn.init.constant_(self.predict_shift.bias, 0)

    def forward(self, x, condition):
        # Size alignment
        if condition.shape[-2:] != x.shape[-2:]:
            condition = F.interpolate(condition, size=x.shape[-2:], mode='bilinear', align_corners=False)

        scale = self.predict_scale(condition)
        shift = self.predict_shift(condition)

        # 🔥 Modified: Use Tanh to limit scale amplitude, prevent exploding
        # Allow features to fluctuate within +/- 40% instead of unlimited
        scale = torch.tanh(scale) * 0.4

        return x * (1 + scale) + shift


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) Module
    Purpose: Introduce global context information.
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # 1. Squeeze: Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 2. Excitation: Learn channel weights
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class CausalResBlock(nn.Module):
    def __init__(self, channels, sft_groups, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.sft_layers = nn.ModuleDict()
        # Add SFT layer only when channels > 0
        for group_name, ch_dim in sft_groups.items():
            if ch_dim > 0:
                self.sft_layers[group_name] = SFTLayer(channels, ch_dim)

        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(channels)

    def forward(self, x, conditions):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)

        # Modulate in order from L3 (weak) -> L2 (strong)
        if 'L3' in self.sft_layers and 'L3' in conditions:
            out = self.sft_layers['L3'](out, conditions['L3'])

        if 'L2' in self.sft_layers and 'L2' in conditions:
            out = self.sft_layers['L2'](out, conditions['L2'])

        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        return out + residual


class PCI_SFTNet(nn.Module):
    def __init__(self, pcmci_json_path, feature_dim=64, num_blocks=8):
        super().__init__()

        # 1. Define channel count for each variable
        # [Modification] Removed 'T2m' as it's no longer an input feature
        self.var_channels = {
            'LST': 1, 'NDVI': 1, 'Albedo': 1, 'SM': 1, 'Wind': 1,
            'DEM': 1, 'Slope': 1, 'Aspect': 1
        }

        # 2. Parse JSON structure (force add topographic variables here)
        self.causal_structure = self._parse_pcmci_json(pcmci_json_path)
        print(">>> Corrected model structure:", self.causal_structure)

        # 3. Calculate SFT configuration
        self.sft_config = {}
        for group in ['L2', 'L3']:
            channels = 0
            valid_vars = []
            # Iterate all variables in the group, calculate total channels
            for v in self.causal_structure[group]:
                # [Modification] Only count variables existing in var_channels
                # This way even if T2m is in JSON, it will be ignored since it's not in var_channels
                if v in self.var_channels:
                    channels += self.var_channels[v]
                    valid_vars.append(v)
                else:
                    # Can print debug info here, e.g., ignored T2m
                    pass

            self.sft_config[group] = channels
            self.causal_structure[group] = valid_vars  # Update to valid variable list

        print(">>> SFT Channel Configuration:", self.sft_config)

        # 4. Backbone Network
        total_input_ch = sum(self.var_channels.values())
        self.head = nn.Conv2d(total_input_ch, feature_dim, 3, padding=1)

        # 5. Residual Stacking
        self.body = nn.ModuleList([
            CausalResBlock(feature_dim, self.sft_config, use_se=True) for _ in range(num_blocks)
        ])

        # 6. Output Layer
        self.tail = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, 1, 1)
        )

    def _parse_pcmci_json(self, json_path):
        """Parse JSON and standardize variable names, while forcibly injecting physical prior knowledge"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        def clean_name(name_list):
            cleaned = []
            for n in name_list:
                # 1. Clean prefixes and suffixes
                n_clean = n.replace('DAILY_', '').replace('_average', '')
                # 2. Table lookup mapping
                n_std = NAME_MAPPING.get(n_clean, n_clean)
                cleaned.append(n_std)
            return list(set(cleaned))

        # L2 Processing
        l2_raw = data.get('2', [])
        l2_vars = clean_name(l2_raw[0]) if l2_raw else []

        structure = {
            'L2': l2_vars,
            'L3': []
        }

        # L3 Processing
        for group in data.get('3', []):
            structure['L3'].extend(clean_name(group))

        # 1. Global deduplication & remove T2m
        # [Modification] Ensure structure doesn't contain T2m to prevent logic confusion
        structure['L2'] = [v for v in dict.fromkeys(structure['L2']) if v != 'T2m']
        structure['L3'] = [v for v in dict.fromkeys(structure['L3']) if v != 'T2m']

        # 🔥🔥🔥 [Modification] Inject prior knowledge respectively 🔥🔥🔥

        # 1. Strong driving factors (L2): Topography is the absolute skeleton of temperature distribution
        strong_priors = ['DEM', 'Slope', 'Aspect']
        for var in strong_priors:
            if var not in structure['L2'] and var not in structure['L3']:
                structure['L2'].append(var)
                print(f"🚀 [Physics Prior] Manually added '{var}' to SFT Group L2 (Strong).")

        # 2. Weak/Auxiliary driving factors (L3): Soil moisture as background adjustment, prevent noise from interfering with topographic texture
        weak_priors = ['SM']
        for var in weak_priors:
            if var not in structure['L2'] and var not in structure['L3']:
                structure['L3'].append(var)
                print(f"🚀 [Physics Prior] Manually added '{var}' to SFT Group L3 (Weak/Auxiliary).")

        return structure

    def forward(self, batch_data):
        # 1. Prepare backbone input (all variables as Base Input)
        # [Modification] Removed 'T2m'
        all_keys = ['DEM', 'LST', 'NDVI', 'Albedo', 'SM', 'Wind', 'Slope', 'Aspect']

        if not batch_data:
            raise ValueError("batch_data is empty")

        ref = next(iter(batch_data.values()))
        B, _, H, W = ref.shape
        device = ref.device

        input_list = []
        for k in all_keys:
            if k in batch_data:
                input_list.append(batch_data[k])
            else:
                # If a variable is missing, pad with 0s (although they should all be there)
                input_list.append(torch.zeros(B, 1, H, W, device=device))

        x_in = torch.cat(input_list, dim=1)

        # 2. Prepare SFT conditions (extract based on grouping in _parse_pcmci_json)
        conditions = {}
        for group in ['L2', 'L3']:
            if self.sft_config[group] > 0:
                tensors = []
                # At this point causal_structure['L2'] already contains forced variables like DEM
                # And T2m is removed
                for v in self.causal_structure[group]:
                    if v in batch_data:
                        tensors.append(batch_data[v])
                    else:
                        tensors.append(torch.zeros(B, 1, H, W, device=device))

                conditions[group] = torch.cat(tensors, dim=1)

        # 3. Forward computation
        feat = self.head(x_in)
        for block in self.body:
            feat = block(feat, conditions)
        out = self.tail(feat)

        # 4. Directly output predicted value (no more residual addition)
        # [Modification] Removed return out + batch_data['T2m']
        return out