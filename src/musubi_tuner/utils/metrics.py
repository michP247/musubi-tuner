import torch
from collections import defaultdict
import numpy as np

def attach_nfn_hooks(model_to_hook, metrics_dict):
    """
    Attaches NFN hooks to a model using the 'proxy' calculation method.
    """
    def hook_fn_proxy(name):
        def hook(module, p_input, p_output):
            try:
                input_tensor = p_input[0].to(torch.float32)
                output_tensor = p_output.to(torch.float32)

                if input_tensor.numel() == 0 or output_tensor.numel() == 0: return

                input_flat = input_tensor.reshape(input_tensor.shape[0], -1)
                output_flat = output_tensor.reshape(output_tensor.shape[0], -1)

                input_norm = torch.linalg.vector_norm(input_flat, dim=1).mean().item()
                output_norm = torch.linalg.vector_norm(output_flat, dim=1).mean().item()

                metrics_dict[name]['nfn'] = output_norm / (input_norm + 1e-8)
            except Exception as e:
                print(f"\n[ERROR] in NFN proxy hook for {name}: {e}")
        return hook

    hooks = []
    for name, module in model_to_hook.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            if 'norm' not in name.lower() and 'emb' not in name.lower():
                hooks.append(module.register_forward_hook(hook_fn_proxy(name)))
    return hooks

def remove_nfn_hooks(hooks):
    for hook in hooks:
        hook.remove()

def average_metrics(metrics_list):
    """
    Averages scores for any module that produced at least one valid result.
    """
    if not metrics_list:
        return {}
        
    sum_metrics = defaultdict(lambda: {'nfn': 0.0, 'count': 0})
    
    for metrics_dict in metrics_list:
        for name, values in metrics_dict.items():
            if 'nfn' in values and isinstance(values['nfn'], (int, float)) and np.isfinite(values['nfn']):
                sum_metrics[name]['nfn'] += values['nfn']
                sum_metrics[name]['count'] += 1
            
    avg_metrics = {}
    for name, data in sum_metrics.items():
        if data['count'] > 0:
            avg_metrics[name] = {'nfn': data['nfn'] / data['count']}
    
    return avg_metrics
