import torch
import torch.nn as nn
from mmseg_custom import *
from mmseg.apis import init_segmentor, inference_segmentor
from mmseg.models import build_segmentor
from mmcv import Config
from mmcv.runner import load_checkpoint
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import random
from PIL import Image
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from IPython.display import display as ipython_display
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import os

image = Image.open("/content/pths/Lenna.png").convert("RGB")
image = image.resize((256, 256))
image_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0

class Hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input[0]

    def close(self):
        self.hook.remove()

def process_model(cfg_path, checkpoint_path, image_tensor, use_original_hook=False):
    cfg = Config.fromfile(cfg_path)
    model = build_segmentor(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location="cpu")

    mean = torch.tensor([123.675, 116.28, 103.53])
    std = torch.tensor([58.395, 57.12, 57.375])
    normalized_image_tensor = (image_tensor - mean[:, None, None]) / std[:, None, None]
    #image_input = normalized_image_tensor.unsqueeze(0)
    image_input = image_tensor.unsqueeze(0)

    if use_original_hook:
        hook = Hook(model.decode_head.fusion_conv)

        def custom_forward(model, x):
            with torch.no_grad():
                model.eval()
                x = model.backbone(x)
                x = [model.decode_head.convs[i](x[i]) for i in range(len(x))]

                target_size = x[0].size()[2:]
                for i in range(1, len(x)):
                    x[i] = F.interpolate(x[i], size=target_size, mode='bilinear', align_corners=False)

                x = torch.cat(x, dim=1)
                y = model.decode_head.fusion_conv(x)
                z = model.decode_head.dropout(y)
                z = model.decode_head.conv_seg(y)
            hook.output = x
            return y

    else:
        hook = Hook(model.decode_head.conv_seg)

        def custom_forward(model, x):
            with torch.no_grad():
                model.eval()
                x = model.backbone(x)
                x = [model.decode_head.convs[i](x[i]) for i in range(len(x))]

                target_size = x[0].size()[2:]
                out = 0
                for i in range(1, len(x)):
                    x[i] = F.interpolate(x[i], size=target_size, mode='bilinear', align_corners=False)
                    out += x[i]

                x = torch.cat(x, dim=1)
                y = model.decode_head.dropout(x)
                y = model.decode_head.conv_seg(out)
            hook.output = x
            return x

    fusion_input = custom_forward(model, image_input)
    fusion_output = hook.output

    hook.close()

    reshaped_tensor = fusion_output.squeeze(0)
    return reshaped_tensor



def process_model_out(cfg_path, checkpoint_path, image_tensor, use_original_hook=False):
    cfg = Config.fromfile(cfg_path)
    model = build_segmentor(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location="cpu")

    mean = torch.tensor([123.675, 116.28, 103.53])
    std = torch.tensor([58.395, 57.12, 57.375])
    normalized_image_tensor = (image_tensor - mean[:, None, None]) / std[:, None, None]
    #image_input = normalized_image_tensor.unsqueeze(0)
    image_input = image_tensor.unsqueeze(0)

    if use_original_hook:
        hook = Hook(model.decode_head.fusion_conv)

        def custom_forward(model, x):
            with torch.no_grad():
                model.eval()
                x = model.backbone(x)
                x = [model.decode_head.convs[i](x[i]) for i in range(len(x))]

                target_size = x[0].size()[2:]
                for i in range(1, len(x)):
                    x[i] = F.interpolate(x[i], size=target_size, mode='bilinear', align_corners=False)

                x = torch.cat(x, dim=1)
                y = model.decode_head.fusion_conv(x)
                z = model.decode_head.dropout(y)
                z = model.decode_head.conv_seg(y)
            hook.output = y
            return y

    else:
        hook = Hook(model.decode_head.conv_seg)

        def custom_forward(model, x):
            with torch.no_grad():
                model.eval()
                x = model.backbone(x)
                x = [model.decode_head.convs[i](x[i]) for i in range(len(x))]

                target_size = x[0].size()[2:]
                out = 0
                for i in range(1, len(x)):
                    x[i] = F.interpolate(x[i], size=target_size, mode='bilinear', align_corners=False)
                    out += x[i]

                x = out
                y = model.decode_head.dropout(x)
                y = model.decode_head.conv_seg(out)
            hook.output = x
            return x

    fusion_input = custom_forward(model, image_input)
    fusion_output = hook.output

    hook.close()

    reshaped_tensor = fusion_output.squeeze(0)
    return reshaped_tensor

models = [
    {
        'cfg_path': 'configs/segformer/segformer_mit-b2_512x512_160k_ade20k_ws.py',
        'checkpoint_path': '../pths/iter_160000.pth',
        'label': 'SegFormer-B2-ADE20K-WSNet',
        'use_original_hook': False,
    },
    {
        'cfg_path': 'configs/segformer/segformer_mit-b2_512x512_160k_ade20k.py',
        'checkpoint_path': '../pths/segformer_mit-b2_512x512_160k_ade20k_20210726_112103-cbd414ac.pth',
        'label': 'SegFormer-B2-ADE20K',
        'use_original_hook': True,
    },
    {
        'cfg_path': 'configs/segformer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes.py',
        'checkpoint_path': '../pths/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth',
        'label': 'SegFormer-B2-Cityscape',
        'use_original_hook': True,
    },
    {
        'cfg_path': 'configs/segformer/segformer_mit-b5_512x512_160k_ade20k.py',
        'checkpoint_path': '../pths/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth',
        'label': 'SegFormer-B5-ADE20K',
        'use_original_hook': True,
    },
    # Add more models if necessary
]

threshold = 1e-5

def feature_visualize(models, input_image):
    #matplotlib.rcParams['figure.dpi'] = 600  # Set the DPI for the figures
  
    #cmap = plt.cm.gray
    #cmap = LinearSegmentedColormap.from_list('black', cmap(np.linspace(1, 0, 256)))    

    # Set plot background color
    matplotlib.rcParams['figure.facecolor'] = 'black'    

    for model in models:
        reshaped_tensor = process_model(
            cfg_path=model['cfg_path'],
            checkpoint_path=model['checkpoint_path'],
            image_tensor=image_tensor,
            use_original_hook=model['use_original_hook']
        )
        input_tensor = reshaped_tensor

        # Split the tensor into groups of 256 filters each
        groups = input_tensor.split(256, dim=0)

        # Visualize filter kernels in a 2x2 grid for each group
        n_rows, n_cols = 16, 16
        fig = plt.figure(figsize=(24, 24), constrained_layout=True)

        for group_idx, group in enumerate(groups):
            grid = plt.GridSpec(2, 2, wspace=0.1, hspace=0.1)
            row, col = group_idx // 2, group_idx % 2
            ax = fig.add_subplot(grid[row, col])
            ax.axis('off')

            sub_grid = grid[row, col].subgridspec(n_rows, n_cols, wspace=0.05, hspace=0.05)
            for i in range(n_rows):
                for j in range(n_cols):
                    filter_idx = i * n_cols + j
                    filter_kernel = group[filter_idx].numpy()
                    print(f'filter_kernel_{group_idx}_{filter_idx}', filter_kernel.shape)
                    sub_ax = fig.add_subplot(sub_grid[i, j])
                    sub_ax.imshow(filter_kernel, cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
                    sub_ax.axis('off')

        print(model['label'], os.path.dirname(os.path.abspath(__file__)))
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        #plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR,"results",f"feat_{model['label']}.png"), dpi=600)
        plt.close()

        ipython_display(os.path.join(BASE_DIR,"results",f"feat_{model['label']}.png"))

        # Compute the mean image for each group
        mean_images = [group.mean(dim=0).numpy() for group in groups]

        # Find the 10 images in each group that are most similar to the mean image
        similar_images = []
        for group_idx, group in enumerate(groups):
            mean_image = mean_images[group_idx]
            group_np = group.numpy()
            print(group_np.reshape(group_np.shape[0], -1).shape)
            distances = cdist(mean_image.reshape(1, -1), group_np.reshape(group_np.shape[0], -1), metric='cosine')
            indices = np.argsort(-distances.squeeze())[10:]
            similar_images.append(group[indices])

        # Visualize the 10 most similar images for each group
        fig, axs = plt.subplots(4, 10, figsize=(24, 12))
        for i in range(4):
            for j in range(10):
                ax = axs[i, j]
                ax.imshow(similar_images[i][j], cmap='gray', vmin=-1, vmax=1)
                ax.axis('off')

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR,"results",f"similar_images_{model['label']}.png"), dpi=600)
        plt.close()        

        ipython_display(os.path.join(BASE_DIR,"results",f"similar_images_{model['label']}.png"))

#feature_visualize(models, image_tensor)

def plot_histogram_0(models, input_image, n_components):
    input_tensors = []
    for model in models:
        reshaped_tensor = process_model(
            cfg_path=model['cfg_path'],
            checkpoint_path=model['checkpoint_path'],
            image_tensor=image_tensor,
            use_original_hook=model['use_original_hook']
        )
        input_tensors.append(reshaped_tensor)

    threshold = 1e-5

    fig, ax = plt.subplots(figsize=(8, 12))
    colors = ['blue', 'orange', 'green', 'red']
    bar_width = 0.2
    gap_multiplier = 1.2

    for idx, input_tensor in enumerate(input_tensors):
        groups = input_tensor.split(256, dim=0)

        explained_variance_ratios_per_group = []
        for group_num, group in enumerate(groups):
            print("group", group.shape)
            explained_variance_ratios = []
            filter_kernel_np = group.numpy().reshape(256, -1)
            print("filter_kernel_np", filter_kernel_np.shape)
            pca = PCA()
            pca.fit(filter_kernel_np)
            print("n_components", pca.n_components_)

            total_variance = np.sum(pca.explained_variance_)
            explained_variance = 0
            for i in range(0, n_components):
                explained_variance += pca.explained_variance_[i]
                print("moder : ", idx, " / ", "stage : ", group_num, f"explaind variance(pc{i}) : ", pca.explained_variance_[i] / total_variance * 100, "%")

            if total_variance > 0 and explained_variance * 100 > threshold:
                explained_variance_ratios.append(explained_variance / total_variance * 100)
            else:
                explained_variance_ratios.append(0)                
            print("explained variance ratio by PC1~4 : ", explained_variance_ratios[0], "%")

                    
            explained_variance_ratios_per_group.append(explained_variance_ratios)

        #print("size : ", pca.explained_variance_ratio_.shape, pca.explained_variance_ratio_ * 100)
        #print("cumsum : ", np.cumsum(pca.explained_variance_ratio_))
        #plt.plot(np.cumsum(pca.explained_variance_ratio_))
        #plt.plot(explained_variance_ratios_per_group)
        #plt.xlabel('Number of components')
        #plt.ylabel('Explained variance')
        #plt.savefig('elbow_plot.png', dpi=100)

        for group_idx, group_ratios in enumerate(explained_variance_ratios_per_group):
            n = len(group_ratios)
            print("n: ", n)
            label = models[idx]['label'] if group_idx == 0 else None
            y_positions = np.arange(n) * len(input_tensors) * bar_width * gap_multiplier + group_idx * (n * len(input_tensors) * bar_width * gap_multiplier) + idx * bar_width
            ax.barh(y_positions, group_ratios, height=bar_width, color=colors[idx], label=label)

    #ax.set_title('Variance Explained by PC1 for Multiple Models, Divided into Stages')
    legend = ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2), fontsize=14)
    ax.set_yticks(np.arange(0, 4 * len(input_tensors) * bar_width * gap_multiplier * n, len(input_tensors) * bar_width * gap_multiplier * n) + bar_width * (len(models) - 1) / 2)
    ax.set_yticklabels([f'Stage {i}' for i in range(1, 5)], fontsize=14)
    ax.set(ylabel='Features from encoder stages', xlabel='Variance explained by PC1 ~ PC4 [%]')
    ax.set_ylabel(ax.get_ylabel(), fontsize=20)
    ax.set_xlabel(ax.get_xlabel(), fontsize=20)

    ax.set_ylim(ax.get_ylim()[::-1])  # Reverse the y-axis limits


    plt.tight_layout()
    plt.savefig('results/histogram_multiple_models_with_stages_0.png')
    plt.close()

    #ipython_display(Image.open('histogram_multiple_models_with_stages_0.png'))

plot_histogram_0(models, image_tensor, 4)


def plot_histogram_1(models, input_image, n_components):
    input_tensors = []
    for model in models:
        reshaped_tensor = process_model_out(
            cfg_path=model['cfg_path'],
            checkpoint_path=model['checkpoint_path'],
            image_tensor=image_tensor,
            use_original_hook=model['use_original_hook']
        )
        input_tensors.append(reshaped_tensor)

    threshold = 1e-5

    fig, ax = plt.subplots(figsize=(8, 12))
    colors = ['blue', 'orange', 'green', 'red']
    bar_width = 0.2
    gap_multiplier = 1.2

    for idx, input_tensor in enumerate(input_tensors):
        groups = input_tensor.split(256, dim=0)

        explained_variance_ratios_per_group = []
        for group_num, group in enumerate(groups):
            print("group", group.shape)
            explained_variance_ratios = []
            filter_kernel_np = group.numpy().reshape(256, -1)
            print("filter_kernel_np", filter_kernel_np.shape)
            pca = PCA()
            pca.fit(filter_kernel_np)
            print("n_components", pca.n_components_)

            total_variance = np.sum(pca.explained_variance_)
            explained_variance = 0
            for i in range(0, n_components):
                explained_variance += pca.explained_variance_[i]
                print("moder : ", idx, " / ", "stage : ", group_num, f"explaind variance(pc{i}) : ", pca.explained_variance_[i] / total_variance * 100, "%")

            if total_variance > 0 and explained_variance * 100 > threshold:
                explained_variance_ratios.append(explained_variance / total_variance * 100)
            else:
                explained_variance_ratios.append(0)                
            print("explained variance ratio by PC1~4 : ", explained_variance_ratios[0], "%")

                    
            explained_variance_ratios_per_group.append(explained_variance_ratios)

        #print("size : ", pca.explained_variance_ratio_.shape, pca.explained_variance_ratio_ * 100)
        #print("cumsum : ", np.cumsum(pca.explained_variance_ratio_))
        #plt.plot(np.cumsum(pca.explained_variance_ratio_))
        #plt.plot(explained_variance_ratios_per_group)
        #plt.xlabel('Number of components')
        #plt.ylabel('Explained variance')
        #plt.savefig('elbow_plot.png', dpi=100)

        for group_idx, group_ratios in enumerate(explained_variance_ratios_per_group):
            n = len(group_ratios)
            print("n: ", n)
            label = models[idx]['label'] if group_idx == 0 else None
            y_positions = np.arange(n) * len(input_tensors) * bar_width * gap_multiplier + group_idx * (n * len(input_tensors) * bar_width * gap_multiplier) + idx * bar_width
            ax.barh(y_positions, group_ratios, height=bar_width, color=colors[idx], label=label)

    #ax.set_title('Variance Explained by PC1 for Multiple Models, Divided into Stages')
    legend = ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2), fontsize=14)
    ax.set_yticks(np.arange(0, 4 * len(input_tensors) * bar_width * gap_multiplier * n, len(input_tensors) * bar_width * gap_multiplier * n) + bar_width * (len(models) - 1) / 2)
    ax.set_yticklabels([f'fusion_out' for i in range(1, 5)], fontsize=14)
    ax.set(ylabel='Features from encoder stages', xlabel='Variance explained by PC1 ~ PC4 [%]')
    ax.set_ylabel(ax.get_ylabel(), fontsize=14)
    ax.set_xlabel(ax.get_xlabel(), fontsize=16)

    ax.set_ylim(ax.get_ylim()[::-1])  # Reverse the y-axis limits
    ax.set_xlim(0, 100)  # Set the x-axis limits from 0 to 100
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('results/histogram_multiple_models_with_stages_1.png')
    plt.close()

    #ipython_display(Image.open('histogram_multiple_models_with_stages_0.png'))

plot_histogram_1(models, image_tensor, 4)


def plot_histogram(models, input_image, n_components):
    input_tensors = []
    for model in models:
        reshaped_tensor = process_model(
            cfg_path=model['cfg_path'],
            checkpoint_path=model['checkpoint_path'],
            image_tensor=image_tensor,
            use_original_hook=model['use_original_hook']
        )
        input_tensors.append(reshaped_tensor)

    threshold = 1e-5

    fig, ax = plt.subplots(figsize=(15, 6))
    colors = ['blue', 'orange', 'green', 'red']
    bar_width = 0.2

    for idx, input_tensor in enumerate(input_tensors):
        groups = input_tensor.split(256, dim=0)

        explained_variance_ratios_per_group = []
        for group_num, group in enumerate(groups):
            print("group", group.shape)
            explained_variance_ratios = []
            filter_kernel_np = group.numpy().reshape(256, -1)
            print("filter_kernel_np", filter_kernel_np.shape)
            pca = PCA()
            pca.fit(filter_kernel_np)
            print("n_components", pca.n_components_)

            total_variance = np.sum(pca.explained_variance_)
            explained_variance = 0
            for i in range(0, n_components):
                explained_variance += pca.explained_variance_[i]
                print("moder : ", idx, " / ", "stage : ", group_num, f"explaind variance(pc{i}) : ", pca.explained_variance_[i] / total_variance * 100, "%")

            if total_variance > 0 and explained_variance * 100 > threshold:
                explained_variance_ratios.append(explained_variance / total_variance * 100)
            else:
                explained_variance_ratios.append(0)                
            print("explained variance ratio by PC1~4 : ", explained_variance_ratios[0], "%")

                    
            explained_variance_ratios_per_group.append(explained_variance_ratios)

        #print("size : ", pca.explained_variance_ratio_.shape, pca.explained_variance_ratio_ * 100)
        #print("cumsum : ", np.cumsum(pca.explained_variance_ratio_))
        #plt.plot(np.cumsum(pca.explained_variance_ratio_))
        #plt.plot(explained_variance_ratios_per_group)
        #plt.xlabel('Number of components')
        #plt.ylabel('Explained variance')
        #plt.savefig('elbow_plot.png', dpi=100)

        for group_idx, group_ratios in enumerate(explained_variance_ratios_per_group):
            n = len(group_ratios)
            print("n: ", n)
            label = models[idx]['label'] if group_idx == 0 else None
            x_positions = np.arange(n) + idx * bar_width + group_idx * len(input_tensors) * bar_width * 1.5
            ax.bar(x_positions, group_ratios, width=bar_width, color=colors[idx], label=label)

    #ax.set_title('Variance Explained by PC1 for Multiple Models, Divided into Stages')
    ax.legend(loc='upper left', ncol=2, bbox_to_anchor=(0.5, 0.98))
    ax.set_xticks(np.arange(0, 4 * len(input_tensors) * bar_width * 1.5, len(input_tensors) * bar_width * 1.5) + 1.5 * bar_width)
    ax.set_xticklabels([f'Stage {i}' for i in range(1, 5)])
    ax.set(xlabel='Features from encoder stages', ylabel='Variance explained by PC1 [%]')

    plt.tight_layout()
    plt.savefig('results/histogram_multiple_models_with_stages.png')
    plt.close()

    ipython_display(Image.open('histogram_multiple_models_with_stages.png'))

plot_histogram(models, image_tensor, 4)


def plot_histogram_3(models, input_image, n_components):
    input_tensors = []
    for model in models:
        reshaped_tensor = process_model(
            cfg_path=model['cfg_path'],
            checkpoint_path=model['checkpoint_path'],
            image_tensor=image_tensor,
            use_original_hook=model['use_original_hook']
        )
        input_tensors.append(reshaped_tensor)

    threshold = 1e-5
    torch.cat(input_tensor)

    fig, ax = plt.subplots(figsize=(15, 6))
    colors = ['blue', 'orange', 'green', 'red']
    bar_width = 0.2

    for idx, input_tensor in enumerate(input_tensors):
        groups = input_tensor.split(256, dim=0)

        explained_variance_ratios_per_group = []
        for group_num, group in enumerate(groups):
            print("group", group.shape)
            explained_variance_ratios = []
            filter_kernel_np = group.numpy().reshape(256, -1)
            print("filter_kernel_np", filter_kernel_np.shape)
            pca = PCA()
            pca.fit(filter_kernel_np)
            print("n_components", pca.n_components_)

            total_variance = np.sum(pca.explained_variance_)
            for i in range(0, n_components):
                explained_variance = pca.explained_variance_[i]
                if total_variance > 0 and explained_variance * 100 > threshold:
                    explained_variance_ratios.append(explained_variance / total_variance * 100)
                else:
                    explained_variance_ratios.append(0)
                print("moder : ", idx, " / ", "stage : ", group_num, f"explaind variance(pc{i}) : ", explained_variance_ratios[i], "%")
            print("explained variance ratio by PC1~4 : ", sum(explained_variance_ratios[i] for i in range (0, 4)), "%")

                    
            explained_variance_ratios_per_group.append(explained_variance_ratios)

        #print("size : ", pca.explained_variance_ratio_.shape, pca.explained_variance_ratio_ * 100)
        #print("cumsum : ", np.cumsum(pca.explained_variance_ratio_))
        #plt.plot(np.cumsum(pca.explained_variance_ratio_))
        #plt.plot(explained_variance_ratios_per_group)
        #plt.xlabel('Number of components')
        #plt.ylabel('Explained variance')
        #plt.savefig('elbow_plot.png', dpi=100)

        for group_idx, group_ratios in enumerate(explained_variance_ratios_per_group):
            n = len(group_ratios)
            print("n: ", n)
            label = models[idx]['label'] if group_idx == 0 else None
            x_positions = np.arange(n) + idx * bar_width * n_components + group_idx * len(input_tensors) * bar_width * n_components * 1
            ax.bar(x_positions, group_ratios, width=bar_width, color=colors[idx], label=label)

    #ax.set_title('Variance Explained by PC1 for Multiple Models, Divided into Stages')
    ax.legend(loc='upper left', ncol=2, bbox_to_anchor=(0.5, 0.98))
    ax.set_xticks(np.arange(0, 4 * len(input_tensors) * bar_width * i * 1.5, len(input_tensors) * bar_width * i * 1) + 1 * bar_width)
    ax.set_xticklabels([f'Stage {i}' for i in range(1, 5)])
    ax.set(xlabel=f'Features from encoder stages (PCs : {n_components})', ylabel='Variance explained by PC1 [%]')

    plt.tight_layout()
    plt.savefig('results/histogram_multiple_models_with_stages_v2.png')
    plt.close()

    ipython_display(Image.open('histogram_multiple_models_with_stages.png'))

plot_histogram_3(models, image_tensor, 5)


def plot_histogram_2(models, input_image):
    input_tensors = []
    for model in models:
        reshaped_tensor = process_model(
            cfg_path=model['cfg_path'],
            checkpoint_path=model['checkpoint_path'],
            image_tensor=image_tensor,
            use_original_hook=model['use_original_hook']
        )
        input_tensors.append(reshaped_tensor)

    threshold = 1e-5

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharey=True)  
    colors = ['blue', 'orange', 'green', 'red']
    group_labels = []
    bar_width = 0.2

    for idx, input_tensor in enumerate(input_tensors):
        ax = axs[idx // 2, idx % 2]
        groups = input_tensor.split(256, dim=0)

        explained_variance_ratios_per_group = []
        for group in groups:
            explained_variance_ratios = []
            filter_kernel_np = group.numpy().reshape(256, -1)
            pca = PCA(n_components=256)
            pca.fit(filter_kernel_np)

            explained_variance = np.sum(pca.explained_variance_[i] for i in range(1))
            total_variance = np.sum(pca.explained_variance_)

            if total_variance > 0 and explained_variance * 100 > threshold:
                explained_variance_ratios.append(explained_variance / total_variance * 100)
            else:
                explained_variance_ratios.append(0)
            explained_variance_ratios_per_group.append(explained_variance_ratios)

        for group_idx, group_ratios in enumerate(explained_variance_ratios_per_group):
            n = len(group_ratios)
            label = f'Model {idx + 1}, Stage {group_idx + 1}' if idx == 0 else None
            x_positions = np.arange(n) + idx * bar_width + group_idx * len(input_tensors) * bar_width
            ax.bar(x_positions, group_ratios, width=bar_width, color=colors[idx], label=label)

    ax.set_title('Variance Explained by PC1 for Multiple Models, Divided into Stages')
    ax.legend()
    ax.set_xticks(np.arange(0, 4 * len(input_tensors) * bar_width, len(input_tensors) * bar_width) + 1.5 * bar_width)
    ax.set_xticklabels([f'Stage {i}' for i in range(1, 5)])
    ax.set(xlabel='Stages', ylabel='Variance explained by PC1 [%]')

    plt.tight_layout()
    plt.savefig('results/histogram_combined.png')
    plt.show()
    plt.close()

#plot_histogram_2(models, image_tensor)

