import os
import PIL
import numpy as np
import torch
import sys
sys.path.append('./cell_vista_segmentation')

from cuml import DBSCAN, PCA

from matplotlib import pyplot as plt
from monai.transforms import ScaleIntensityRangePercentiles

from cell_vista_segmentation.scripts.dynamics import compute_masks
from cell_vista_segmentation.scripts.cell_sam_wrapper import CellSamWrapper

# Wrapper for Vista2D model so we can extract embeddings
class CellFeatureExtractor(CellSamWrapper):
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=self.network_resize_roi, mode='bilinear')
        x = self.model.image_encoder(x)
        return x

# Input: 
# img_path - File path to an image 
# Output: 
# img_arry - Numpy image (1, 3, 256, 256) to be ingested by model 
def load_and_pre_process_image(img_path):

    # Load and massage dimensions 
    img_arr = np.array(PIL.Image.open(img_path)) # Load the image (400, 400, 3)  
    if img_arr.ndim < 3: 
        img_arr = np.expand_dims(img_arr, axis=2)
        img_arr = np.repeat(img_arr, 3, axis=2)
    img_arr = np.transpose(img_arr, (2, 0, 1)) # Image must be channel first (3, 400, 400) 
    img_arr = np.expand_dims(img_arr, axis=0) # Add batch dimension in front (1, 3, 400, 400)   
    # img_arr = img_arr[:,:,:256,:256] # Crop (1, 3, 256, 256) 

    # Convert to Torch object
    patch = torch.from_numpy(img_arr).float().cuda()

    # Normalize
    norm_transform = ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0.0, b_max=1.0, channel_wise=True, clip=True)
    patch[0] = norm_transform(patch[0])

    return patch

# Input: 
# patch - torch array to extract features from 
# checkpoint_path - Path to model.print
# Output: 
# patch_features - Features extracted from img_arr
def extract_features(patch, checkpoint_path): 
    
    # Re arrange dimensions to go into network 
    patch = np.transpose(patch, (2, 0, 1)) # Image must be channel first (3, 400, 400) 
    patch = np.expand_dims(patch, axis=0) # Add batch dimension in front (1, 3, 400, 400)  
    patch = torch.from_numpy(patch).float().cuda()

    # Load feature extractor
    checkpoint = torch.load(checkpoint_path)
    feature_extractor = CellFeatureExtractor(checkpoint=None)
    feature_extractor.load_state_dict(checkpoint['state_dict'], strict=False)
    feature_extractor.cuda().eval()

    # Extract features from patch 
    features = feature_extractor(patch) # (1, 256, 64, 64)
    patch_features = features.mean(dim=[2,3]) # (1, 256) 

    return patch_features.cpu().detach().numpy()

# Input: 
# img_arr - Numpy array to extract features from
# checkpoint_path - Path to model.print
# Output: 
# segmentation - Segmentation map of img_arr
def segment_cells(img_path, checkpoint_path): 

    # Load image 
    patch = load_and_pre_process_image(img_path)

    # Load segmentation model
    checkpoint = torch.load(checkpoint_path)
    seg_model = CellSamWrapper(checkpoint=None)
    seg_model.load_state_dict(checkpoint['state_dict'], strict=False)
    seg_model.cuda().eval()

    # Extract features from patch 
    segmentation = seg_model(patch) # (1, 3, 256, 256)
    segmentation = segmentation.cpu().detach().numpy()

    # Calculate individual masks
    pred_mask_all = []
    for b_ind in range(segmentation.shape[0]):  # go over batch dim
        dP = segmentation[b_ind, 1:]  # vectors
        cellprob = segmentation[b_ind, 0]  # foreground prob (logit)
        resize = [segmentation.shape[2], segmentation.shape[3]]
        pred_mask, p = compute_masks(
            dP,
            cellprob,
            niter=200,
            cellprob_threshold=0.4,
            flow_threshold=0.4,
            interp=True,
            resize=resize,
            use_gpu=True,
            device=None,
        )
        pred_mask_all.append(pred_mask)

    pred_mask_all = np.array(pred_mask_all)

    # Post process for visualization
    segmentation = np.squeeze(segmentation[0,0,:,:])

    print("There are %d unique cells" % len(np.unique(pred_mask_all[0])))

    return patch, segmentation, pred_mask_all

def plot_segmentation(patch, segmentation, pred_mask_all): 

    # Re-arrange axes and datatypes so they can be plotted easily 
    segmentation = segmentation.astype(np.uint8)
    pred_mask = pred_mask_all[0].astype(np.uint8)
    patch_image = np.transpose(np.squeeze(patch.cpu().detach().numpy()), (1, 2, 0))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_figwidth(30)
    fig.suptitle('Cell Segmentations')

    ax1.set_title("Original Cell Image")
    ax1.imshow(patch_image)

    ax2.set_title("Segmentations")
    ax2.imshow(segmentation)

    ax3.set_title("Individual Masks")
    ax3.imshow(pred_mask)

def feature_extract(pred_mask, patch, checkpoint_path): 
    pred_mask = pred_mask[0].astype(np.uint8)
    patch_image = np.transpose(np.squeeze(patch.cpu().detach().numpy()), (1, 2, 0))

    feature_size = 256
    num_cells = len(np.unique(pred_mask))-1
    cell_features = np.zeros((num_cells, feature_size))

    for i in range(num_cells): 
        
        # Get masked image for each cell 
        patch_mask = (pred_mask == (i+1))
        patch_mask = np.repeat(np.expand_dims(patch_mask,2), 3, axis=2)
        cell_mask = np.multiply(patch_image, patch_mask) # (256, 256, 3)
        
        # Get features for each cell 
        features = np.squeeze(extract_features(cell_mask, checkpoint_path)) # (,256)
        cell_features[i,:] = features
        
    return cell_features

def plot_cells_by_class(seg_class, labels_dict, patch, pred_mask): 
 
    cells_in_class = labels_dict[seg_class][0]
    patch_image = np.transpose(np.squeeze(patch.cpu().detach().numpy()), (1, 2, 0))

    # Determine how many rows and columns to plot 
    num_cells = len(cells_in_class)
    num_cols = 5
    num_rows = num_cells // num_cols + 1

    # Plot segmentation of each cell 
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.suptitle('Cell Class %d' % seg_class)
    fig.set_figheight(3*num_rows)
    fig.set_figwidth(15)

    for i in range(num_cells): 

        # mask this cell 
        cell_id = cells_in_class[i] 
        patch_mask = (pred_mask[0].astype(np.uint8) == cell_id)
        patch_mask = np.repeat(np.expand_dims(patch_mask,2), 3, axis=2)
        cell_mask = np.multiply(patch_image, patch_mask) 

        # calculate bounding box
        rows = np.any(cell_mask, axis=1)
        cols = np.any(cell_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cell_mask = cell_mask[rmin:rmax, cmin:cmax,:]

        # plot 
        if num_rows==1: 
            axs[i%num_cols].imshow(cell_mask)
        else: 
            axs[i//num_cols, i%num_cols].imshow(cell_mask)