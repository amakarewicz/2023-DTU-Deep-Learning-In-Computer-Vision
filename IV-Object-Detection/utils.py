from torchvision import models
import torchvision.transforms.functional as fn
import torch.nn as nn
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from tqdm.notebook import tqdm
import cv2
import random
import numpy as np
from itertools import compress

from config import *
from eval import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_image_and_annotations(imgs, anns, img_id,):
    
    image_path = [img for img in imgs if img['id'] == img_id]
    assert len(image_path) == 1
    image_path = image_path[0]['file_name']
    ann = [{'id':a['id'],
            'image_id':a['image_id'],
            'category_id': a['category_id'],
            # change w, h to x1, y1 
            'bbox':[a['bbox'][0],
                    a['bbox'][1],
                    a['bbox'][0] + a['bbox'][2],
                    a['bbox'][1]+ a['bbox'][3]],
            'iscrowd':a['iscrowd']
           } for a in anns if a['image_id'] == img_id]
    
    return image_path, ann


def transfer_model_set(model, freeze_convs=False,):
    
    if freeze_convs:
        print('Freezing Convs')
        # freeze the feature extractors
        for param in model.parameters():
            param.requires_grad = False
    
    if type(model) == models.densenet.DenseNet:
        in_features = model.classifier.in_features
    
    elif type(model) == models.resnet.ResNet:
        in_features = model.fc.in_features
    
    
    size_hidden = 512
    out_features = 1
    
    head = nn.Sequential(
                    nn.Linear(in_features, size_hidden),
                    nn.Dropout(DROP_OUT_RATE),
                    nn.ReLU(),
                    nn.BatchNorm1d(size_hidden),
                    nn.Linear(size_hidden, out_features),
                    nn.Sigmoid()        
    )
                    
    
    if type(model) == models.densenet.DenseNet:
        model.classifier = head
    
    elif type(model) == models.resnet.ResNet:
        model.fc = head

    else:
        raise Exception('Not implemented the classifier for this type of model')

    model = model.to(device)

    return model


def selective_search(img):
    """
    Takes image as an input (np.array not Tensor!)
    Returns np.array (number of bboxes x 4)
    Bboxes in format x, y, w, h (see demo notebook for example)
    """
    # create selective search segmentation object
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img) 
    # Choose between fast or accurate selective Search method: fast but low recall V.S. high recall but slow 
    ss.switchToSelectiveSearchFast()
    # AM: Quality takes a looong time, maybe better to try with fast for now and see the results, if bad then change to quality
    # ss.switchToSelectiveSearchQuality() 
    # run selective search
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects))) # TODO: comment out after making the whole trainset work
    return rects


def selective_search_train(images: list, bboxes: list, k: float = 0.5, p: float = 0.01, img_size: int = 256):
    """
    Takes lists of images and bboxes and returns proposals, cropped images and predictions.
    """
    proposals_all = []
    predictions_all = []
    cropped_images_all = []
    for image, img_bboxes in zip(images, bboxes):
        proposals = selective_search(image.permute([1,2,0]).numpy())
        proposals_img = []

        # IoU
        for proposal in proposals:
            scores_all = []
            for bbox in img_bboxes:
                score = IoU(proposal, bbox)
                scores_all.append(score)

            prediction = max(scores_all) > k # Binary classification

            # Extract image
            if prediction or random.random() < p:
                cropped_image = fn.crop(image, *proposal)
                resized_image = fn.resize(cropped_image, size=[img_size, img_size])

                cropped_images_all.append(resized_image)
                predictions_all.append(prediction)
                proposals_img.append(proposal.tolist())
                
        proposals_all.append(proposals_img)
                
    return cropped_images_all, proposals_all, predictions_all


# CUDA out of memory :(
def selective_search_test(images: list, bboxes: list, img_size: int = 256):
    """
    Takes lists of images and bboxes and returns proposals, cropped images and predictions.
    """
    proposals_all = []
    cropped_images_all = []
    for image, img_bboxes in zip(images, bboxes):
        proposals = selective_search(image.permute([1,2,0]).numpy())

        # Extract image
        for proposal in proposals:
            cropped_image = fn.crop(image, *proposal)
            resized_image = fn.resize(cropped_image, size=[img_size, img_size])
            cropped_images_all.append(resized_image)
                
        proposals_all.append(proposals)
                
    return cropped_images_all, proposals_all


def train(model, train_loader, test_loader, loss_function, optimizer, num_epochs, model_name, lr_scheduler=None, save_model=False):
  
    for epoch in tqdm(range(num_epochs), unit='epoch'):
        model.train()
        for minibatch_no, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = [image for image, _, _ in batch]
            bboxes = [bbox for _, bbox, _ in batch]
            labels = [label for _, _, label in batch]
            
            # Selective search
            cropped_images_all, proposals_all, predictions_all = selective_search_train(images, bboxes)            
            data, target = torch.stack(cropped_images_all).to(device), torch.FloatTensor(predictions_all).to(device)
            
            # CNN
            optimizer.zero_grad()
            output = model(data)[:,0]
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
#             train_loss.append(loss.item())
#             predicted = output > 0.5
#             train_correct += (target==predicted).sum().cpu().item()
#             train_len += data.shape[0]
            break
            
        # Test evaluation
        model.eval()
        for batch in test_loader:
            test_images = [image for image, _, _ in batch]
            test_bboxes = [bbox for _, bbox, _ in batch]
            test_labels = [label for _, _, label in batch]
            
            # Selective search
            test_cropped_images_all, test_proposals_all, _ = selective_search_train(test_images, test_bboxes)
            # selective_search_test runs out of memory :(
            test_data = torch.stack(test_cropped_images_all).to(device)
            
            with torch.no_grad():
                outputs = model(test_data)[:,0]
            predicted = (outputs > 0.5).tolist()
            
            # Reshaping
            outputs = outputs.tolist()
            new_shape = [len(l) for l in test_proposals_all]
            output_new_shape, predicted_new_shape = [], []
            head = 0
            for l in new_shape:
                output_new_shape.append(outputs[head:l+head])
                predicted_new_shape.append(predicted[head:l+head])
                head += l

            # Filitering classes from background
            predicted_bboxes = list(compress(test_proposals_all, predicted_new_shape))
            output_new_shape = list(compress(output_new_shape, predicted_new_shape))
            
            pred = [dict(
                boxes=torch.FloatTensor(bboxes),
                scores=torch.FloatTensor(output),
                labels=torch.ones(len(output)) # Simplification for Binary
            ) for bboxes, output in zip(predicted_bboxes, output_new_shape)]
            
            target = [dict(
                boxes=torch.FloatTensor(bboxes),
                labels=torch.FloatTensor(label)
            ) for bboxes, label in zip(test_bboxes, test_labels)]
            
            # Computing mAP
            metric = MeanAveragePrecision()
            metric.update(pred, target)
            print(metric.compute())
            
#             test_correct += (target==predicted).sum().cpu().item()
#             test_len += data.shape[0]

#         if save_model and epoch > 0 and test_correct/test_len > max(out_dict['test_acc']):
#             torch.save(model, 'models/' + model_name)
            
            
#         out_dict['train_acc'].append(train_correct/train_len)
#         out_dict['test_acc'].append(test_correct/test_len)
#         out_dict['train_loss'].append(np.mean(train_loss))
#         out_dict['test_loss'].append(np.mean(test_loss))

        
#         print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
#               f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%")
        
#     return out_dict