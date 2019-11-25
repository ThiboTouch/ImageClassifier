import argparse
import torch
import torchvision
from torch import optim
import json
from PIL import Image
import numpy as np

def main():
    in_arg = get_input_args()
    
    device = in_arg.device()
    
    model, optimizer = load_checkpoint(in_arg.checkpoint)
    
    cat_mapping = get_category_names(in_arg.category_names)
    
    prob, classes = predict(in_arg.input, model, int(in_arg.top_k), device)
    
    names = map_labels_to_names(cat_mapping, classes, int(in_arg.top_k))
    
    print(names)
    print(prob)
    
    
def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return model, optimizer

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    
    model = model.eval()
    
    image = process_image(image_path)
    
    image  = torch.from_numpy(np.array([image])).float()
    
    image = image.to(device)
    
    output = model.forward(image)
    
    probs = torch.exp(output).data
    
    prob = torch.topk(probs, topk)[0].tolist()[0]
    index = torch.topk(probs, topk)[1].tolist()[0]
    
    indices = []
    for i in range(len(model.class_to_idx.items())):
        indices.append(list(model.class_to_idx.items())[i][0])
        
    label = []
    for i in range(topk):
        label.append(indices[index[i]])
    
    return prob, label

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im = im.resize((256,256))
    value = 0.5*(256-224)
    im = im.crop((value,value,256-value,256-value))
    im = np.array(im)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std

    return im.transpose(2,0,1)

def get_category_names(cat_to_name_path):
    with open(cat_to_name_path, 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name

def map_labels_to_names(label_mapping, labels ,top_k):
    label_names = [''] * top_k
    
    for i in range(top_k):
        label_names[i] = label_mapping[labels[i]]
    
    return label_names

def set_device_to_gpu():
    return 'cuda'

def set_device_to_cpu():
    return 'cpu'

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates parse 
    parser = argparse.ArgumentParser()

    # Creates 3 command line arguments args.dir for path to images files,
    # args.arch which CNN model to use for classification, args.labels path to
    # text file with names of dogs.
    parser.add_argument('input', type=str, default='flowers/test/1/image_06743.jpg', 
                        help='path to image test file')
    parser.add_argument('checkpoint', type=str, default='checkpoint.ph', 
                        help='path to saved checkpoint')
    parser.add_argument('--top_k', type=str, default=1, 
                        help='return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='Set mapping of categories to real names')
    parser.add_argument('--gpu', dest='device', action='store_const',
                    const=set_device_to_gpu, default=set_device_to_cpu,
                    help='sets the device to gpu')

    # returns parsed argument collection
    return parser.parse_args()

if __name__ == "__main__":
    main()