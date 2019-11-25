import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
from os import listdir
import sys
from collections import OrderedDict
from workspace_utils import active_session

def main():
    in_arg = get_input_args()
    
    device = in_arg.device()
    
    data_loaders, image_datasets = get_dataloaders(in_arg.data_directory)
    
    models = get_training_models(int(in_arg.hidden_units), image_datasets, int(in_arg.epochs), float(in_arg.learning_rate))
    
    training_model, checkpoint, optimizer = models[in_arg.arch]
    
    train(data_loaders, training_model, int(in_arg.hidden_units), int(in_arg.epochs), device, optimizer)

    save_checkpoint(in_arg.save_dir, checkpoint)
    
    print('Done')
    

def train(dataloaders, model, hidden_units, epochs, device, optimizer):
    criterion = nn.NLLLoss()

    model.to(device)
    
    steps = 0
    running_loss = 0
    print_every = 40
    
    with active_session():
        for e in range(epochs):
            model.train()

            for images, labels in dataloaders[0]:
                steps += 1

                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    # Make sure network is in eval mode for inference
                    model.eval()

                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        test_loss, accuracy = validation(model, dataloaders[1], criterion, device)

                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(test_loss/len(dataloaders[1])),
                          "Accuracy: {:.3f}".format(accuracy/len(dataloaders[1])))

                    running_loss = 0

                    # Make sure training is back on
                    model.train()

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def save_checkpoint(directory, checkpoint):
    torch.save(checkpoint, directory + '/checkpoint.pth')

def get_training_models(hidden_units, image_datasets, epochs, learning_rate):
    vgg = get_vgg16_model(hidden_units, image_datasets, epochs, learning_rate)
    
    densenet = get_densenet_model(hidden_units, image_datasets, epochs, learning_rate)
    
    models = {'vgg' : vgg, 'densenet': densenet}
    
    return models

def get_vgg16_model(hidden_units, image_datasets, epochs, learning_rate):
    vgg16 = models.vgg16(pretrained=True)
    
    for param in vgg16.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(25088, hidden_units)),
                                ('relu', nn.ReLU()),
                                ('drop', nn.Dropout(p=0.5)),
                                ('fc2', nn.Linear(hidden_units, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                              ]))

    vgg16.classifier = classifier
    
    vgg16.class_to_idx = image_datasets['train'].class_to_idx
        
    optimizer = optim.Adam(vgg16.classifier.parameters(), learning_rate)
        
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'arch': 'vgg16',
                  'batch_size': 64,
                  'epochs': epochs,
                  'classifier': classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': vgg16.state_dict(),
                  'class_to_idx': vgg16.class_to_idx}
    
    return vgg16, checkpoint, optimizer

def get_densenet_model(hidden_units, image_datasets, epochs, learning_rate):
    densenet121 = models.densenet121(pretrained=True)
    
    for param in densenet121.parameters():
        param.requires_grad = False
   
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(1024, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units,102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    densenet121.classifier = classifier
    
    densenet121.class_to_idx = image_datasets['train'].class_to_idx
    
    optimizer = optim.Adam(densenet121.classifier.parameters(), learning_rate)
    
    checkpoint = {'input_size': 1024,
                  'output_size': 2,
                  'arch': 'densenet121',
                  'batch_size': 64,
                  'epochs': epochs,
                  'classifier': classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': densenet121.state_dict(),
                  'class_to_idx': densenet121.class_to_idx}
    
    return densenet121, checkpoint, optimizer

def get_dataloaders(data_directory):
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'
    
    data_transforms = [transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                            std = [0.229, 0.224, 0.225])
                        ]), 
                        transforms.Compose([transforms.Resize(224), 
                                            transforms.CenterCrop(224), 
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                std = [0.229, 0.224, 0.225])])
                      ]

    image_datasets = {'train': datasets.ImageFolder(train_dir, data_transforms[0]),
                      'valid': datasets.ImageFolder(valid_dir, data_transforms[1]),
                      'test':  datasets.ImageFolder(test_dir, data_transforms[1])}

    dataloaders = [torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                      torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
                      torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)]
    
    return dataloaders, image_datasets

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
    parser.add_argument('data_directory', type=str, default='flowers', 
                        help='path to folder of images')
    parser.add_argument('--save_dir', type=str, default='./', 
                        help='path to folder of checkpoints')
    parser.add_argument('--arch', type=str, default='vgg', 
                        help='chosen model')
    parser.add_argument('--learning_rate', type=str, default=0.01,
                        help='learning rate for chosen model')
    parser.add_argument('--hidden_units', type=str, default= 512,
                        help='hidden units for chosen model')
    parser.add_argument('--epochs', type=str, default=3,
                        help='epochs for chosen model')
    parser.add_argument('--gpu', dest='device', action='store_const',
                    const=set_device_to_gpu, default=set_device_to_cpu,
                    help='sets the device to gpu')

    # returns parsed argument collection
    return parser.parse_args()

if __name__ == "__main__":
    main()