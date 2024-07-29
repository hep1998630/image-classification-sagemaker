#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torchvision.transforms as T
from zipfile import ZipFile 
import os
import argparse
import json
import shutil
from io import BytesIO
import base64
from PIL import Image

try:
    import sagemaker
    import boto3
    from smdebug import modes
    from smdebug.profiler.utils import str2bool
    from smdebug.pytorch import get_hook, Hook
    from sagemaker.s3 import S3Downloader
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    s3 = boto3.resource('s3')  # assumes credentials & configuration are handled outside python in .aws directory or environment variables
except ImportError:
    print("Import error")
    pass 
    
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Predictor():

    def __init__(self, model, n_classes=133):
        super().__init__()
        d = {"mean": torch.tensor([0.4870, 0.4665, 0.3972]),
             "std": torch.tensor([0.2636, 0.2572, 0.2639])}

        h_w = {"mean_h": torch.tensor(532.1574850299402), 
               "mean_w": torch.tensor(571.3823353293413)}
        # model.cuda()

        self.model = model.eval()


        # is compatible with torch.script, while the latter isn't
        self.transforms = T.Compose([T.ToTensor(), 
                                     T.Resize((int(h_w["mean_h"]),int(h_w["mean_w"]))),
                                    T.Normalize(d["mean"], d["std"])])

    def predict_cls(self, image):
        
        image = self.transforms(image.copy())
        image = torch.unsqueeze(image, 0)

        # set the module to evaluation mode
        with torch.no_grad():


            # move data to GPU
            if torch.cuda.is_available():
                image = image.cuda()

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            logits = self.model(image) 

        return logits.argmax(dim=1, keepdim=True) 


def model_fn(model_dir):
    """
    Load the model for inference
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = net(133)
    model.load_state_dict(torch.load(model_dir + "/efficientnet.pt", map_location=device))
    predictor = Predictor(model)
    # else:
    #     model.load_state_dict(torch.load(model_dir + "/efficientnet.pt"))
    # model.eval()
    return predictor


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """

    if request_content_type == "application/json" or request_content_type == "image/png":
        data = base64.b64decode(request_body, validate= True)

        im = Image.open(BytesIO(data))
        # im.save('image1.png', 'PNG')

        return im


def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """

    with torch.no_grad():
        return model.predict_cls(input_data)



def test(model, test_loader, criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    test_loss = 0
    correct = 0
    model = model.to(device) 
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f} \n".format(
            test_loss,
        )
    )


def train(model, train_loader, criterion, optimizer, epoch):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    pass
    model = model.to(device)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    return model


def net(n_classes):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    pass
    model = models.efficientnet_b2(weights="DEFAULT")
    n_inputs = model.classifier[1].in_features

    # Change the number of classes in the last layer 
    model.classifier[1] = nn.Sequential(nn.Linear(n_inputs, n_classes),)
    for param in model.parameters():
        param.requires_grad = True

    # model.classifier = nn.Sequential(nn.Linear(1408,512),
    #                        nn.ReLU(),
    #                        nn.Dropout(p=0.4),
    #                        nn.Linear(512,128),
    #                        nn.ReLU(),
    #                        nn.Dropout(p=0.4),
    #                        nn.Linear(128,n_classes))

    return model




def get_data(s3_path, local_path="./Data"):

    S3Downloader.download(s3_path, "./")
    with ZipFile("./dogImages.zip", 'r') as zObject:  
        # Extracting all the members of the zip 
        # into a specific location. 
        zObject.extractall(path=local_path)


def create_data_loaders(data_path, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass

    d = {"mean": torch.tensor([0.4870, 0.4665, 0.3972]),
         "std": torch.tensor([0.2636, 0.2572, 0.2639])}

    h_w = {"mean_h": torch.tensor(532.1574850299402), 
           "mean_w": torch.tensor(571.3823353293413)}
    transforms = {"train": None, "val": None}
    transforms['train'] = T.Compose([T.Resize((int(h_w["mean_h"]), int(h_w["mean_w"]))),

                # take a random part of the image
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(brightness=.5, hue=.3),
                T.RandomRotation(90), 
                T.RandomPerspective(distortion_scale=0.4, p=0.5),
                T.ToTensor(),
                T.Normalize(d["mean"], d["std"])])
    transforms['val']= T.Compose([T.Resize((int(h_w["mean_h"]),int(h_w["mean_w"]))),
                        T.ToTensor(), 
                        T.Normalize(d["mean"], d["std"])])

    train_img_folder = ImageFolder(root=os.path.join(data_path, "train"), transform=transforms['train'])

    test_img_folder = ImageFolder(root=os.path.join(data_path, "test"), transform=transforms['val'])


    train_loader = torch.utils.data.DataLoader(train_img_folder, batch_size=batch_size,num_workers=1, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_img_folder, batch_size= batch_size,num_workers= 1)

    return train_loader, test_loader


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model = net(n_classes=133)  # Number of clases in the dogs breed dataset 

    '''
    TODO: Create your loss and optimizer
    '''
    hook = Hook.create_from_json_file()
    hook.register_hook(model)
    # hook = get_hook(create_if_not_exists=True)
    loss_criterion = nn.CrossEntropyLoss()
    lr = args.lr
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    if hook:
        hook.register_loss(loss_criterion)
    # Get data from S3 
    sess = sagemaker.Session(boto3.session.Session())
    bucket = sess.default_bucket()

    get_data("s3://sagemaker-us-east-1-471112614246/course4-data/dogImages.zip")

    dataset_path = "./Data/dogImages"
    batch_size = args.batch_size
    train_loader, test_loader = create_data_loaders(dataset_path, batch_size)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''

    for epoch in range(1, args.epochs + 1):
        if hook:
            hook.set_mode(modes.TRAIN)
        model = train(model, train_loader, loss_criterion, optimizer, epoch)
        '''
        TODO: Test the model to see its accuracy
        '''
        if hook:
            hook.set_mode(modes.EVAL)
        test(model, test_loader, loss_criterion)

    '''
    TODO: Save the trained model
    '''
    model_name = "/efficientnet.pt"
    # torch.save(model.state_dict(), path)
    torch.save(model.state_dict(), args.model_dir + model_name)
    # PyTorch requires that the inference script must
    # be in the .tar.gz model file and Step Functions SDK doesn't do this.
    inference_code_path = args.model_dir + "/code/"

    if not os.path.exists(inference_code_path):
        os.mkdir(inference_code_path)
        print("Created a folder at {}!".format(inference_code_path))

    shutil.copy("train_model.py", inference_code_path)
    # shutil.copy("requirements.txt", inference_code_path)
    print("Saving models files to {}".format(inference_code_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )

    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    args = parser.parse_args()
    
    main(args)




