import sys, getopt
# custom
from Datasets.GeneralDataset import GeneralDataset
from Transformers.ToTensor import ToTensor
from torch.utils.data import DataLoader, random_split
from util.load_config import load_config
from load_functions.openpose_json_body25_load_fun import openpose_json_body25_load_fun

# model and loss
from model.ExerciseModel.st_gcn.st_gcn_aaai18 import ST_GCN_18
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss

# other
import numpy as np
import torch, torchvision
from torchvision.transforms import Compose
from datetime import datetime
import matplotlib.pyplot as plt

def batchLabels(dic, labels):
    t = torch.zeros(len(labels), requires_grad=False, dtype=torch.long)
    for i, lab in enumerate(labels):
        t[i] = dic[lab]
    return t

def valueToKey(dic, value):
    ret = []
    for val in value:
        for k, v in dic.items():
            if v == val:
                ret.append(k)
                break
    return ret

def train(model, optimizer, loss_fn, dataset, device, labels):
    losses = []
    model.train()
    for i_batch, sample_batched in enumerate(dataset, 0):
        video = sample_batched["data"].to(device)
        label = batchLabels(labels, sample_batched["label"]).to(device)

        optimizer.zero_grad()

        output = model(video)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        losses.append(loss.data.item())

    return losses

def eval(model, dataset, labels, device):
    model.eval()
    
    labels_len = max(labels.values()) +1

    confusion_matrix = np.zeros((labels_len,labels_len))
    count_no_errors = 0
    mispredictions = []
    for _, sample_batched in enumerate(dataset):
        video = sample_batched["data"].to(device)
        label = batchLabels(labels, sample_batched["label"]).to(device)

        with torch.no_grad():
            output = model(video)
        
        _, predicted_class = torch.max(output, 1)

        results = label - predicted_class

        wrong_indices = torch.flatten(torch.nonzero(results, as_tuple=False)).cpu().numpy()
        predicted_class = predicted_class.cpu().numpy()
        label           = label.cpu().numpy()

        count_no_errors += len(wrong_indices)
        confusion_matrix[predicted_class, label] += 1

        for i in wrong_indices:
            mispredictions.append((valueToKey(labels, [predicted_class[i]]), valueToKey(labels, [label[i]]), sample_batched["name"][i]))

    failure_rate = float(count_no_errors)/len(dataset)

    return failure_rate, confusion_matrix, mispredictions



def main(annotations_path):

    config = load_config("config.json")

    # Hyperparameters
    device      = config["train"]["device"]
    layout      = config["train"]["layout"]
    strategy    = config["train"]["strategy"]
    lr          = config["train"]["lr"]
    gamma       = config["train"]["gamma"]
    momentum    = config["train"]["momentum"]
    decay       = config["train"]["decay"]
    test_split  = config["train"]["test_split"]
    batch_size  = config["train"]["batch_size"]
    epochs      = config["train"]["epochs"]
    no_workers  = config["train"]["no_workers"]

    loss_fn  = CrossEntropyLoss()

    transform = [ToTensor(dtype=torch.float32, requires_grad=False, device="cpu")] # preprocessing done on cpu
    dataset = GeneralDataset(annotations_path, openpose_json_body25_load_fun, transform=Compose(transform))

    print(dataset.annotations["label"].value_counts())
    labels_list = list(set(dataset.annotations["label"]))
    labels_dict = {l: i for i, l in enumerate(labels_list)}
    labels_len = len(labels_list)

    test_len  = int(len(dataset)*test_split)
    train_len = len(dataset)-test_len

    trainset, testset = random_split(dataset, [train_len, test_len])

    dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=no_workers)

    graph_cfg = {"layout":layout, "strategy":strategy}
    model = ST_GCN_18(3, labels_len, graph_cfg, edge_importance_weighting=True, data_bn=True).to(device)


    dataloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
    failure_rate, confusion_matrix, mispredictions = eval(model, dataloader, labels_dict, device)

    exit()

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay, nesterov=True)
    lr_scheduler = StepLR(optimizer, 2, gamma=gamma)
    model.train()

    # training
    losses = []
    mean_loss_per_epoch = []
    for e in range(epochs):
        losses = train(model, optimizer, loss_fn, dataloader, device, labels_dict)
        # TODO: Eval here
        lr_scheduler.step()

        mean_loss = np.mean(losses)
        print("Epoch {}, Mean loss: {}".format(e, mean_loss))
        mean_loss_per_epoch.append(mean_loss)

    print("Mean loss after training: {}".format(np.mean(mean_loss_per_epoch)))

    torch.save(model.state_dict(), "ST_GCN_18.pth")

    # evaluation
    # NOTE: Batch eval currently not supported
    dataloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)
    failure_rate, confusion_matrix, mispredictions = eval(model, dataloader, labels_dict, device)
    print("Failure rate: {}%".format(failure_rate))

    # Print statistics 
    ct = datetime.now()
    current_time = "{}-{}-{}-{}:{}:{}".format(ct.year, ct.month, ct.day, ct.hour, ct.minute, ct.second)
    log_name = "../log/mispredictions_{}.csv".format(current_time)
    with open(log_name, "a") as f:
        f.write("# predicted,correct,filename\n")
        for i in mispredictions:
            f.write("{},{},{}\n".format(i[0], i[1], i[2]))
    
    fig = plt.figure()
    
    ax = fig.add_subplot(212)
    ax.set_title("Confusion Matrix")
    ax.imshow(confusion_matrix)

    classes = []
    for i in range(labels_len):
        classes.append(valueToKey(labels_dict, [i])[0])

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    ax.set_xlabel("Correct Class")
    ax.set_ylabel("Predicted Class")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(classes)):
        for j in range(len(classes)):
            _ = ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="w")
    
    ax2 = fig.add_subplot(211)
    ax2.set_title("Training Loss")
    ax2.plot(mean_loss_per_epoch)

    ax2.set_ylabel("Loss")
    ax2.set_xlabel("50 Steps")


    fig.tight_layout()
    fig.savefig("../doc/statistics.png")

def parseArgs(argv):
    def printHelp():
        print("Start training a ST_GCN_18 network on a dataset. Provide the path to the")
        print("annotations.csv file.")
        print("\n")
        print("Args:")
        print("\n")
        print("    -a, --annotations_path           Path to annotations file. The annotations file must have")
        print("                                     a path to a sample as a first value and label as a second.")
        print("\n")
        print("                                     Example:")
        print("                                     path_to_a_sample1,label")
        print("                                     path_to_a_sample2,label")
        print("\n")
        print("Usage:")
        print("    python train.py -a /path/to/annotations/file")


    annotations_path = ''
    try:
       opts, args = getopt.getopt(argv,"ha:",["annotations_path="])
    except getopt.GetoptError:
       printHelp()
       sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            printHelp()
            sys.exit(0)
        elif opt in ("-a", "--annotations_path"):
            annotations_path = arg
    return annotations_path


if __name__ == "__main__":
    #TODO: Optional, fix the batch labeling
    #TODO: Optional, enable cross validation
    annotations_path = parseArgs(sys.argv[1:])
    main(annotations_path)