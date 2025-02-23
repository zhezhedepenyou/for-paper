import torch
import argparse
import os
import json
import sys
import time
from ptflops import get_model_complexity_info
from tqdm import tqdm
from torchvision import transforms
from my_dataset import MyDataSet
from vit_model_Knowledge import vit_student as student_model
from utils import read_split_data_evaluation
from sklearn.metrics import recall_score, precision_score,fbeta_score
from Confusion_Matrix import ConfusionMatrix



def evaluate_Knowledge(model, data_loader, device, batch_size, num_classes):

    model.eval()
    accu_num = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    pre_list = []
    label_list = []

    # Confusion matrix
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=num_classes, labels=labels, normalize=False, batch_size=batch_size)
    start_time = time.time()

    for step, data in enumerate(data_loader):
        images, labels, name = data
        sample_num += images.shape[0]

        pred, _ , class_token = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        B = pred_classes.shape

        if step == 0:
            start_time = time.time()
        for i in range(B[0]):
            pre_list.append(pred_classes[i].item())
            label_list.append(labels[i].item())

        confusion.update(pred_classes.to("cpu").numpy(), labels.to("cpu").numpy())

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        data_loader.desc = "[eval_acc: {:.3f}".format(accu_num.item() / sample_num)
    end_time = time.time()
    total_time = end_time - start_time
    fps = len(label_list) / total_time

    confusion.plot()
    confusion.summary()


    return  accu_num.item() / sample_num,pre_list,label_list,confusion,fps





def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_transform = {
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    eval_images_path, eval_images_label = read_split_data_evaluation(args.data_path)

    eval_dataset = MyDataSet(images_path=eval_images_path,
                            images_class=eval_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=eval_dataset.collate_fn)

    model_student = student_model(num_classes=args.num_classes,
                                  has_logits=False).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict_student = torch.load(args.weights, map_location=device)
        del_keys = []
        if model_student.has_logits:
            del_keys = ['head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict_student[k]
        print(model_student.load_state_dict(weights_dict_student, strict=False))

    if args.freeze_layers:
        for name, para in model_student.named_parameters():
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    val_acc,pre_list,label_list,confusion,fps = evaluate_Knowledge(model=model_student,
                                                                   data_loader=eval_loader,
                                                                   device=device, batch_size=batch_size, num_classes=args.num_classes)

    macro_precision = precision_score(label_list, pre_list, average='macro')
    macro_recall = recall_score(label_list, pre_list, average='macro')
    F2_score = fbeta_score(label_list, pre_list, beta=2, average='macro')

    confusion.plot()
    print("\nval_acc={}".format(val_acc))
    print("macro_precision={}".format(macro_precision))
    print("macro_recall={}".format(macro_recall))
    print("F2_score_recall={}\n".format(F2_score))

    flops, params = get_model_complexity_info(model_student, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)
    print("FPS={}Hz\n".format(fps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # The root directory of the data set
    parser.add_argument('--data-path', type=str,
                        default="")
    parser.add_argument('--model-name', default='', help='create model name')

    # Pre-train the weight path，Set to null if you do not want to load，
    parser.add_argument('--weights', type=str, default="",
                        help='initial weights path')

    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)