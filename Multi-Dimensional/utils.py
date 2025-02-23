import os
import sys
import json
import random
import torch
from tqdm import tqdm
from torch.nn import functional as F



def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."
    return train_images_path, train_images_label, val_images_path, val_images_label


def read_split_data_evaluation(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    eval_images_path = []
    eval_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]
        every_class_num.append(len(images))

        for img_path in images:
            eval_images_path.append(img_path)
            eval_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for evalution.".format(len(eval_images_path)))


    return  eval_images_path, eval_images_label



def ChannelAttention(featmap): #Channel_weight
    n, c, h, w = featmap.shape
    featmap = featmap.reshape((n, c, -1))
    featmap = featmap.softmax(dim=-1)
    return featmap



def SpatialAttention(featmap): # Spatial_weight
    avg_values = torch.mean(featmap, dim=1, keepdim=True)
    return avg_values


def CBAKD_loss(student_featmap, teacher_featmap, temperature=1.0):
    n, c, h, w = student_featmap.shape
    CA_s = ChannelAttention(student_featmap/temperature)
    CA_t = ChannelAttention(teacher_featmap/temperature)
    CA_loss = torch.nn.KLDivLoss(reduction='sum')(CA_s.log(), CA_t)
    CA_loss /= n * c
    SA_s = SpatialAttention(student_featmap)
    avg_s = torch.sigmoid(SA_s)
    SA_t = SpatialAttention(teacher_featmap)
    avg_t = torch.sigmoid(SA_t)
    avg_loss = torch.nn.MSELoss(reduction='sum')(avg_s, avg_t)
    SA_loss = avg_loss
    return CA_loss,SA_loss


def Class_loss(student_cls_token,teacher_cls_token):
    B,C = student_cls_token.shape
    cls_loss = torch.nn.KLDivLoss(reduction='sum')(student_cls_token.softmax(dim=-1).log(), teacher_cls_token.softmax(dim=-1))
    return cls_loss/B



def train_one_epoch_Knowledge(model_student,model_teacher, optimizer, data_loader, device, epoch):
    model_student.train()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    accu_num_teacher = torch.zeros(1).to(device)
    accu_teacher = 0.0
    accu_student = 0.0
    optimizer.zero_grad()
    student_loss = torch.nn.CrossEntropyLoss()
    distilation_loss = torch.nn.MSELoss()
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)


    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        #hard_pred_student, feature_student = model_student(images.to(device))
        hard_pred_student, student_featmap, student_cls_token = model_student(images.to(device))
        soft_pred_student = F.softmax(hard_pred_student/1, dim=1)

        with torch.no_grad():
            accu_num_teacher, soft_pred_teacher, pred_classes_teacher,teacher_featmap,teacher_cls_token = evaluate_teacher_model(model=model_teacher,
                                                                                               images=images,
                                                                                               labels=labels,
                                                                                               device=device,
                                                                                               accu_num=accu_num_teacher)


        CA_loss,SA_loss = CBAKD_loss(student_featmap=student_featmap,teacher_featmap=teacher_featmap,temperature=1.0)
        loss1 = CA_loss+SA_loss
        loss2 = distilation_loss(soft_pred_student,soft_pred_teacher)
        cls_loss = Class_loss(student_cls_token,teacher_cls_token)
        pred_classes = torch.max(hard_pred_student, dim=1)[1]
        loss0 = student_loss(hard_pred_student, labels.to(device))
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss =0.6*loss0+0.4*(loss1+loss2+cls_loss)
        loss.backward()
        accu_loss += loss.detach()
        accu_teacher = accu_num_teacher.item() / sample_num
        accu_student = accu_num.item() / sample_num

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc_student: {:.3f}, acc_teacher: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_student,
                                                                               accu_teacher)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()


    return accu_loss.item() / (step + 1), accu_student, accu_teacher



@torch.no_grad()
def evaluate_Knowledge(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred, _ , class_token = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate_teacher_model(model, images,labels, device,accu_num):

    model.eval()
    hard_pred_teacher, teacher_featmap, teacher_cls_token = model(images.to(device))
    soft_pred_teacher = F.softmax(hard_pred_teacher / 1, dim=1)
    pred_classes_teacher = torch.max(hard_pred_teacher, dim=1)[1]
    accu_num += torch.eq(pred_classes_teacher, labels.to(device)).sum()
    return  accu_num,soft_pred_teacher,pred_classes_teacher,teacher_featmap,teacher_cls_token