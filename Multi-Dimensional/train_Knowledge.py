import os
import math
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from my_dataset import MyDataSet
from vit_model_Knowledge import vit_student as student_model
from vit_model_Knowledge import vit_teacher as teacher_model
from utils import read_split_data, train_one_epoch_Knowledge, evaluate_Knowledge


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     #transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}


    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])


    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model_student = student_model(num_classes=args.num_classes,
                         has_logits=False).to(device)

    model_teacher = teacher_model(num_classes=args.num_classes,
                         has_logits=False).to(device)


    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict_student = torch.load(args.weights, map_location=device)

        del_keys = []
        if model_teacher.has_logits:
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

    pg = [p for p in model_student.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Pre-train the weight path (Teacher)，
    weights_dict_teacher = torch.load(
        "",
        map_location=device)
    del_keys_teacher = []
    if model_teacher.has_logits:
        del_keys_teacher = ['head.weight', 'head.bias']
    for k in del_keys_teacher:
        del weights_dict_teacher[k]
    print(model_teacher.load_state_dict(weights_dict_teacher, strict=False))
    for param in model_teacher.parameters():
        param.requires_grad = False

    top_acc = 0.0
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc_student, train_acc_teacher  = train_one_epoch_Knowledge(model_student=model_student,
                                                          model_teacher=model_teacher,
                                                          optimizer=optimizer,
                                                          data_loader=train_loader,
                                                          device=device,
                                                          epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate_Knowledge(model=model_student,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc_student", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc_student, epoch)
        tb_writer.add_scalar(tags[1], train_acc_teacher, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if ((val_acc + train_acc_student)/2) > top_acc:
            torch.save(model_student.state_dict(),"./weights/" +  "_best" + ".pth")
            top_acc = (val_acc + train_acc_student)/2

        torch.save(model_student.state_dict(),"./weights/" + " {}.pth".format(epoch))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4) #定义batch-size
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # The root directory of the data set
    parser.add_argument('--data-path', type=str,
                        default="")
    parser.add_argument('--model-name', default='', help='create model name')

    # Pre-train the weight path(student)，Set to null if you do not want to load，
    parser.add_argument('--weights', type=str, default="",
                        help='initial weights path')
    # Freeze weight or not
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
