import torch
from torch.autograd import Variable
from torch import optim
import numpy as np
import argparse
from sklearn.metrics import auc,recall_score,precision_score,precision_recall_curve,accuracy_score,roc_auc_score,roc_curve,matthews_corrcoef
from sklearn import metrics
import shutil
import os
import torch.nn.functional as F
import time
import pandas as pd
from model_my_drop2 import *
import sys
from data_load import *
from model_torch import *

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device = torch.device("cuda:0")

def cal_base(y_true, y_pred):
    y_pred_positive = np.round(np.clip(y_pred, 0, 1))
    y_pred_negative = 1 - y_pred_positive

    y_positive = np.round(np.clip(y_true, 0, 1))
    y_negative = 1 - y_positive

    TP = np.sum(y_positive * y_pred_positive)
    TN = np.sum(y_negative * y_pred_negative)

    FP = np.sum(y_negative * y_pred_positive)
    FN = np.sum(y_positive * y_pred_negative)

    return TP, TN, FP, FN

def specificity(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SP = TN / (TN + FP )
    return SP

def train(model, loss, optimizer, x101_val,x71_val,x41_val, y_val,args):
    x_101 = Variable(x101_val, requires_grad=False).cuda()
    x_71 = Variable(x71_val, requires_grad=False).cuda()
    x_41 = Variable(x41_val, requires_grad=False).cuda()
    y = Variable(y_val, requires_grad=False).cuda()

    model.train()

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x_101,x_71,x_41,args)


    if args.if_bce == 'Y':
        output = loss.forward(F.sigmoid(fx).squeeze().cuda(), y.type(torch.FloatTensor).cuda())
        pred_prob=F.sigmoid(fx)

    else:
        output = loss.forward(fx, y)
        pred_prob = F.log_softmax(fx)


    # Backward
    output.backward()


    #grad_clip
    torch.nn.utils.clip_grad_norm(model.parameters(),args.grad_clip)
    # for p in model.parameters():
    #     p.data.add_(-args.learning_rate, p.grad.data)


    # Update parameters
    optimizer.step()

    return output.item(),pred_prob,list(np.array(y_val)) #cost,pred_probability and true y value


def predict(model, x_41,x_71,x_101, args):
    model.eval() #evaluation mode do not use drop out
    with torch.no_grad():
      x41 = Variable(x_41, requires_grad=False)
      x71 = Variable(x_71, requires_grad=False)
      x101 = Variable(x_101, requires_grad=False)
      n_examples = len(x41)
      out = torch.tensor([]).to(torch.float32).cuda()
      num_batches = n_examples // 2
      for k in range(num_batches):
         start, end = k * 2, (k + 1) * 2
         output = model.forward(x101[start:end],x71[start:end],x41[start:end],args)
         out = torch.cat((out,output),dim=0)
    return out


def save_checkpoint(state,is_best,model_path):
    if is_best:
        print('=> Saving a new best from epoch %d"' % state['epoch'])
        torch.save(state, model_path + '/' + 'm1a_cross_gpu314121_checkpoint-N_enc=3_batch=10-5e-4-101.pth.tar')  

    else:
        print("=> Validation Performance did not improve")


def ytest_ypred_to_file(y_test, y_pred, out_fn):
    with open(out_fn,'w') as f:
        for i in range(len(y_test)):
            f.write(str(y_test[i])+'\t'+str(y_pred[i])+'\n')



if __name__ == '__main__':

    torch.manual_seed(1000)

    parser = argparse.ArgumentParser()   

    # main option
    parser.add_argument("-m", "--mode", action="store", dest='mode', choices=['cnn','cnn-rnn'],default='transformer',  
                        help="mode")

    parser.add_argument("-pos_fa", "--positive_fasta", action="store", dest='pos_fa', default=r'E:\0\DeepM6ASeq-master\data\dr\train_pos.fa',
                        help="positive fasta file")
    parser.add_argument("-neg_fa", "--negative_fasta", action="store", dest='neg_fa', default=r'E:\0\DeepM6ASeq-master\data\dr\train_neg.fa',
                        help="negative fasta file")

    parser.add_argument("-od", "--out_dir", action="store", dest='out_dir',default=r'E:\0\DeepM6ASeq-master\result',
                        help="output directory")

    # cnn option
    parser.add_argument("-fltnum", "--filter_num", action="store", dest='filter_num', default='256-128',
                        help="filter number")
    parser.add_argument("-fltsize", "--filter_size", action="store", dest='filter_size', default='10-5',
                        help="filter size")
    parser.add_argument("-pool", "--pool_size", action="store", dest='pool_size', default=0, type=int,
                        help="pool size")
    parser.add_argument("-cnndrop", "--cnndrop_out", action="store", dest='cnndrop_out', default=0.5, type=float,
                        help="cnn drop out")
    parser.add_argument("-bn", "--if_bn", action="store", dest='if_bn', default='Y',
                        help="if batch normalization")

    # rnn option
    parser.add_argument("-rnnsize", "--rnn_size", action="store", dest='rnn_size', default=32, type=int,
                        help="rnn size")


    # fc option
    parser.add_argument("-fc", "--fc_size", action="store", dest='fc_size', default=0, type=float,
                        help="fully connected size")



    # optimization option
    parser.add_argument("-bce", "--if_bce", action="store", dest='if_bce', default='Y',
                        help="if use BCEloss function")
    parser.add_argument("-maxepc", "--max_epochs", action="store", dest='max_epochs', default=100, type=int,
                        help="max epochs")
    parser.add_argument("-lr", "--learning_rate", action="store", dest='learning_rate', default=5e-4, type=float,
                        help="learning rate")
    parser.add_argument("-lrstep", "--lr_decay_step", action="store", dest='lr_decay_step', default=10, type=int,
                        help="learning rate decay step")
    parser.add_argument("-lrgamma", "--lr_decay_gamma", action="store", dest='lr_decay_gamma', default=0.5, type=float,
                        help="learning rate decay gamma")
    parser.add_argument("-gdclp", "--grad_clip", action="store", dest='grad_clip', default=5, type=int,
                        help="gradient clip value magnitude")
    parser.add_argument("-patlim", "--patience_limit", action="store", dest='patience_limit', default=50, type=int,
                        help="patience")
    parser.add_argument("-batch", "--batch_size", action="store", dest='batch_size', default=10, type=int,
                        help="batch size")




    args = parser.parse_args()    

    #load m6A data

    #data_path = args.data_dir
    pos_train_fa = args.pos_fa
    neg_train_fa = args.neg_fa

    wordvec_len=4


    X101_train = np.load(r'E:\0\m1A20210206\word2vec_m1A-31-cz-593-5930_100d_f_train_wd_q.dat', allow_pickle=True)
    X101_train = torch.from_numpy(X101_train).to(torch.float32)
    X71_train = np.load(r'E:\0\m1A20210206\word2vec_m1A-41-cz-593-5930_100d_f_train_wd_q.dat', allow_pickle=True)
    X71_train = torch.from_numpy(X71_train).to(torch.float32)
    X41_train = np.load(r'E:\0\m1A20210206\word2vec_m1A-21-cz-593-5930_100d_f_train_wd_q.dat', allow_pickle=True)
    X41_train = torch.from_numpy(X41_train).to(torch.float32)
    y_train = np.load(r'E:\0\m1A20210206\word2vec_m1A-71-cz-593-5930_100d_l_train_wd_q.dat', allow_pickle=True)
    y_train = torch.from_numpy(y_train).to(torch.float32)
    X101_test = np.load(r'E:\0\m1A20210206\word2vec_m1A-31-cz-114-1140_100d_f_test_wd_q.dat', allow_pickle=True)
    X101_test = torch.from_numpy(X101_test).to(torch.float32)
    X71_test = np.load(r'E:\0\m1A20210206\word2vec_m1A-41-cz-114-1140_100d_f_test_wd_q.dat', allow_pickle=True)
    X71_test = torch.from_numpy(X71_test).to(torch.float32)
    X41_test = np.load(r'E:\0\m1A20210206\word2vec_m1A-21-cz-114-1140_100d_f_test_wd_q.dat', allow_pickle=True)
    X41_test = torch.from_numpy(X41_test).to(torch.float32)
    y_test = np.load(r'E:\0\m1A20210206\word2vec_m1A-71-cz-114-1140_100d_l_test_q.dat', allow_pickle=True)

    # y_test = torch.from_numpy(y_test).to(torch.float32)
    model_dir_base = args.filter_num + '_' + args.filter_size + '_' + str(
                args.pool_size) + '_' + str(args.cnndrop_out) \
                        + '_' + args.if_bn + '_' +args.if_bce + '_' +\
        str(args.fc_size) + '_' + str(args.learning_rate) + '_' + str(args.batch_size)

    n_classes = 2
    n_examples = len(X101_train)
    loss = torch.nn.CrossEntropyLoss(size_average=False)  

    if args.if_bce == 'Y':  
        n_classes = n_classes -1
        loss = torch.nn.BCELoss(size_average=False)


    if args.mode == 'cnn':
        model = ConvNet(output_dim=n_classes, args=args,wordvec_len=wordvec_len)
        model_dir = args.mode + '/' + model_dir_base

    if args.mode == 'cnn-rnn':
        model = ConvNet_BiLSTM(output_dim=n_classes, args=args,wordvec_len=wordvec_len)
        model_dir = args.mode + '/' +model_dir_base +'_'+str(args.rnn_size)

    if args.mode == 'cnn1':
        model = ConvNet1()
        model_dir = args.mode + '/' + model_dir_base + '_' + str(args.rnn_size)

    if args.mode == 'transformer':
        model = TransformerModel().cuda()
        model_dir = args.mode + '/' + model_dir_base + '_' + str(args.rnn_size)

    model_path = args.out_dir + '/' + model_dir
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    print("> model_dir:",model_path)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    batch_size = args.batch_size
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)


    best_acc=0
    best_auc=0
    patience=0


    for i in range(args.max_epochs):

        start_time = time.time()
        scheduler.step()   

        cost = 0.
        y_pred_prob_train = []
        y_batch_train = []

        num_batches = n_examples // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            output_train, y_pred_prob, y_batch = train(model, loss, optimizer, X101_train[start:end],X71_train[start:end],X41_train[start:end] ,y_train[start:end], args)
            cost += output_train

            prob_data = y_pred_prob.data.cpu().numpy()
            if args.if_bce == 'Y':
                for m in range(len(prob_data)):
                    y_pred_prob_train.append(prob_data[m][0])

            else:
                for m in range(len(prob_data)):
                    y_pred_prob_train.append(np.exp(prob_data)[m][1])

            y_batch_train += y_batch




        #train AUC
        fpr_train, tpr_train, thresholds_train = roc_curve(y_batch_train, y_pred_prob_train)



        #predict test  x_41,x_71,x_101,
        output_test = predict(model, X41_test.cuda(),X71_test.cuda(),X101_test.cuda(),args)
        y_pred_prob_test = []

        if args.if_bce == 'Y':
            y_pred_test=[]
            prob_data=F.sigmoid(output_test).data.cpu().numpy()
            for m in range(len(prob_data)):
                y_pred_prob_test.append(prob_data[m][0])
                if prob_data[m][0]>=0.5:
                    y_pred_test.append(1)
                else:
                    y_pred_test.append(0)
        else:
            y_pred_test=output_test.data.numpy().argmax(axis=1)
            prob_data =F.log_softmax(output_test).data.numpy()
            for m in range(len(prob_data)):
                y_pred_prob_test.append(np.exp(prob_data)[m][1])


        #test AUROC
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_prob_test)

        end_time = time.time()
        hours, rem = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        # 100. *

        print("Epoch %d, cost = %f, AUROC_train = %0.3f, acc = %.4f%%, AUROC_test = %0.3f"
              % (i + 1, cost / num_batches, auc(fpr_train, tpr_train),100. * np.mean(y_pred_test == y_test),auc(fpr_test, tpr_test)))
        print("time cost: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))





        cur_auc = auc(fpr_test, tpr_test)
        is_best = bool(cur_auc >= best_auc)
        best_auc = max(cur_auc, best_auc)
 


        #patience
        if not is_best:
            patience+=1
            if patience>=args.patience_limit:
                break

        else:
            patience=0




        if is_best:

            save_checkpoint({
                'epoch': i + 1,
                'state_dict': model.state_dict(),
                'best_auroc': best_auc,
                'optimizer': optimizer.state_dict()
            }, is_best, model_path)

            ytest_ypred_to_file(y_batch_train, y_pred_prob_train,
                                model_path + '/' + 'm1A_cross-attention-gpu314121-predout_train.tsv')

            ytest_ypred_to_file(y_test, y_pred_prob_test,
                                model_path + '/' + 'm1A_cross-attention-gpu314121-predout_val.tsv')

            a0 = y_test

            f = open("m1A_gpu314121_word2vec_101-N_enc=3_batch=5-5e-4-100d+cross-attention-mean.txt", 'w')
            for i in range(0, len(a0)):
                f.write(np.str(a0[i]))
                f.write('\n')
            f.close()

            # a = label_pre
            a = y_pred_prob_test
            f = open("m1A_gpu314121_word2vec_101-N_enc=3_batch=5-5e-4-100d+cross-attention-mean_pre.txt", 'w')
            # for i in range(0,long):
            for i in range(0, len(a)):
                f.write(np.str(a[i]))
                f.write('\r')
            f.close()

            label_predict = [0 if item <= 0.5 else 1 for item in y_pred_prob_test]
            y_pred_classes = []  
            y_pred_classes.extend(list(label_predict))  

            Y_tests = y_test
            y_preds = y_pred_prob_test

            print("AUROC: %f " % roc_auc_score(Y_tests, y_preds))
            print("ACC:  %f " % accuracy_score(Y_tests, y_pred_classes))
            Sen = metrics.recall_score(Y_tests, y_pred_classes)  
            print('Sen', round(Sen * 100, 2))
            recall = metrics.recall_score(Y_tests, y_pred_classes)  
            print('recall', round(recall * 100, 2))
            precision = precision_score(Y_tests, y_pred_classes)
            print("Precision: ", round(precision * 100, 2))
            Mcc = matthews_corrcoef(Y_tests, y_pred_classes)  
            print('Mcc', round(Mcc * 100, 2))
            Spe = specificity(Y_tests, y_pred_classes)
            print("specificity: ", round(Spe * 100, 2))
            precision, recall, thresholds = precision_recall_curve(Y_tests, y_preds)
            prc_auc = auc(recall, precision)  
            print("AUPRC:", round(prc_auc, 4))


            result = np.zeros(8)
            result[0] = roc_auc_score(Y_tests, y_preds)  
            result[1] = accuracy_score(Y_tests, y_pred_classes)  
            result[2] = Sen
            result[3] = Spe
            result[4] = Mcc
            precision = precision_score(Y_tests, y_pred_classes)
            result[5] = round(precision * 100, 2)
            precision, recall, thresholds = precision_recall_curve(Y_tests, y_preds)
            result[6] = round(auc(recall, precision), 4)  
            cost_time = end_time - start_time
            result[7] = cost_time
            now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
            df = pd.DataFrame(result)
            writer = pd.ExcelWriter("E:/0/wd/0718/DeepM6ASeq-master-word2vec/DeepM6ASeq-master-word2vec/result/" + now + r"m1A_gpu314121_word2vec_101-N_enc=3_batch=5-5e-4-100d+cross-attention.xlsx") #m1A_word2vec_101-N_enc=3_batch=5-5e-4-100d+cross-attention
            df.to_excel(writer, 'page_1', float_format='%.5f')
            writer.save()

    print('> best auc:',best_auc)