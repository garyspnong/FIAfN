import numpy as np
from time import time
from model import FIAfN
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--is_save', type=bool, default=False) 
    parser.add_argument('--greater_is_better', type=bool, default=False, help='early stop criterion')
    parser.add_argument('--has_residual', type=bool, default=True, help='add residual')

    parser.add_argument('--blocks', type=int, default=1, help='#blocks')
    parser.add_argument('--block_shape', default=[64,64,64], help='output shape of each block')
    parser.add_argument('--heads', type=int, default=2, help='#heads') 
    parser.add_argument('--embedding_size', type=int, default=16) 
    parser.add_argument('--dropout_keep_prob', default=[0.6, 0.9, 0.5]) 
    parser.add_argument('--epoch', type=int, default=50) 
    parser.add_argument('--batch_size', type=int, default=512) 
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer_type', type=str, default='adam') 
    parser.add_argument('--l2_reg', type=float, default=0.0) 
    parser.add_argument('--random_seed', type=int, default=2018) 
    parser.add_argument('--save_path', type=str, default='./model/') 
    parser.add_argument('--field_size', type=int, default=8, help='#fields') 
    parser.add_argument('--loss_type', type=str, default='logloss')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--run_times', type=int, default=10,help='run multiple times to eliminate error')
    parser.add_argument('--deep_layers',default=[200,200], help='config for dnn in joint train')
    parser.add_argument('--batch_norm', type=int, default=0)
    parser.add_argument('--batch_norm_decay', type=float, default=0.995)
    #parser.add_argument('--data', type=str, help='data name')
    parser.add_argument('--data_path', type=str, default='../data', help='root path for all the data')
    return parser.parse_args()



def _run_(args, run_cnt):
    path_prefix = args.data_path
    #feature_size = np.load(path_prefix + '/feature_size.npy')[0]
    feature_size = 3600

    # test: file1, valid: file2, train: file3-10 for criteo
    model = FIAfN(args=args, feature_size=feature_size, run_cnt=run_cnt)

    Xi_valid = np.load(path_prefix + '/valid_i_other.npy')
    Xv_valid = np.load(path_prefix + '/valid_x_other.npy')
    Xi_valid_title = np.load(path_prefix + '/valid_i_title.npy')
    Xv_valid_title = np.load(path_prefix + '/valid_x_title.npy')
    Xi_valid_genre = np.load(path_prefix + '/valid_i_genre.npy')
    Xv_valid_genre = np.load(path_prefix + '/valid_x_genre.npy')    
    y_valid = np.load(path_prefix + '/valid_y.npy')

    is_continue = True
    for k in range(model.epoch):
        if not is_continue:
            print('early stopping at epoch %d' % (k+1))
            break
        time_epoch = 0
        for j in range(1):
            if not is_continue:
                print('early stopping at epoch %d' % (k+1))
                break
            Xi_train = np.load(path_prefix + '/train_i_other.npy')
            Xv_train = np.load(path_prefix + '/train_x_other.npy')
            Xi_train_title = np.load(path_prefix + '/train_i_title.npy')
            Xv_train_title = np.load(path_prefix + '/train_x_title.npy')
            Xi_train_genre = np.load(path_prefix + '/train_i_genre.npy')
            Xv_train_genre = np.load(path_prefix + '/train_x_genre.npy')
            y_train = np.load(path_prefix + '/train_y.npy')

            t1 = time()
            is_continue = model.fit_once(Xi_train, Xv_train, Xi_train_title,Xv_train_title,Xi_train_genre, Xv_train_genre, y_train, k+1,
                      Xi_valid, Xv_valid,Xi_valid_title,Xv_valid_title, Xi_valid_genre, Xv_valid_genre, y_valid, early_stopping=True)
            time_epoch += time() - t1

        print("epoch %d, time %d" % (k+1, time_epoch))


    print('\n.........start testing!.......')
    Xi_test = np.load(path_prefix + '/test_i_other.npy')
    Xv_test = np.load(path_prefix + '/test_x_other.npy')
    Xi_test_title = np.load(path_prefix + '/test_i_title.npy')
    Xv_test_title = np.load(path_prefix + '/test_x_title.npy')
    Xi_test_genre = np.load(path_prefix + '/test_i_genre.npy')
    Xv_test_genre = np.load(path_prefix + '/test_x_genre.npy')
    y_test = np.load(path_prefix + '/test_y.npy')    

    model.restore()

    test_result, test_loss = model.evaluate(Xi_test, Xv_test,Xi_test_title, Xv_test_title, Xi_test_genre, Xv_test_genre, y_test)
    print("test-result = %.4lf, test-logloss = %.4lf\n" % (test_result, test_loss))
    return test_result, test_loss

if __name__ == "__main__":
    args = parse_args()
    print(args.__dict__)
    print('**************')
    test_auc = []
    test_log = []

    print('run time : %d' % args.run_times)
    for i in range(1, args.run_times + 1):
        test_result, test_loss = _run_(args, i)
        test_auc.append(test_result)
        test_log.append(test_loss)
    print('test_auc', test_auc)
    print('test_log_loss', test_log)
    print('avg_auc', sum(test_auc)/len(test_auc))
    print('avg_log_loss', sum(test_log)/len(test_log))