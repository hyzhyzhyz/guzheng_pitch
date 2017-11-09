#encoding:utf-8
from  data.data import*
import data.dnn as dnn
import batch_one_copy
import truth_console
import evaluate
import os
import logging
import logging.handlers
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier




if __name__=='__main__':
    wav_path = '../label_data_filter/wav_11.05'
    label_path = '../label_data_filter/bin_11.05'
    LOG_FILE = 'tst.log'
    handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes = 1024*1024, backupCount = 5) #实例化handler
    fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
    
    formatter = logging.Formatter(fmt) #实例化formatter
    handler.setFormatter(formatter) #为handler添加formatter

    logger = logging.getLogger('tst')#获取名为tst的logger
    logger.addHandler(handler) #为logger添加handler
    logger.setLevel(logging.DEBUG)
    #logger.info('first info message: {}'.format(i))
    #logger.debug('first debug message')
    
    average_point_recall = 0#hpcp(shs)之后的召回率
    average_frame_recall = 0
    average_contour_recall = 0
    average_frame_accuracy = 0
    average_contour_accuracy = 0
    files_wav=[os.path.join(wav_path,f) for f in os.listdir(wav_path)
            if os.path.isfile(os.path.join(wav_path,f))]
    files_label = [os.path.join(label_path, f) for f in os.listdir(label_path)
                if os.path.isfile(os.path.join(label_path, f))]
    files_wav.sort()
    files_label.sort()
    logger.info('='*50)
    contour_flag_all = []
    contour_character_all = []
    first = True
    for f_wav, f_label in zip(files_wav, files_label):
        print ('='*50)
        contour_truth, contour_fre, frame_all_truth = truth_console.contour_truth(file_name = f_label)
        contour_test, salience, contour_character, frame_all_test = batch_one_copy.batch_one(inputFile = f_wav, debug = True)
        frame_all = min(frame_all_truth, frame_all_test)
        contour_flag , recall , accuracy = evaluate.contour_evaluate(np.array(contour_truth), contour_test)
        average_contour_recall += recall
        average_contour_accuracy += accuracy
        recall_frame, accuracy_frame = evaluate.frame_evaluate(contour_truth, contour_test, frame_all)
        average_frame_recall += recall_frame
        average_frame_accuracy += accuracy_frame
        average_point_recall += evaluate.pith_evaluate(salience, contour_truth, frame_all)
        if first:
            first = False
            contour_character_all = (contour_character)
            contour_flag_all = (contour_flag)
        else:
            contour_character_all = np.vstack((contour_character_all, contour_character))
            contour_flag_all = np.hstack((contour_flag_all, contour_flag))
    #print contour_flag_all.shape, contour_character_all.shape
    character_name = ('fre_mean', 'fre_std', 'salience_mean', 'salience_std', 'contour_num', 
            'salience_all', 'salience_mean_norm', 'contour_num_continue')
    i=0
    for i in range(8):
        frame = 0
        character_right = []
        character_false = []
        for character, flag in zip(contour_character_all[:,i], contour_flag_all):
            if flag:
                plt.scatter(frame, character, c = 'red', marker = 'o')
                character_right.append(character)
            else:
                plt.scatter(frame, character, c='green', marker = 'x')
                character_false.append(character)
            frame += 1
        plt.xlabel('contour_num')
        plt.ylabel(character_name[i])
        plt.savefig('./picture/character/'+character_name[i]+'.png')
        plt.close()
        fig, axes = plt.subplots(1, 1)
        sns.distplot(character_right, rug = True, hist = False)
        sns.distplot(character_false, rug = True, hist = False)
        plt.savefig('./picture/character/'+character_name[i]+'_pdf.png')
        plt.close()
    logger.info('pitch(shs)_recall: {}'.format(float(average_point_recall)/float(len(files_wav))))
    logger.info('contour_recall: {}'.format(float(average_contour_recall)/float(len(files_wav))))
    logger.info('contour_accuracy: {}'.format(float(average_contour_accuracy)/float(len(files_wav))))
    logger.info('frame_recall: {}'.format(float(average_frame_recall)/float(len(files_wav))))
    logger.info('frame_average: {}'.format(float(average_frame_accuracy)/float(len(files_wav))))
    #-----------------------------分类器logistic regression-------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
            contour_character_all, contour_flag_all, test_size = 0.3, random_state = 0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std =  sc.transform(X_test)
    lr = LogisticRegression(C = 100.0, random_state = 0)
    lr.fit(X_train_std, y_train)
    y_pred = lr.predict(X_test_std)
    final_recall = 0
    final_accuracy = 0
    sum_label = 0
    for i, j in zip(y_test, y_pred):
        if(i==1 and j ==1):
            final_recall += 1
        if i==1:
            sum_label += 1
        if j==1:
            final_accuracy += 1
    print ('recall_logistic: {}'.format(float(final_recall)/float(sum_label)))
    print ('accuracy_logistic: {}'.format(float(final_recall)/float(final_accuracy)))
    logger.info ('recall_logistic: {}'.format(float(final_recall)/float(sum_label)))
    logger.info ('accuracy_logistic: {}'.format(float(final_recall)/float(final_accuracy)))
    #-----------------------------SVM-----------------------------------------------
    #-------------------------------------------------------------------------------
    svm = SVC(kernel = 'linear', C = 1.0, random_state = 0)
    svm.fit(X_train_std, y_train)
    y_pred = svm.predict(X_test_std)
    final_recall_svm = 0
    final_accuracy_svm = 0
    sum_label_svm = 0
    for i, j in zip(y_test, y_pred):
        if(i==1 and j ==1):
            final_recall_svm += 1
        if i==1:
            sum_label_svm += 1
        if j==1:
            final_accuracy_svm += 1
    print ('recall_svm: {}'.format(float(final_recall_svm)/float(sum_label_svm)))
    print ('accuracy_svm: {}'.format(float(final_recall_svm)/float(final_accuracy_svm)))
    logger.info ('recall_svm: {}'.format(float(final_recall_svm)/float(sum_label_svm)))
    logger.info ('accuracy_svm: {}'.format(float(final_recall_svm)/float(final_accuracy_svm)))
    #-----------------------------random forrest-----------------------------------------------
    #------------------------------------------------------------------------------------------
    forest = RandomForestClassifier(criterion = 'entropy',
            n_estimators = 10,
            random_state = 0,
            n_jobs = 2)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test_std)
    final_recall_forest = 0
    final_accuracy_forest = 0
    sum_label_forest = 0
    for i, j in zip(y_test, y_pred):
        if(i==1 and j ==1):
            final_recall_forest += 1
        if i==1:
            sum_label_forest += 1
        if j==1:
            final_accuracy_forest += 1
    print ('recall_forest: {}'.format(float(final_recall_forest)/float(sum_label_forest)))
    print ('accuracy_forest: {}'.format(float(final_recall_forest)/float(final_accuracy_forest)))
    logger.info ('recall_forest: {}'.format(float(final_recall_forest)/float(sum_label_forest)))
    logger.info ('accuracy_forest: {}'.format(float(final_recall_forest)/float(final_accuracy_forest)))
    #-------------------------train dnn-----------------------------------------------
    #-------------------------train dnn-----------------------------------------------
    #---------------------------------------------------------------------------------
    train_dataset = myDataset(X_train_std, y_train)
    train_loader = myDataLoader(train_dataset, batch_size = 8, shuffle = True,
            num_workers = 4, pin_memory = False)
    test_datsset = myDataset(X_test_std, y_test)
    test_loader = myDataLoader(test_datsset, batch_size = 16, shuffle = True,
            num_workers = 4, pin_memory = False)
    dnn.Training(train_loader)
    final_recall_dnn, final_accuracy_dnn = dnn.testing(test_loader)
    print ('recall_dnn: {}'.format(final_recall_dnn))
    print ('accuracy_dnn: {}'.format(final_accuracy_dnn))
    logger.info ('recall_dnn: {}'.format(final_recall_dnn))
    logger.info ('accuracy_dnn: {}'.format(final_accuracy_dnn))
    #------------------------------------------------------------------------
    

    


    #print files

