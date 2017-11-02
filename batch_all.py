#encoding:utf-8
import batch_one
import batch_one_copy
import truth_console
import evaluate
import os
import logging
import logging.handlers
import numpy as np

if __name__=='__main__':
    wav_path = '../label_data/wav'
    label_path = '../label_data/bin'
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
    for f_wav, f_label in zip(files_wav, files_label):
        print ('='*50)
        contour_truth, contour_fre, frame_all_truth = truth_console.contour_truth(file_name = f_label)
        contour_test, frame_all_test = batch_one_copy.batch_one(inputFile = f_wav, debug = True)
        frame_all = min(frame_all_truth, frame_all_test)
        logger.info('frame_all: {}'.format(frame_all))
        contour_flag , recall , accuracy = evaluate.contour_evaluate(np.array(contour_truth), contour_test)
        average_contour_recall += recall
        average_contour_accuracy += accuracy
        recall_frame, accuracy_frame = evaluate.frame_evaluate(contour_truth, contour_test, frame_all)
        average_frame_recall += recall_frame
        average_frame_accuracy += accuracy_frame
    logger.info('contour_recall: {}'.format(float(average_contour_recall)/float(len(files_wav))))
    logger.info('contour_accuracy: {}'.format(float(average_contour_accuracy)/float(len(files_wav))))
    logger.info('frame_recall: {}'.format(float(average_frame_recall)/float(len(files_wav))))
    logger.info('frame_average: {}'.format(float(average_frame_accuracy)/float(len(files_wav))))





    #print files
