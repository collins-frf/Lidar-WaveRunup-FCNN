from network import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import *
import glob
import imageio
import argparse
import sys
from oct2py import octave
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.interpolate import griddata
import scipy.stats
import scipy.interpolate as interp
import astropy
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import ManualInterval, SqrtStretch
from astropy.convolution import interpolate_replace_nans, convolve
from astropy.convolution import Gaussian2DKernel, Gaussian1DKernel
import mat73
np.set_printoptions(threshold=sys.maxsize)
np.seterr(all="ignore")


def index2d(CalcRunupCalc, i):
    pred2d = np.ndarray((len(CalcRunupCalc[i]['lineCalcRunupCalc']['downLineIndex'][0][0][0]), 1024))
    if len(CalcRunupCalc[i]['lineRunupCalc']['downLineIndex'][0][0]) == 1:
        label2d = np.ndarray((len(CalcRunupCalc[i]['lineRunupCalc']['downLineIndex'][0][0][0]), 1024))
    else:
        label2d = np.ndarray((len(CalcRunupCalc[i]['lineRunupCalc']['downLineIndex'][0][0]), 1024))
    for j in range(len(CalcRunupCalc[i]['lineCalcRunupCalc']['downLineIndex'][0][0][0])):
        pred2d[j, :(CalcRunupCalc[i]['lineCalcRunupCalc']['downLineIndex'][0][0][0][j])] = 0
        pred2d[j, (CalcRunupCalc[i]['lineCalcRunupCalc']['downLineIndex'][0][0][0][j]):] = 1
        try:
            label2d[j, :(CalcRunupCalc[i]['lineRunupCalc']['downLineIndex'][0][0][j][0])] = 0
            label2d[j, (CalcRunupCalc[i]['lineRunupCalc']['downLineIndex'][0][0][j][0]):] = 1
        except:
            label2d[j, :(CalcRunupCalc[i]['lineRunupCalc']['downLineIndex'][0][0][0, j])] = 0
            label2d[j, (CalcRunupCalc[i]['lineRunupCalc']['downLineIndex'][0][0][0, j]):] = 1
    diff2d = pred2d - label2d
    #plt.subplot(1, 3, 1), plt.imshow(pred2d)
    #plt.subplot(1, 3, 2), plt.imshow(label2d)
    #plt.subplot(1, 3, 3), plt.imshow(diff2d)
    #plt.show()

    return pred2d, label2d, diff2d


def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))


def calc_CalcRunup():
    test_fnamelist = glob.glob('./MATLAB files/calculateRunupLine/K*')
    test_fnamelist.sort(key=natural_keys)
    print(test_fnamelist)

    CalcRunupCalc = []
    for i, filename in enumerate(test_fnamelist):
        print(test_fnamelist[i])
        CalcRunupCalc.append(loadmat(test_fnamelist[i]))
    Krmslist = []
    Kmaelist = []
    Kdifferencelist = []
    Kmeanmaelist = []
    Kmeandifferencelist = []

    for i in range(len(test_fnamelist)):
        print(test_fnamelist[i])
        try:
            Kpredictions, Klabels, Kdifference = index2d(CalcRunupCalc, i)
            Kconfidence = CalcRunupCalc[i]['lineCalcRunupCalc']['zDiffCumulative'][0][0]
            Kconfidence = Kconfidence[:, :1024]
            x, y = np.indices(Kconfidence.shape)
            interp = np.array(Kconfidence)
            interp[np.isnan(interp)] = griddata((x[~np.isnan(interp)], y[~np.isnan(interp)]),
                                                interp[~np.isnan(interp)],
                                                (x[np.isnan(interp)], y[np.isnan(interp)]))
            Kconfidence = interp
            Kpredictions = cv2.resize(Kpredictions, (512, pred_length), interpolation=cv2.INTER_AREA)
            Klabels = cv2.resize(Klabels, (512, pred_length), interpolation=cv2.INTER_AREA)
            Kdifference = cv2.resize(Kdifference, (512, pred_length), interpolation=cv2.INTER_AREA)
            Kconfidence = cv2.resize(Kconfidence, (512, pred_length), interpolation=cv2.INTER_AREA)
            #plt.subplot(1, 3, 1), plt.imshow(Kpredictions)
            #plt.subplot(1, 3, 2), plt.imshow(interp)
            #plt.subplot(1, 3, 3), plt.imshow(Klabels)
            #plt.show()

            Krms = np.power(np.nansum(np.power(Kconfidence - Klabels, 2)) / (np.count_nonzero(~np.isnan(Kconfidence))), .5)
            Kmae = np.abs(Kconfidence - Klabels)
            #plt.subplot(1, 3, 1), plt.imshow(Kconfidence)
            #plt.subplot(1, 3, 2), plt.imshow(Klabels)
            #plt.subplot(1, 3, 3), plt.imshow(Kmae)
            #plt.show()

            Kmaelist.append(Kmae)
            Kdifferencelist.append(Kdifference)
            Kmean_mae = np.nanmean(Kmae)
            Kmean_difference = np.nanmean(Kdifference)

            Krmslist.append(Krms)
            Kmeanmaelist.append(Kmean_mae)
            Kmeandifferencelist.append(Kmean_difference)
        except:
            print("Failure on this file")

    print("KRMS: ", np.mean(Krmslist))
    print("KMean MAE: ", np.mean(Kmeanmaelist))
    print("KMean Difference: ", np.mean(Kmeandifferencelist))
    np.save('./results/CalcRunupCalc' + str(row_skip) + '.npy', CalcRunupCalc)
    np.save('./results/Krmslist' + str(row_skip) + '.npy', Krmslist)
    np.save('./results/Kaelist' + str(row_skip) + '.npy', Kmaelist)
    np.save('./results/Kdifferencelist' + str(row_skip) + '.npy', Kdifferencelist)
    np.save('./results/Kmeanmaelist' + str(row_skip) + '.npy', Kmeanmaelist)
    np.save('./results/Kmeandifferencelist' + str(row_skip) + '.npy', Kmeandifferencelist)
    return CalcRunupCalc, Krmslist, Kmaelist, Kdifferencelist, Kmeanmaelist, Kmeandifferencelist

class Predictor(object):
    def __init__(self):
        self.test_fnamelist = glob.glob('./data/test/*.npy')

    def main(self):
        print("predicting...")
        predictions_ensemble = []
        images_ensemble = []
        labels_ensemble = []
        for e in range(ensemble_runs):
            import settings
            test_dataset = LidarDataset(LidarDataset)
            test_net = LidarNet()
            model = test_net.load_model()

            if e == 0:
                model[0].summary()
            #tf.keras.backend.set_learning_phase(0)
            print("noise std: ", model[0].layers[4].stddev)
            for layer in model[0].layers:
                layer.stddev = noise_std
            print("noise std: ", model[0].layers[4].stddev)

            if args.save_activations:
                layers_to_viz = [7, 51, 52]
                outputs = [model[0].layers[i].output for i in layers_to_viz]
                model = tf.keras.Model(inputs=model[0].inputs, outputs=outputs)

            predictions = []
            images = []
            labels = []
            print("Test Images: " + str(test_size))
            for i in range(int((matlength)/row_skip)):
                print(str(i) + "/" + str(int((matlength)/row_skip)))
                img_batch, label_batch = test_net.get_unet_batch(test_dataset, train_flag='test')
                final_label_batch = np.zeros((test_size, int(.5 * dimensions), int(.5 * dimensions)), dtype=np.float32)
                final_img_batch = np.zeros((test_size, int(.5 * dimensions), int(.5 * dimensions), 4), dtype=np.float32)
                img_batch = np.where(np.isnan(img_batch), 0, img_batch)
                label_batch = np.where(np.isnan(label_batch), 0, label_batch)
                for l in range(test_size):
                    final_img_batch[l, :, :, 0] = cv2.resize(np.abs(img_batch[l, :, :, 0]/31), (6144, 6144), interpolation=cv2.INTER_AREA)
                    final_img_batch[l, :, :, 1] = cv2.resize(img_batch[l, :, :, 1]/7, (6144, 6144), interpolation=cv2.INTER_AREA)
                    final_img_batch[l, :, :, 2] = cv2.resize(img_batch[l, :, :, 2], (6144, 6144), interpolation=cv2.INTER_AREA)
                    final_img_batch[l, :, :, 3] = cv2.resize(img_batch[l, :, :, 3], (6144, 6144), interpolation=cv2.INTER_AREA)
                    final_label_batch[l, :, :] = cv2.resize(label_batch[l, :, :, 0], (6144, 6144), interpolation=cv2.INTER_AREA)

                if args.save_activations:
                    output = model.predict(final_img_batch, batch_size=1, verbose=0)

                    layer_50 = output[0]
                    layer_51 = output[1]
                    predictions.append(output[2])
                    # plot the output from each block

                    square = 5
                    l = 0
                    date = sorted(glob.glob('./data/test/features_labels*.npy'))
                    date = list(reversed(date))

                    predictions = np.asarray(predictions)
                    print(np.shape(predictions))
                    predictions = predictions[0]
                    for fmap in predictions:
                        which = l
                        date = date[which]
                        pred = predictions[which]
                        # plot all 64 maps in an 8x8 squares
                        ix = 1
                        print(l)
                        for _ in range(square):
                            for _ in range(square*2):
                                # specify subplot and turn of axis
                                ax = plt.subplot(square, 2*square, ix)
                                ax.set_xticks([])
                                ax.set_yticks([])
                                # plot filter channel in grayscale
                                if ix == 1:
                                    plt.imshow(img_batch[which, :, :, 0], cmap='gray')
                                    plt.title("aGrid")
                                elif ix == 2:
                                    plt.imshow(img_batch[which, :, :, 1], cmap='gray')
                                    plt.title("zGrid")
                                elif ix == 3:
                                    plt.imshow(img_batch[which, :, :, 2], cmap='gray')
                                    plt.title("zDiffCumulative")
                                elif ix == 4:
                                    plt.imshow(img_batch[which, :, :, 3], cmap='gray')
                                    plt.title("Z-Zmin")
                                elif ix == 5:
                                    plt.imshow(label_batch[which, :, :], cmap='gray')
                                    plt.title("Label " + date)
                                elif ix == 6:
                                    plt.imshow(pred[which, :, :], cmap='gray')
                                    plt.title("Prediction" + date)
                                else:
                                    plt.imshow(layer_50[which, :, :, ix - 1], cmap='gray')
                                ix += 1
                        l += 1
                        # show the figure
                        #plt.subplots_adjust(right=.54)
                        plt.show()
                    continue
                else:
                    predictions.append(model[0].predict(final_img_batch, batch_size=10, verbose=0))


                """print(np.shape(predictions))
                plt.subplot(3, 3, 1), plt.imshow(predictions[i][0][:, :, 0])
                plt.subplot(3, 3, 2), plt.imshow(predictions[i][1][:, :, 0])
                plt.subplot(3, 3, 3), plt.imshow(predictions[i][2][:, :, 0])
                plt.show()"""
                images.append(final_img_batch)
                labels.append(final_label_batch)
                settings.start_row += row_skip
            settings.start_row = 0
            predictions_ensemble.append(predictions)
            images_ensemble.append(images)
            labels_ensemble.append(labels)

        predictions_ensemble = np.asarray(predictions_ensemble)
        images_ensemble = np.asarray(images_ensemble)
        labels_ensemble = np.asarray(labels_ensemble)

        # SIZE [ENSEMBLES, WINDOWS, HOURS, Y, X, CHANNELS]
        """fig = plt.figure()
        ax0 = fig.add_subplot(3, 3, 1), plt.imshow(images_ensemble[0, 0, 0, :, :, 0])
        ax0[0].set_ylabel("Cross-shore distance")
        ax0[0].set_xlabel("Time")
        ax1 = fig.add_subplot(3, 3, 2), plt.imshow(labels_ensemble[0, 0, 0, :, :])
        ax1[0].set_ylabel("Cross-shore distance")
        ax1[0].set_xlabel("Time")
        ax2 = fig.add_subplot(3, 3, 3), plt.imshow(predictions_ensemble[0, 0, 0, :, :, 0])
        ax2[0].set_ylabel("Cross-shore distance")
        ax2[0].set_xlabel("Time")
        ax3 = fig.add_subplot(3, 3, 4), plt.imshow(images_ensemble[1, 0, 0, :, :, 0])
        ax3[0].set_ylabel("Cross-shore distance")
        ax3[0].set_xlabel("Time")
        ax4 = fig.add_subplot(3, 3, 5), plt.imshow(labels_ensemble[1, 0, 0, :, :])
        ax4[0].set_ylabel("Cross-shore distance")
        ax4[0].set_xlabel("Time")
        ax5 = fig.add_subplot(3, 3, 6), plt.imshow(predictions_ensemble[1, 0, 0, :, :, 0])
        ax5[0].set_ylabel("Cross-shore distance")
        ax5[0].set_xlabel("Time")
        ax6 = fig.add_subplot(3, 3, 7), plt.imshow(images_ensemble[2, 0, 0, :, :, 0])
        ax6[0].set_ylabel("Cross-shore distance")
        ax6[0].set_xlabel("Time")
        ax7 = fig.add_subplot(3, 3, 8), plt.imshow(labels_ensemble[2, 0, 0, :, :])
        ax7[0].set_ylabel("Cross-shore distance")
        ax7[0].set_xlabel("Time")
        ax8 = fig.add_subplot(3, 3, 9), plt.imshow(predictions_ensemble[2, 0, 0, :, :, 0])
        ax8[0].set_ylabel("Cross-shore distance")
        ax8[0].set_xlabel("Time")
        plt.show()"""
        return predictions_ensemble, images_ensemble, labels_ensemble

    def merge_ensembles(self, predictions_ensemble, images_ensemble, labels_ensemble):

        print("Merging ensembles...")
        predictions_std = np.std(predictions_ensemble, axis=0)
        predictions = np.mean(predictions_ensemble, axis=0)
        images = np.mean(images_ensemble, axis=0)
        labels = np.mean(labels_ensemble, axis=0)

        """for i in range(len(images[0])):
            fig = plt.figure()
            ax0 = fig.add_subplot(2, 2, 1), plt.imshow(images[0, i, :, :, 0])
            ax0[0].set_ylabel("Cross-shore distance")
            ax0[0].set_xlabel("Time")
            ax1 = fig.add_subplot(2, 2, 2), plt.imshow(labels[0, i, :, :])
            ax1[0].set_ylabel("Cross-shore distance")
            ax1[0].set_xlabel("Time")
            ax2 = fig.add_subplot(2, 2, 3), plt.imshow(predictions[0, i, :, :, 0])
            ax2[0].set_ylabel("Cross-shore distance")
            ax2[0].set_xlabel("Time")
            ax3 = fig.add_subplot(2, 2, 4), plt.imshow(predictions_std[0, i, :, :, 0])
            ax3[0].set_ylabel("Cross-shore distance")
            ax3[0].set_xlabel("Time")
            plt.show()"""

        np.save('./results/predictions_std' + str(row_skip) + '.npy', predictions_std)
        np.save('./results/predictions' + str(row_skip) + '.npy', predictions)
        np.save('./results/images' + str(row_skip) + '.npy', images)
        np.save('./results/labels' + str(row_skip) + '.npy', labels)

    def reconstruct_window(self):
        # reconstruct inferences into the long time series the lidar data is originally taken in
        # support for different sliding window sizes with "row_skip" variable in settings.py

        images = np.load('./results/images' + str(row_skip) + '.npy')
        labels = np.load('./results/labels' + str(row_skip) + '.npy')
        predictions = np.load('./results/predictions' + str(row_skip) + '.npy')
        predictions = predictions[:, :, :, :, 0]
        print(np.shape(images))
        print(np.shape(labels))
        print(np.shape(predictions))
        temp_images = np.ndarray((test_size, len(images)*int(.5*row_skip), 512, 4), dtype=np.float32)
        temp_labels = np.ndarray((test_size, len(labels)*int(.5*row_skip), 512), dtype=np.float32)
        temp_predictions = np.ndarray((test_size, len(predictions)*int(.5*row_skip), 512), dtype=np.float32)
        temp_images[:, :, :, :] = np.NaN
        temp_labels[:, :, :] = np.NaN
        temp_predictions[:, :, :] = np.NaN
        print("Reconstructing individual predictions...")
        for j in range(test_size):
            print("reconstructing image... %d of %d " % (j, test_size))
            row = 0
            for i in range(len(images)):
                try:
                    temp_images[j, row:(row+512), :, :] = images[i, j, :, :, :]
                except ValueError:
                    pass#temp_images[j, row:(row + 512), :, :] = 10*images[i, j, -len(temp_predictions[j, row:]):, :, :]
                row += int(.5*row_skip)
            row = 0
            for i in range(len(images)):
                try:
                    temp_labels[j, row:(row+512), :] = labels[i, j, :, :]
                except ValueError:
                    pass
                    #temp_labels[j, row:(row + 512), :] = labels[i, j, -len(temp_predictions[j, row:]):, :]
                row += int(.5*row_skip)
            row = 0
            pass_no = 0
            for i in range(len(images)):
                """fig = plt.figure()
                ax0 = fig.add_subplot(1, 3, 1), plt.imshow(temp_predictions[j, row:(row + 512), :])
                ax0[0].set_ylabel("Cross-shore distance")
                ax0[0].set_xlabel("Time")
                ax1 = fig.add_subplot(1, 3, 2), plt.imshow(temp_images[j, row:(row + 512), :, 0])
                ax1[0].set_ylabel("Cross-shore distance")
                ax1[0].set_xlabel("Time")
                ax2 = fig.add_subplot(1, 3, 3), plt.imshow(predictions[i, j, :, :])
                ax2[0].set_ylabel("Cross-shore distance")
                ax2[0].set_xlabel("Time")
                plt.show()"""
                try:
                    temp_predictions[j, row:(row + 512), :] = \
                        np.where(np.isnan(temp_predictions[j, row:(row+512), :]), predictions[i, j, :, :],
                             (temp_predictions[j, row:(row+512), :]*pass_no + predictions[i, j, :, :])/(pass_no+1))
                except ValueError:
                    pass
                    """temp_predictions[j, row:(row + 512), :] = np.where(np.isnan(temp_predictions[j, row:(row+512), :]),
                             predictions[i, j, -len(temp_predictions[j, row:]):, :],
                             (temp_predictions[j, row:(row+512), :]*pass_no + predictions[i, j, -len(temp_predictions[j, row:]):, :])/(pass_no+1))"""
                """fig = plt.figure()
                ax0 = fig.add_subplot(1, 3, 1), plt.imshow(temp_predictions[j, row:(row + 512), :])
                ax0[0].set_ylabel("Cross-shore distance")
                ax0[0].set_xlabel("Time")
                ax1 = fig.add_subplot(1, 3, 2), plt.imshow(temp_images[j, row:(row + 512), :, 0])
                ax1[0].set_ylabel("Cross-shore distance")
                ax1[0].set_xlabel("Time")
                ax2 = fig.add_subplot(1, 3, 3), plt.imshow(predictions[i, j, :, :])
                ax2[0].set_ylabel("Cross-shore distance")
                ax2[0].set_xlabel("Time")
                plt.show()"""
                row += int(.5*row_skip)
        images = temp_images
        labels = temp_labels
        predictions = temp_predictions
        print("Images shape: ", np.shape(images))
        print("Labels shape: ", np.shape(labels))
        print("Predictions shape: ", np.shape(predictions))
        return images, labels, predictions

    def define_runupline(self, labels, predictions):
        # defines runupline
        # saves to mat file
        print("Defining runup line...")
        downLineIndex_list = []
        calc_downlineindex_list = []
        human_downlineindex_list = []
        test_fnamelist = glob.glob('./MATLAB files/*')
        test_fnamelist.sort(key=natural_keys)
        runup_calc_fnamelist = glob.glob('./MATLAB files/calculateRunupLine/A*')
        runup_calc_fnamelist.sort(key=natural_keys)

        for i in range(test_size):
            print(test_fnamelist[i])
            #try:
            loaded_mat_QAQCd = mat73.loadmat(test_fnamelist[i])
            runup_calc = loadmat(runup_calc_fnamelist[i])
            human_runup_calc = runup_calc['lineRunupCalc']
            calc_runup_calc = runup_calc['lineAnalyticalRunupCalc']
            human_binary = human_runup_calc['griddedDataIsWater'][0][0][:-490, :settings.dimensions]
            calc_binary = calc_runup_calc['griddedDataIsWater'][0][0][:-490, :settings.dimensions]
            calc_downlineindex = calc_runup_calc['downLineIndex'][0][0]
            human_downlineindex = human_runup_calc['downLineIndex'][0][0]
            downlineindex = np.zeros((len(human_binary)))

            prediction_binary = np.where(predictions[i] >=.5, 1.0, 0.0)
            human_binary = np.where(human_binary>.5, 1.0, 0.0)
            calc_binary = np.where(calc_binary>.5, 1.0, 0.0)
            prediction = cv2.resize(predictions[i], (1024, len(human_binary)), interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(labels[i], (1024, len(human_binary)), interpolation=cv2.INTER_CUBIC)
            defined_prediction = cv2.resize(prediction_binary, (1024, len(human_binary)), interpolation=cv2.INTER_CUBIC)
            Y = np.linspace(0, len(human_binary), len(human_binary))
            X = np.linspace(0, 6144, 6144)

            fig = plt.figure()
            ax0 = fig.add_subplot(1, 4, 1), plt.imshow(prediction)
            cs = ax0[0].contour(X, Y, label,
                             zorder=10,
                             vmin=0, vmax=1, alpha=1,
                             colors=['red'],
                             levels=[.5],
                             linestyles=['dashed'],
                             linewidths=[2, 2])
            cs = ax0[0].contour(X, Y, prediction,
                             zorder=10,
                             vmin=0, vmax=1, alpha=1,
                             colors=['magenta'],
                             levels=[.5],
                             linestyles=['dashed'],
                             linewidths=[2, 2])
            p = cs.collections[0].get_paths()
            list_len = [len(i) for i in p]
            longest_contour_index = np.argmax(np.array(list_len))
            v = p[longest_contour_index].vertices
            x = v[:, 0]
            y = v[:, 1]
            ax1 = fig.add_subplot(1, 4, 2), plt.imshow(label)
            cs = ax1[0].contour(X, Y, defined_prediction,
                             zorder=10,
                             vmin=0, vmax=1, alpha=1,
                             colors=['magenta'],
                             levels=[.5],
                             linestyles=['dashed'],
                             linewidths=[2, 2])
            cs = ax1[0].contour(X, Y, label,
                             zorder=10,
                             vmin=0, vmax=1, alpha=1,
                             colors=['red'],
                             levels=[.5],
                             linestyles=['dashed'],
                             linewidths=[2, 2])
            ax2 = fig.add_subplot(1, 4, 3), plt.imshow(human_binary)
            cs = ax2[0].contour(X, Y, defined_prediction,
                             zorder=10,
                             vmin=0, vmax=1, alpha=1,
                             colors=['magenta'],
                             levels=[.5],
                             linestyles=['dashed'],
                             linewidths=[2, 2])
            cs = ax2[0].contour(X, Y, label,
                             zorder=10,
                             vmin=0, vmax=1, alpha=1,
                             colors=['red'],
                             levels=[.5],
                             linestyles=['dashed'],
                             linewidths=[2, 2])
            ax3 = fig.add_subplot(1, 4, 4), plt.imshow(calc_binary)
            cs = ax3[0].contour(X, Y, defined_prediction,
                             zorder=10,
                             vmin=0, vmax=1, alpha=1,
                             colors=['magenta'],
                             levels=[.5],
                             linestyles=['dashed'],
                             linewidths=[2, 2])
            cs = ax3[0].contour(X, Y, label,
                             zorder=10,
                             vmin=0, vmax=1, alpha=1,
                             colors=['red'],
                             levels=[.5],
                             linestyles=['dashed'],
                             linewidths=[2, 2])
            #plt.show()
            plt.close('all')
            for j in range(len(human_binary)):
                #find all indexes in x with value i
                #find all indexes in x with value i
            #   #average y values of those xes
            #   #return the average
                indexes = np.where((j+1>y) & (y>=j))
                #print("old indexes", indexes)
                #print("new index", j)
                if len(indexes[0]) > 0:
                    avg_x = np.mean(x[indexes])
                    downlineindex[j] = avg_x
                else:
                    downlineindex[j] = downlineindex[(j-1)]

            loaded_mat_QAQCd['lineRunupCalc']['downLineIndex'] = downlineindex
            #except:
                #print("Error in definining runup line on this ^ file")

            """fig = plt.figure()
            ax0 = fig.add_subplot(1, 2, 1), plt.imshow(label)
            plt.plot(x, y)
            ax1 = fig.add_subplot(1, 2, 2), plt.imshow(prediction)
            plt.plot(downLineIndex_list[0], np.linspace(0, matlength, matlength))
            #plt.show()
            plt.close('all')"""

            for l in range(len(loaded_mat_QAQCd['lineGriddedData']['zGridInterp'])):
                try:
                    temp_indx = int(downlineindex[l])
                    loaded_mat_QAQCd['lineRunupCalc']['Z'][l] = loaded_mat_QAQCd['lineGriddedData']['zGridInterp'][l, temp_indx]
                except:
                    pass

            print(test_fnamelist[i][15:])
            print(downlineindex)
            print(human_downlineindex.T[0][:-490])
            print(calc_downlineindex[0][:-490])

            print(np.shape(downlineindex))
            print(np.shape(human_downlineindex.T[0][:-490]))
            print(np.shape(calc_downlineindex[0][:-490]))

            downLineIndex_list.append(downlineindex)
            human_downlineindex_list.append(human_downlineindex.T[0][:-490])
            calc_downlineindex_list.append(calc_downlineindex[0][:-490])

            final_matfile = {}
            final_matfile['lineRunupCalc'] = loaded_mat_QAQCd['lineRunupCalc']
            final_matfile['lineCoredat'] = loaded_mat_QAQCd['lineCoredat']
            savemat('./MATLAB files/calculateRunupLine/ML_' + str(row_skip) + '_' + test_fnamelist[i][15:], final_matfile)

        np.save('./results/human_downlineindex_list' + str(row_skip) + '.npy', human_downlineindex_list)
        np.save('./results/calc_downlineindex_list' + str(row_skip) + '.npy', calc_downlineindex_list)
        np.save('./results/downLineIndex_list' + str(row_skip) + '.npy', downLineIndex_list)
        return downLineIndex_list

    def runupline_5cm(self, images):
        downlineindex_list = np.load('./results/downLineIndex_list512.npy', allow_pickle=True)
        calc_downlineindex_list = np.load('./results/calc_downLineIndex_list512.npy', allow_pickle=True)
        runup_calc_fnamelist = glob.glob('./MATLAB files/calculateRunupLine/A*')
        runup_calc_fnamelist.sort(key=natural_keys)

        test_fnamelist = glob.glob('./MATLAB files/calculateRunupLine/Ana*')
        test_fnamelist.sort(key=natural_keys)

        ml_cm_runup_index_list = []
        calc_cm_runup_index_list = []
        human_cm_runup_index_list = []

        for i in range(32, 33):
            runup_calc = loadmat(runup_calc_fnamelist[i])
            human_runup_calc = runup_calc['lineRunupCalc']
            human_downlineindex = human_runup_calc['downLineIndex'][0][0][:-490]
            if len(human_downlineindex) < 6144:
                human_downlineindex = human_runup_calc['downLineIndex'][0][0][0][:-490]

            ml_index = (downlineindex_list[i]/2).astype(int)
            calc_index = (calc_downlineindex_list[i]/2).astype(int)
            human_index = (human_downlineindex/2).astype(int)

            ml_index = cv2.resize(ml_index, (1, 6144), interpolation=cv2.INTER_NEAREST)
            calc_index = cv2.resize(calc_index, (1, 6144), interpolation=cv2.INTER_NEAREST)
            human_index = cv2.resize(human_index, (1, 6144), interpolation=cv2.INTER_NEAREST)
            orig_ml_index = np.copy(ml_index)
            orig_human_index = np.copy(human_index)

            human_min = np.nanmin(human_index)
            human_max = np.nanmax(human_index)

            z_elev = np.where(images[i, :, :, 1]*7 == 0, np.nan, images[i, :, :, 1]*7)
            orig_z_elev = np.copy(z_elev)
            kernal = Gaussian2DKernel(x_stddev=1, y_stddev=1)
            #kernal = Gaussian1DKernel(stddev=.1)
            #for ii in range(6144):
            #    z_elev[ii, :] = convolve(z_elev[ii, :], kernal)
            #for iii in range(512):
            #    z_elev[:, iii] = convolve(z_elev[:, iii], kernal)
            #z_elev = convolve(z_elev, kernal)
            temp_images = np.copy(images[i])

            interpolated_z = np.copy(z_elev)

            for k in range(512):
                interpolated_z[:, k] = np.asarray(pd.DataFrame(z_elev[:, k]).interpolate().values.ravel().tolist())
                #interpolated_z[:, k] = smooth(interpolated_z[:, k], 21)
                #interpolated_z[:, k] = smooth(interpolated_z[:, k], 15)
                interpolated_z[:, k] = smooth(interpolated_z[:, k], 7)

            waternan_MLimage = np.copy(interpolated_z)
            waternan_Calcimage = np.copy(interpolated_z)
            waternan_Himage = np.copy(interpolated_z)

            print(np.shape(waternan_MLimage))
            for j in range(6144):
                waternan_MLimage[j, int(ml_index[j, 0]):-1] = np.nan
                waternan_Calcimage[j, int(calc_index[j, 0]):-1] = np.nan
                waternan_Himage[j, int(human_index[j, 0]):-1] = np.nan

            interpolated_MLdry = np.copy(waternan_MLimage)
            interpolated_Calcdry = np.copy(waternan_Calcimage)
            interpolated_Hdry = np.copy(waternan_Himage)

            for k in range(512):
                interpolated_MLdry[:, k] = np.asarray(pd.DataFrame(interpolated_MLdry[:, k]).interpolate().values.ravel().tolist())
                interpolated_Calcdry[:, k] = np.asarray(pd.DataFrame(interpolated_Calcdry[:, k]).interpolate().values.ravel().tolist())
                interpolated_Hdry[:, k] = np.asarray(pd.DataFrame(interpolated_Hdry[:, k]).interpolate().values.ravel().tolist())
                interpolated_MLdry[:, k] = smooth(interpolated_MLdry[:, k], 21)
                interpolated_Calcdry[:, k] = smooth(interpolated_Calcdry[:, k], 21)
                interpolated_Hdry[:, k] = smooth(interpolated_Hdry[:, k], 21)
                interpolated_MLdry[:, k] = smooth(interpolated_MLdry[:, k], 15)
                interpolated_Calcdry[:, k] = smooth(interpolated_Calcdry[:, k], 15)
                interpolated_Hdry[:, k] = smooth(interpolated_Hdry[:, k], 15)
                interpolated_MLdry[:, k] = smooth(interpolated_MLdry[:, k], 7)
                interpolated_Calcdry[:, k] = smooth(interpolated_Calcdry[:, k], 7)
                interpolated_Hdry[:, k] = smooth(interpolated_Hdry[:, k], 7)

            for k in range(512):
                interpolated_MLdry[:, k] = np.asarray(pd.DataFrame(np.flip(interpolated_MLdry[:, k])).interpolate().values.ravel().tolist())
                interpolated_Calcdry[:, k] = np.asarray(pd.DataFrame(np.flip(interpolated_Calcdry[:, k])).interpolate().values.ravel().tolist())
                interpolated_Hdry[:, k] = np.asarray(pd.DataFrame(np.flip(interpolated_Hdry[:, k])).interpolate().values.ravel().tolist())
                interpolated_MLdry[:, k] = smooth(interpolated_MLdry[:, k], 21)
                interpolated_Calcdry[:, k] = smooth(interpolated_Calcdry[:, k], 21)
                interpolated_Hdry[:, k] = smooth(interpolated_Hdry[:, k], 21)
                interpolated_MLdry[:, k] = smooth(interpolated_MLdry[:, k], 15)
                interpolated_Calcdry[:, k] = smooth(interpolated_Calcdry[:, k], 15)
                interpolated_Hdry[:, k] = smooth(interpolated_Hdry[:, k], 15)

            interpolated_MLdry = np.flip(interpolated_MLdry, axis=0)
            interpolated_Calcdry = np.flip(interpolated_Calcdry, axis=0)
            interpolated_Hdry = np.flip(interpolated_Hdry, axis=0)


            for k in range(6144):
                interpolated_MLdry[k, :] = smooth(interpolated_MLdry[k, :], 21)
                interpolated_Calcdry[k, :] = smooth(interpolated_Calcdry[k, :], 21)
                interpolated_Hdry[k, :] = smooth(interpolated_Hdry[k, :], 21)
                interpolated_MLdry[k, :] = smooth(interpolated_MLdry[k, :], 15)
                interpolated_Calcdry[k, :] = smooth(interpolated_Calcdry[k, :], 15)
                interpolated_Hdry[k, :] = smooth(interpolated_Hdry[k, :], 15)
                interpolated_MLdry[k, :] = smooth(interpolated_MLdry[k, :], 7)
                interpolated_Calcdry[k, :] = smooth(interpolated_Calcdry[k, :], 7)
                interpolated_Hdry[k, :] = smooth(interpolated_Hdry[k, :], 7)

            """zdiff_ml = orig_z_elev - interpolated_MLdry
            zdiff_k = orig_z_elev - interpolated_Calcdry
            zdiff_t = orig_z_elev - interpolated_Hdry"""

            zdiff_ml = interpolated_z - interpolated_MLdry
            zdiff_k = interpolated_z - interpolated_Calcdry
            zdiff_t = interpolated_z - interpolated_Hdry

            above_cm_ml = np.copy(zdiff_ml)
            above_cm_calc = np.copy(zdiff_k)
            above_cm_human = np.copy(zdiff_t)

            ml_cm_runup_index = np.zeros((6144, 1))
            calc_cm_runup_index = np.zeros((6144, 1))
            human_cm_runup_index = np.zeros((6144, 1))

            for j in range(6144):
                ml_cm_index = np.where(zdiff_ml[j, ml_index[j, 0]:] > .03)
                calc_cm_index = np.where(zdiff_k[j, calc_index[j, 0]:] > .03)
                human_cm_index = np.where(zdiff_t[j, human_index[j, 0]:] > .03)
                ml_cm_index = ml_cm_index[0]
                calc_cm_index = calc_cm_index[0]
                human_cm_index = human_cm_index[0]

                if len(ml_cm_index)>1:
                    ml_cm_index = ml_cm_index[0]
                if len(calc_cm_index)>1:
                    calc_cm_index = calc_cm_index[0]
                if len(human_cm_index)>1:
                    human_cm_index = human_cm_index[0]

                ml_cm_index += ml_index[j, 0]
                calc_cm_index += calc_index[j, 0]
                human_cm_index += human_index[j, 0]

                """if ml_cm_index > 0:
                    fig = plt.figure()
                    ax3 = fig.add_subplot(2, 3, 4), plt.imshow(above_cm_ml, cmap='inferno')
                    ax4 = fig.add_subplot(2, 3, 5), plt.imshow(above_cm_calc, cmap='inferno')
                    ax5 = fig.add_subplot(2, 3, 6), plt.imshow(above_cm_human, cmap='inferno')
                    try:
                        ax3[0].axvline(ml_cm_index)
                    except:
                        ax3[0].axvline(np.nan)
                    try:
                        ax4[0].axvline(calc_cm_index)
                    except:
                        ax4[0].axvline(np.nan)
                    try:
                        ax5[0].axvline(human_cm_index)
                    except:
                        ax5[0].axvline(np.nan)
                    plt.show()"""

                if ml_cm_index >=511:
                    ml_cm_index = np.nan
                if calc_cm_index >=511:
                    calc_cm_index = np.nan
                if human_cm_index >=511:
                    human_cm_index = np.nan

                try:
                    above_cm_ml[j, int(ml_cm_index):-1] = 1
                except:
                    above_cm_ml[j] = np.nan
                try:
                    above_cm_calc[j, int(calc_cm_index):-1] = 1
                except:
                    above_cm_calc[j] = np.nan
                try:
                    above_cm_human[j, int(human_cm_index):-1] = 1
                except:
                    above_cm_human[j] = np.nan
                try:
                    ml_cm_runup_index[j] = int(ml_cm_index)
                except:
                    pass
                try:
                    calc_cm_runup_index[j] = int(calc_cm_index)
                except:
                    pass
                try:
                    human_cm_runup_index[j] = int(human_cm_index)
                except:
                    pass

            ml_cm_runup_index = np.where(ml_cm_runup_index == 0, np.nan, ml_cm_runup_index)
            calc_cm_runup_index = np.where(calc_cm_runup_index == 0, np.nan, calc_cm_runup_index)
            human_cm_runup_index = np.where(human_cm_runup_index == 0, np.nan, human_cm_runup_index)

            ml_cm_runup_index = np.asarray(pd.DataFrame(ml_cm_runup_index.T[0]).interpolate().values.ravel().tolist())
            calc_cm_runup_index = np.asarray(pd.DataFrame(calc_cm_runup_index.T[0]).interpolate().values.ravel().tolist())
            human_cm_runup_index = np.asarray(pd.DataFrame(human_cm_runup_index.T[0]).interpolate().values.ravel().tolist())

            ml_index = self.get_smooth_runup_indices(ml_cm_runup_index)
            calc_index = self.get_smooth_runup_indices(calc_cm_runup_index)
            human_index = self.get_smooth_runup_indices(human_cm_runup_index)

            ml_cm_runup_index_list.append(ml_index)
            calc_cm_runup_index_list.append(calc_index)
            human_cm_runup_index_list.append(human_index)

            plot_name = test_fnamelist[i]
            print(plot_name)
            predict.plot_presentation_input_simple(interpolated_z, temp_images, 0, plot_name)
            imageio.mimsave('./Paper/full_' + str(i) + '.avi',
                            [predict.plot_presentation_timestep_simple(interpolated_z, temp_images, timestep, ml_index, human_index,
                                                    interpolated_Hdry, orig_human_index, orig_ml_index, plot_name)
                             for timestep in range(800)], fps=8)

        np.save("./ml_cm_runup_index_list.npy", ml_cm_runup_index_list)
        np.save("./calc_cm_runup_index_list.npy", calc_cm_runup_index_list)
        np.save("./human_cm_runup_index_list.npy", human_cm_runup_index_list)

    def plot_presentation_input_simple(self, image, images, timestep, plot_name):
        if timestep % 400 == 0:
            print(timestep)
            settings.start_row = timestep
            settings.timestep_plot_indice = 0

        z_min = np.nanpercentile(np.where(image[:, :] == 0, np.nan, image[:, :]), 5, axis=0)

        ymax = z_min[50] + 1

        fig = plt.figure(figsize=(16, 9))
        fig.suptitle(plot_name[-43:-30])
        grid = gridspec.GridSpec(2, 3, figure=fig)
        ax0 = fig.add_subplot(grid[0, 0])
        ax1 = fig.add_subplot(grid[1, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        ax3 = fig.add_subplot(grid[1, 1])
        ax6 = fig.add_subplot(grid[:, 2])

        norm = mpl.cm.colors.Normalize(vmax=ymax, vmin=0)
        ax0.set_title("Elevation Timestack", fontsize=20)
        ax0.imshow(images[settings.start_row:settings.start_row + 400, 50:400, 1]*7, cmap='jet', norm=norm)
        ax0.axhline(y=settings.timestep_plot_indice, color='black')
        ax0.set_ylabel("Time", fontsize=16)
        ax0.set_xticklabels((ax0.get_xticks() / 3.5).astype(int))
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax0)
        cbar.set_label('Elevation (m)', fontsize=16)

        norm = mpl.cm.colors.Normalize(vmax=31.0, vmin=0)
        ax1.set_title("Reflectance Timestack", fontsize=20)
        ax1.imshow(images[settings.start_row:settings.start_row + 400, 50:400, 0]*31, cmap='jet_r', norm=norm)
        ax1.axhline(y=settings.timestep_plot_indice, color='black')
        ax1.set_ylabel("Time", fontsize=16)
        ax1.set_xlabel("Cross-shore distance (m)", fontsize=16)
        ax1.set_xticklabels((ax1.get_xticks() / 3.5).astype(int))
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax1)
        cbar.set_label('Intensity', fontsize=16)

        norm = mpl.cm.colors.Normalize(vmax=np.nanmax(images[settings.start_row:settings.start_row + 400, 50:400, 2]), vmin=0)
        ax2.set_title("Elevation - Min Elevation", fontsize=20)
        ax2.imshow(images[settings.start_row:settings.start_row + 400, 50:400, 2], cmap='jet', norm=norm)
        ax2.axhline(y=settings.timestep_plot_indice, color='black')
        ax2.set_xticklabels((ax2.get_xticks() / 3.5).astype(int))
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax2)
        cbar.set_label('Elevation (m)', fontsize=16)

        norm = mpl.cm.colors.Normalize(vmax=1, vmin=0)
        ax3.set_title("Cumulative Elevation Diff", fontsize=20)
        ax3.imshow(images[settings.start_row:settings.start_row + 400, 50:400, 3], cmap='jet', norm=norm)
        ax3.axhline(y=settings.timestep_plot_indice, color='black')
        ax3.set_xlabel("Cross-shore distance (m)", fontsize=16)
        ax3.set_xticklabels((ax3.get_xticks() / 3.5).astype(int))
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax3)
        cbar.set_label('Count (norm)', fontsize=16)

        fig.canvas.draw()  # draw the canvas, cache the renderer
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.subplots_adjust(hspace=.25, left=0.1, bottom=0.1, right=.9, top=.9, wspace=.3)
        #plt.show()
        plt.savefig('./Paper/inputs.png')
        plt.close('all')
        settings.timestep_plot_indice += 1
        return plot_image
        #except:
        #    plot_image = np.zeros((1600, 900))
        #    return plot_image

    def plot_presentation_timestep_simple(self, image, images, timestep, ml_index, human_index, interpolated_Hdry, orig_human_index, orig_ml_index, plot_name):
        if timestep % 400 == 0:
            print(timestep)
            settings.start_row = timestep
            settings.timestep_plot_indice = 0

        z_min = np.nanpercentile(np.where(image[:, :] == 0, np.nan, image[:, :]), 5, axis=0)

        ymax = z_min[50] + 1

        fig = plt.figure(figsize=(16, 9))
        fig.suptitle(plot_name[-43:-30], fontsize=24)
        grid = gridspec.GridSpec(2, 3, figure=fig)
        ax0 = fig.add_subplot(grid[0, 0])
        ax1 = fig.add_subplot(grid[1, 0])
        ax6 = fig.add_subplot(grid[:, 1:])

        norm = mpl.cm.colors.Normalize(vmax=ymax, vmin=0)
        ax0.set_title("Elevation Timestack", fontsize=20)
        ax0.imshow(images[settings.start_row:settings.start_row + 400, 50:400, 1]*7, cmap='jet', norm=norm)
        ax0.plot(ml_index[settings.start_row:settings.start_row + 400] - 50, np.linspace(0, 400, 400),
                 linewidth=2,
                 color='white', label='ML 3cm')
        ax0.plot(orig_ml_index[settings.start_row:settings.start_row + 400] - 50, np.linspace(0, 400, 400),
                 linewidth=2,
                 color='white', label='ML w/d', ls='--')
        ax0.plot(human_index[settings.start_row:settings.start_row + 400] - 50, np.linspace(0, 400, 400), linewidth=2,
                 color='black', label='H 3cm')
        ax0.plot(orig_human_index[settings.start_row:settings.start_row + 400] - 50, np.linspace(0, 400, 400),
                 linewidth=2,
                 color='black', label='Human w/d', ls='--')
        ax0.axhline(y=settings.timestep_plot_indice, color='black')
        ax0.set_ylabel("Time", fontsize=16)
        ax0.set_xticklabels((15 + ax0.get_xticks() / 3.5).astype(int))
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax0)
        cbar.set_label('Elevation (m)', fontsize=16)

        norm = mpl.cm.colors.Normalize(vmax=31.0, vmin=0)
        ax1.set_title("Reflectance Timestack", fontsize=20)
        ax1.imshow(images[settings.start_row:settings.start_row + 400, 50:400, 0]*31, cmap='jet_r', norm=norm)
        ax1.plot(human_index[settings.start_row:settings.start_row + 400]-50, np.linspace(0, 400, 400), linewidth=2,
                 color='black', label='H 3cm')
        ax1.plot(ml_index[settings.start_row:settings.start_row + 400] - 50, np.linspace(0, 400, 400),
                 linewidth=2,
                 color='white', label='ML 3cm')
        ax1.plot(orig_ml_index[settings.start_row:settings.start_row + 400] - 50, np.linspace(0, 400, 400),
                 linewidth=2,
                 color='white', label='ML w/d', ls='--')
        ax1.plot(orig_human_index[settings.start_row:settings.start_row + 400] - 50, np.linspace(0, 400, 400),
                 linewidth=2,
                 color='black', label='Human b/w', ls='--')
        ax1.axhline(y=settings.timestep_plot_indice, color='black')
        ax1.set_ylabel("Time", fontsize=16)
        ax1.set_xlabel("Cross-shore distance (m)", fontsize=16)
        ax1.set_xticklabels((15 + ax1.get_xticks() / 3.5).astype(int))
        #ax1.legend(framealpha=1, facecolor='grey')

        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax1)
        cbar.set_label('Intensity', fontsize=16)

        ax6.axvline(x=human_index[timestep], linewidth=4, color='black', label='Human 3cm')
        ax6.axvline(x=ml_index[timestep], linewidth=4, color='green', label='ML 3cm')
        ax6.axvline(x=orig_human_index[timestep], linewidth=2, color='black', label='Human b/w', ls='--')
        ax6.axvline(x=orig_ml_index[timestep], linewidth=2, color='green', label='ML b/w', ls='--')
        ax6.plot(z_min, color='grey', label='Zmin', ls='--', linewidth=8, alpha=0.35)
        ax6.plot(image[timestep, :]-z_min, color='orange', label='Z-Zmin', linewidth=1)
        ax6.plot(image[timestep, :], color='cyan', label='Z', linewidth=2)
        ax6.plot(interpolated_Hdry[timestep, :], color='purple', label='InterpDry', linewidth=5, alpha=.75)
        ax6.plot(image[timestep, :] - interpolated_Hdry[timestep, :], color='darkblue', label='Z-HInterp',
                 linewidth=2)
        ax6.set_title("Timestep Plots", fontsize=20)
        ax6.set_xlabel("Cross-shore distance (m)", fontsize=16)
        ax6.set_ylabel("Elevation (m)", fontsize=16)
        ax6.set_ylim(ymin=-1, ymax=ymax)
        ax6.set_xlim(xmax=500, xmin=30)
        ax6.set_xticklabels(ax6.get_xticks()/5)
        ax6.legend()

        fig.canvas.draw()  # draw the canvas, cache the renderer
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.subplots_adjust(hspace=.25, left=0.1, bottom=0.1, right=.9, top=.9, wspace=.3)
        #plt.show()
        plt.close('all')
        settings.timestep_plot_indice += 1
        return plot_image

    def calc_peaks_and_r2(self, runupTS):
        # calculate up and downcrossings
        temp_runupTS = runupTS/np.nanmean(runupTS)-1
        downcrossings, upcrossing = [], []

        for rr, R in enumerate(temp_runupTS):
            if rr > 1 and temp_runupTS[rr] >= 0 and temp_runupTS[rr - 1] < 0:
                upcrossing.append(rr),
            elif rr > 1 and temp_runupTS[rr] <= 0 and temp_runupTS[rr - 1] > 0:
                downcrossings.append(rr)

        downcrossings = np.array(downcrossings)
        upcrossing = np.array(upcrossing)
        downcrossings = downcrossings[downcrossings > upcrossing[0]]  # start on an upcrossing
        upcrossing = upcrossing[upcrossing < downcrossings[-1]]  # finish on a downcrossing
        peaks = []

        for ii, uc in enumerate(upcrossing):
            rangeR2 = temp_runupTS[upcrossing[ii]:downcrossings[ii]]
            idx = uc + np.argwhere(max(rangeR2) == rangeR2).squeeze()
            if np.size(idx) > 1:
                idx = idx[0]
            peaks.append(idx)  # arg max provides weird answers some times

        assert len(peaks) == len(upcrossing) == len(downcrossings), 'Calculating R2, the peak count does''t match ' \
                                                                    ' up/down crossings'
        R2 = np.percentile((np.array(runupTS))[np.array(peaks).astype(int)], 98)

        """plt.figure()
        plt.plot(runupTS, label='runupTS')
        plt.plot(upcrossing,np.zeros_like(upcrossing), 'r.', label="identifed upcrossing points")
        plt.plot(peaks, runupTS[peaks], 'x', label='identified runupTS peaks')
        plt.legend()
        plt.show()"""

        return peaks, R2

    def get_smooth_runup_indices(self, cm_runup_index):
        index = cv2.resize(cm_runup_index, (1, pred_length), interpolation=cv2.INTER_NEAREST)
        index = smooth(index[:, 0], 3)
        index = np.where(np.isnan(index), np.nanmean(index), index)
        index = smooth(index, 3)
        index = smooth(index, 3)
        return index

    def calc_wavestats(self, images):
        ml_3m_runup_index_list = np.load("./ml_cm_runup_index_list.npy")
        k_3m_runup_index_list = np.load("./calc_cm_runup_index_list.npy")
        t_3m_runup_index_list = np.load("./human_cm_runup_index_list.npy")

        test_fnamelist = glob.glob('./MATLAB files/calculateRunupLine/Ana*')
        test_fnamelist.sort(key=natural_keys)

        t_hm0 = []
        ml_hm0 = []
        t_tm = []
        ml_tm = []
        t_tp = []
        ml_tp = []
        calc_tave = []
        calc_tm = []
        calc_hm0 = []
        t_zm = []
        ml_zm = []
        calc_zm = []
        t_hsig = []
        ml_hsig = []
        t_hsin = []
        ml_hsin = []
        t_mean = []
        ml_mean = []
        h_peaks = []
        ml_peaks = []
        h_r2s = []
        ml_r2s = []
        for i in list(range(89)) + list(range(98, test_size)):
            #print(t_3m_runup_index_list[i])
            image = images[i, :, :, :]
            z = image[:, :, 1]

            z_ml = z[np.arange(6144), np.where((np.asarray(ml_3m_runup_index_list[i])).astype(int) >= 0, (np.asarray(ml_3m_runup_index_list[i])).astype(int), 0)] * 7
            z_calc = z[np.arange(6144), np.where((np.asarray(k_3m_runup_index_list[i])).astype(int) >= 0, (np.asarray(k_3m_runup_index_list[i])).astype(int), 0)] * 7
            z_human = z[np.arange(6144), np.where(np.asarray(t_3m_runup_index_list[i]).astype(int) >= 0, (np.asarray(t_3m_runup_index_list[i])).astype(int), 0)] * 7
            h_peak, h_r2 = self.calc_peaks_and_r2(z_human)
            ml_peak, ml_r2 = self.calc_peaks_and_r2(z_ml)
            h_fspec, h_frqbins = self.timeSeriesAnalysis1D(z_human)
            ml_fspec, ml_frqbins = self.timeSeriesAnalysis1D(z_ml)
            calc_fspec, calc_frqbins = self.timeSeriesAnalysis1D(z_calc)
            h_stats = self.stats1D(h_fspec, h_frqbins)
            ml_stats = self.stats1D(ml_fspec, ml_frqbins)
            calc_stats = self.stats1D(calc_fspec, calc_frqbins)

            t_zm.append(np.nanmean(t_3m_runup_index_list[i]))
            ml_zm.append(np.nanmean(ml_3m_runup_index_list[i]))
            calc_zm.append(np.nanmean(k_3m_runup_index_list[i]))
            t_hm0.append(h_stats['Hm0'])
            ml_hm0.append(ml_stats['Hm0'])
            t_hsig.append(h_stats['HsIG'])
            ml_hsig.append(ml_stats['HsIG'])
            t_hsin.append(h_stats['HsIN'])
            ml_hsin.append(ml_stats['HsIN'])
            t_tm.append(h_stats['Tm'])
            ml_tm.append(ml_stats['Tm'])
            t_tp.append(h_stats['Tp'])
            ml_tp.append(ml_stats['Tp'])
            calc_tave.append(calc_stats['Tave'])
            calc_tm.append(calc_stats['Tm'])
            calc_hm0.append(calc_stats['Hm0'])
            t_mean.append(np.nanmean(z_human))
            ml_mean.append(np.nanmean(z_ml))
            h_peaks.append(h_peak)
            h_r2s.append(h_r2)
            ml_peaks.append(ml_peak)
            ml_r2s.append(ml_r2)
            """print(h_stats['Hm0'])
            print(ml_stats['Hm0'])
            print("____________")"""

            """fig = plt.figure(figsize=(16,9))
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.plot(h_frqbins, h_fspec)
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.plot(ml_frqbins, ml_fspec)
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.plot(h_frqbins, ml_fspec - h_fspec)
            plt.show()"""

        fig = plt.figure(figsize=(16, 9))
        ax0 = fig.add_subplot(2, 3, 1)
        ax0.scatter(h_r2s, ml_r2s)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(h_r2s, ml_r2s)
        ax0.text(2, .75, r'$R^{2}$' + str(r_value**2)[:4])
        ax0.set_title(r'$R_{2\%}$')
        ax0.set_ylabel("Predicted " + r'$R_{2\%}$' + " (m)")
        ax0.set_xlabel('Human ' + r'$R_{2\%}$' + ' (m)')

        ax1 = fig.add_subplot(2, 3, 2)
        ax1.scatter(t_hm0, ml_hm0)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(t_hm0, ml_hm0)
        ax1.text(1.5, 1, r'$R^{2}$' + str(r_value ** 2)[:4])
        ax1.set_title(r'$S_{total}$')
        ax1.set_ylabel("Predicted " + r'$S_{total}$' + " (m)")
        ax1.set_xlabel('Human ' + r'$S_{total}$' + ' (m)')

        ax2 = fig.add_subplot(2, 3, 3)
        ax2.scatter(t_tm, ml_tm)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(t_tm, ml_tm)
        ax2.text(10, 8, r'$R^{2}$' + str(r_value ** 2)[:4])
        ax2.set_title(r'$T_{m}$')
        ax2.set_ylabel("Predicted " + r'$T_{m}$' + " (s)")
        ax2.set_xlabel('Human ' + r'$T_{m}$' + ' (s)')

        ax12 = fig.add_subplot(2, 3, 4)
        ax12.scatter(t_hsig, ml_hsig)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(t_hsig, ml_hsig)
        ax12.text(.8, .5, r'$R^{2}$' + str(r_value ** 2)[:4])
        ax12.set_title(r'$S_{IG}$')
        ax12.set_ylabel("Predicted " + r'$S_{IG}$' + " (m)")
        ax12.set_xlabel('Human ' + r'$S_{IG}$' + ' (m)')

        ax13 = fig.add_subplot(2, 3, 5)
        ax13.scatter(t_hsin, ml_hsin)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(t_hsin, ml_hsin)
        ax13.text(1.35, .85, r'$R^{2}$' + str(r_value ** 2)[:4])
        ax13.set_title(r'$S_{ss}$')
        ax13.set_ylabel("Predicted " + r'$S_{ss}$' + " (m)")
        ax13.set_xlabel('Human ' + r'$S_{ss}$' + ' (m)')

        ax14 = fig.add_subplot(2, 3, 6)
        ax14.scatter(t_mean, ml_mean)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(t_mean, ml_mean)
        ax14.text(.8, .1, r'$R^{2}$' + str(r_value ** 2)[:4])
        ax14.set_title(r'$R_{mean}$')
        ax14.set_ylabel("Predicted " + r'$R_{mean}$' + " (m)")
        ax14.set_xlabel('Human ' + r'$R_{mean}$' + ' (m)')
        plt.show()

    @staticmethod
    def timeSeriesAnalysis1D(eta, **kwargs):
        """process 1D timeserise analysis, function will demean data by default.  It can operate on
        2D spatial surface elevation data, but will only do 1D analysis (not puv/2D directional waves)
        for frequency band averaging, will label with band center

        Args:
            time: time (datetime object)
            eta: surface timeseries

            **kwargs:
                'windowLength': window length for FFT, units are minutes (Default = 10 min)
                'overlap': overlap of windows for FFT, units are percentage of window length (Default=0.75)
                'bandAvg': number of bands to average over
                'timeAx' (int): a number defining which axis in eta is time (Default = 0)
                'returnSetup' (bool): will calculate and return setup (last postion)  (Default = False)

        Returns:
            fspec (array): array of power spectra, dimensioned by [space, frequency]
            frqOut (array): array of frequencys associated with fspec

        Raises:
            Warnings if not all bands are processed (depending on size of freqeuency bands as output by FFT
                and band averaging chosen, function will neglect last few (high frequency) bands

        TODO:
            can add surface correction for pressure data

        """
        from scipy.signal import welch
        import warnings
        bandAvg = kwargs.get('bandAvg', 3)  # average 6 bands #half of typical default because time and distance both downsampled /2?
        myAx = kwargs.get('timeAx', 0)  # time dimension of eta
        nperseg=4096/2 #half of typical default because time and distance both downsampled /2
        fs=7.1225 #double typical default because time downsampled 2
        etaDemeaned = np.nan_to_num(eta - np.mean(eta))

        freqsW, fspecW = welch(x=etaDemeaned, window='hanning', fs=fs, nperseg=nperseg, noverlap=nperseg * .75,
                               nfft=None, return_onesided=True, detrend='linear', axis=myAx)
        # remove first index of array (DC components)--?
        freqW = freqsW[1:]
        fspecW = fspecW[1:]
        ## TODO: add surface correction here

        # initalize for band averaging
        # dk = np.floor(bandAvg/2).astype(int)  # how far to look on either side of band of interest
        frqOut, fspec = [], []
        for kk in range(0, len(freqsW) - bandAvg, bandAvg):
            avgIdxs = np.linspace(kk, kk + bandAvg - 1, num=bandAvg).astype(int)
            frqOut.append(
                freqW[avgIdxs].sum(axis=myAx) / bandAvg)  # taking average of freq for label (band centered label)
            fspec.append(fspecW[avgIdxs].sum(axis=myAx) / bandAvg)
        if max(avgIdxs) < len(freqW):  # provide warning that we're not capturing all energy
            warnings.warn('neglected {} freq bands (at highest frequency)'.format(len(freqW) - max(avgIdxs)))

        frqOut = np.array(frqOut).T
        fspec = np.array(fspec).T
        # output as
        #fspec = np.ma.masked_array(fspec, mask=np.tile((fspec == 0).all(axis=1), (frqOut.size, 1)).T)
        return fspec, frqOut

    @staticmethod
    def stats1D(fspec, frqbins, lowFreq=0.0061, highFreq=0.5):
        """Calculates bulk statistics from a 1 dimentional spectra, calculated as inclusive with
            high/low frequency cuttoff

        Args:
          fspec: frequency spectra
          frqbins: frequency bins associated with the 1d spectra
          lowFreq: low frequency cut off for analysis (Default value = 0.05)
          highFreq: high frequency cutoff for analysis (Default value = 0.5)

        Returns:
          a dictionary with statistics
             'Hmo':   Significant wave height

             'Tp':   Period of the peak energy in the frequency spectra, (1/Fp).  AKA Tpd, not to be
                 confused with parabolic fit to spectral period

             'Tm':    Tm02   Mean spectral period (Tm0,2, from moments 0 & 2), sqrt(m0/m2)

             'Tave':  Tm01   Average period, frequency sprectra weighted, from first moment (Tm0,1)

             'sprdF':  Freq-spec spread (m0*m4 - m2^2)/(m0*m4)  (one definition)

             'Tm10': Mean Absolute wave Period from -1 moment

             'meta': expanded variable name/descriptions

        """
        assert fspec.shape[-1] == len(frqbins), '1D stats need a 1 d spectra'
        if highFreq is None:
            highFreq = np.max(frqbins)
        if lowFreq is None:
            lowFreq = np.min(frqbins)
        frqbins = np.array(frqbins)

        df = np.diff(np.append(frqbins[0], frqbins), n=1)

        # truncating spectra as useful
        idx = np.argwhere((frqbins >= lowFreq) & (frqbins <= highFreq)).squeeze()

        m0 = np.sum(fspec[idx] * df[idx])  # 0th momment
        m1 = np.sum(fspec[idx] * df[idx] * frqbins[idx])  # 1st moment
        m2 = np.sum(fspec[idx] * df[idx] * frqbins[idx] ** 2)  # 2nd moment
        # m3 = np.sum(fspec[:, idx] * df[idx] * frqbins[idx] ** 3, axis=1)  # 3rd moment
        m4 = np.sum(fspec[idx] * df[idx] * frqbins[idx] ** 4)  # 4th moment
        m11 = np.sum(fspec[idx] * df[idx] * frqbins[idx] ** -1)  # negitive one moment
        # wave height
        Hm0 = 4 * np.sqrt(m0)

        ipf = fspec.argmax()  # indix of max frequency
        Tp = 1 / frqbins[ipf]  # peak period
        Tm02 = np.sqrt(m0 / m2)  # mean period
        Tm01 = m0 / m1  # average period - cmparible to TS Tm
        Tm10 = m11 / m0
        sprdF = (m0 * m4 - m2 ** 2) / (m0 * m4)

        # separate into ig and in bands
        frqbins = frqbins[idx]
        fspec = fspec[idx]

        lim2s = np.where(np.abs(frqbins - 0.5) == np.amin(np.abs(frqbins - 0.5)))
        lim25s = np.where(np.abs(frqbins - 0.04) == np.amin(np.abs(frqbins - 0.04)))
        lim2s = int(lim2s[0])
        lim25s = int(lim25s[0])

        indIg = np.linspace(0, lim25s, lim25s + 1, dtype='uint8')
        indIn = np.linspace(lim25s, lim2s, int(lim2s - lim25s + 1), dtype='uint8')

        f_ig = frqbins[indIg];
        f_in = frqbins[indIn];
        amp_ig = fspec[indIg];
        amp_in = fspec[indIn];

        HsRMS = 4 * np.sqrt(np.trapz(fspec, frqbins))
        HsIG = 4 * np.sqrt(np.trapz(amp_ig, f_ig))
        HsIN = 4 * np.sqrt(np.trapz(amp_in, f_in))

        meta = 'Tp - peak period, Tm - mean period, Tave - average period, comparable to Time series mean period, sprdF - frequency spread, sprdD - directional spread'
        stats = {'Hm0': HsRMS,
                 'HsIG': HsIG,
                 'HsIN': HsIN,
                 'Tp': Tp,
                 'Tm': Tm02,
                 'Tave': Tm01,
                 'sprdF': sprdF,
                 'Tm10': Tm10,
                 'meta': meta}


        return stats

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('-act', '--save_activations', action='store_true', help="save activations to predict r2")
    args = parser.parse_args()

    #CalcRunupCalc, Krmslist, Kmaelist, Kdifferencelist, Kmeanmaelist, Kmeandifferencelist = calc_CalcRunup()

    predict = Predictor()
    #predictions_ensemble, images_ensemble, labels_ensemble = predict.main()
    #predict.merge_ensembles(predictions_ensemble, images_ensemble, labels_ensemble)

    images, labels, predictions = predict.reconstruct_window()
    #predict.define_runupline(labels, predictions)
    predict.runupline_5cm(images)
    predict.calc_wavestats(images)