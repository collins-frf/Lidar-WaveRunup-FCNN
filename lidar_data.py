import settings
from torch.utils.data.dataset import Dataset
import glob
import numpy as np
import mat73
from scipy.io import loadmat, savemat

if __name__ == '__main__':
    linedatamat = glob.glob('F:/DuneLidar/processed/unet/*.mat')

    #linedatamat = glob.glob('./MATLAB files/*.mat')
    linedatamat = sorted(linedatamat)
    print(linedatamat)
    #runupsciencemat = glob.glob('./MATLAB files/doRunupScience/Truth*.mat')
    npydata = glob.glob('./data/features_labels*')
    #runupsciencemat.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    print("loading all data...")
    for i in range(len(linedatamat)):
        #try:
        #linedataname = linedatamat[i][28:-26]
        linedataname = linedatamat[i][30:-26]

        #matched_runupscience = [i for i in runupsciencemat if linedataname in i]
        already_Exists = [i for i in npydata if linedataname in i]
        if len(already_Exists) > 0:
            print("npyFile " + linedataname + " already exists")
            continue
        print(linedataname)
        try:
            matfile = mat73.loadmat(linedatamat[i])
        except:
            matfile = loadmat(linedatamat[i])
        #runupsciencefile = loadmat(matched_runupscience[0])

        #plot_spectra(matfile, runupsciencefile)
        QAQC = matfile['lineRunupCalc']['isQAQC']
        print("isQAQC: ", QAQC)
        if QAQC == 1:
            lineGriddedData = matfile['lineGriddedData']
            lineRunupCalc = matfile['lineRunupCalc']

            # assign variables to write to pickle file
            aGrid = lineGriddedData['aGridInterp']
            zGrid = lineGriddedData['zGridInterp']
            zDiffCumulative = lineRunupCalc['zDiffCumulative']

            # calculate Zmin and Z-Zmin
            z_min = np.nanpercentile(zGrid, 95, axis=0)
            z_minus_zmin = np.zeros((len(zGrid), len(zGrid[0])))
            for i in range(len(zGrid)):
                z_minus_zmin[i, :] = zGrid[i, :] - z_min

            aGrid = aGrid[:, :settings.dimensions]
            zGrid = zGrid[:, :settings.dimensions]
            zDiffCumulative = zDiffCumulative[:, :settings.dimensions]
            z_minus_zmin = z_minus_zmin[:, :settings.dimensions]
            aGrid = np.expand_dims(aGrid, axis=-1)
            zGrid = np.expand_dims(zGrid, axis=-1)
            zDiffCumulative = np.expand_dims(zDiffCumulative, axis=-1)
            Z_minus_zmin = np.expand_dims(z_minus_zmin, axis=-1)

            lidar_obs = np.concatenate((aGrid, zGrid, zDiffCumulative, Z_minus_zmin), axis=-1)
            lidar_obs = np.where(np.isnan(lidar_obs), 0, lidar_obs)

            label_classes = lineRunupCalc['griddedDataIsWater']
            label_classes = label_classes[:, :settings.dimensions]
            label_classes = np.where(label_classes, 1, 0)

            #r2 = runupsciencefile['lineRunupData'][0,0][1]['r2per'][0,0][0,0]

            """fig = plt.figure()
            ax0 = fig.add_subplot(2, 3, 1), plt.imshow(label_classes[:settings.dimensions, :])
            ax0[0].set_ylabel("Time")
            ax0[0].set_title("griddedDataIsWater")
            ax1 = fig.add_subplot(2, 3, 2), plt.imshow(lidar_obs[:settings.dimensions, :, 0])
            ax1[0].set_title("aGrid")
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.cm.colors.Normalize(
                vmax=np.amax(lidar_obs[:settings.dimensions, :, 0]),
                vmin=np.amin(lidar_obs[:settings.dimensions, :, 0]))), ax=ax1[0])
            cbar.set_label('Intensity', fontsize=12)
            ax2 = fig.add_subplot(2, 3, 3), plt.imshow(lidar_obs[:settings.dimensions, :, 1])
            ax2[0].set_title("zGrid")
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.cm.colors.Normalize(
                vmax=np.amax(lidar_obs[:settings.dimensions, :, 1]),
                vmin=np.amin(lidar_obs[:settings.dimensions, :, 1]))), ax=ax2[0])
            cbar.set_label('Intensity', fontsize=12)
            ax3 = fig.add_subplot(2, 3, 4), plt.plot(z_min[:settings.dimensions])
            ax3[0].set_xlabel("Cross-shore distance")
            ax3[0].set_ylabel("Elevation")
            ax3[0].set_title("Zmin")

            ax4 = fig.add_subplot(2, 3, 5), plt.imshow(lidar_obs[:settings.dimensions, :, 2])
            ax4[0].set_xlabel("Cross-shore distance")
            ax4[0].set_ylabel("Time")
            ax4[0].set_title("zDiffCumulative")
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.cm.colors.Normalize(
                vmax=np.amax(lidar_obs[:settings.dimensions, :, 2]),
                vmin=np.amin(lidar_obs[:settings.dimensions, :, 2]))), ax=ax4[0])
            cbar.set_label('Intensity', fontsize=12)
            ax5 = fig.add_subplot(2, 3, 6), plt.imshow(lidar_obs[:settings.dimensions, :, 3])
            ax5[0].set_xlabel("Cross-shore distance")
            ax5[0].set_ylabel("Time")
            ax5[0].set_title("Z-Zmin")
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.cm.colors.Normalize(
                vmax=np.amax(lidar_obs[:settings.dimensions, :, 3]),
                vmin=np.amin(lidar_obs[:settings.dimensions, :, 3]))), ax=ax5[0])
            cbar.set_label('Intensity', fontsize=12)
            plt.show()"""

            sample = {'lidar': lidar_obs, 'label': label_classes}#, "r2": r2}
            np.save('./data/features_labels' + linedataname + '.npy', sample)


class LidarDataset(Dataset):
    def __init__(self, transform=None):
        self.data_list = glob.glob('./data/features_labels*.npy')
        self.test_list = glob.glob('./data/test/features_labels*.npy')
        self.transform = transform

    def __getitem__(self, idx):
        return self.load_file(idx)

    def __len__(self):
        return len(self.data_list)

    def load_file(self, idx):
        self.test_list.sort(key=settings.natural_keys)
        self.data_list.sort(key=settings.natural_keys)
        if idx >= 10000:
            new_idx = idx-10000
            idx = new_idx
            sample = np.load(self.test_list[idx], allow_pickle=True)
            startTime = settings.start_row

            if startTime > (settings.matlength - settings.dimensions):
                startTime = (settings.matlength - (settings.dimensions+50))
            endTime = startTime + settings.dimensions

            sample[()]['lidar'] = sample[()]['lidar'][startTime:endTime, :, :]
            sample[()]['label'] = sample[()]['label'][startTime:endTime, :]

        else:
            sample = np.load(self.data_list[idx], allow_pickle=True)

            startTime = np.random.randint(12784-(settings.dimensions+50))
            endTime = startTime + settings.dimensions

            sample[()]['lidar'] = sample[()]['lidar'][startTime:endTime, :, :]
            sample[()]['label'] = sample[()]['label'][startTime:endTime, :]

        """print(self.data_list[idx])
        print(startTime)
        image = sample[()]['lidar']
        label = sample[()]['label']
        fig = plt.figure()
        ax0 = fig.add_subplot(2, 3, 1), plt.imshow(image[:, :, 0])
        ax0[0].set_xlabel("Cross-shore distance")
        ax0[0].set_ylabel("Time")
        ax1 = fig.add_subplot(2, 3, 2), plt.imshow(image[:, :, 1])
        ax1[0].set_xlabel("Cross-shore distance")
        ax2 = fig.add_subplot(2, 3, 3), plt.imshow(label)
        ax2[0].set_ylabel("Cross-shore distance")
        ax2[0].set_xlabel("Time")
        ax3 = fig.add_subplot(2, 3, 4), plt.imshow(image[:, :, 2])
        ax3[0].set_xlabel("Cross-shore distance")
        ax3[0].set_ylabel("Time")
        ax4 = fig.add_subplot(2, 3, 5), plt.imshow(image[:, :, 3])
        ax4[0].set_xlabel("Cross-shore distance")
        ax5 = fig.add_subplot(2, 3, 6), plt.imshow(label)
        ax5[0].set_ylabel("Cross-shore distance")
        ax5[0].set_xlabel("Time")
        plt.tight_layout()
        plt.show()"""
        return sample
