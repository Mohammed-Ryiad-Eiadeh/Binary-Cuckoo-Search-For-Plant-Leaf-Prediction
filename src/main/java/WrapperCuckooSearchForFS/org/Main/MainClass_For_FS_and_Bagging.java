package WrapperCuckooSearchForFS.org.Main;

import WrapperCuckooSearchForFS.org.Discreeting.TransferFunction;
import WrapperCuckooSearchForFS.org.Optimizers.CuckooSearchOptimizer;
import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.ensemble.VotingCombiner;
import org.tribuo.common.nearest.KNNModel;
import org.tribuo.common.nearest.KNNTrainer;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.data.csv.CSVSaver;
import org.tribuo.dataset.SelectedFeatureDataset;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.math.distance.L2Distance;
import org.tribuo.math.neighbour.NeighboursQueryFactoryType;
import org.tribuo.util.Util;

import java.io.IOException;
import java.nio.file.Paths;

public class MainClass_For_FS_and_Bagging {
    public static void main(String[] args) throws IOException {
        // read the data
        var dataPath = "C:\\Users\\20187\\Desktop\\BCS-Leaf Classification\\Binary Cuckoo Search For Plant Leaf Prediction\\Entire Data Folder\\Original Dataset\\Flavia Leaf data.csv";
        var dataSource = new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get(dataPath), "Class");

        var dataSplitting = new TrainTestSplitter<>(dataSource, 0.6, Trainer.DEFAULT_SEED);
        var trainData = new MutableDataset<>(dataSplitting.getTrain());
        var testData = new MutableDataset<>(dataSplitting.getTest());

        // use the feature selection optimizer based on the given learner
        var learner = new KNNTrainer<>(1,
                new L2Distance(),
                Runtime.getRuntime().availableProcessors(),
                new VotingCombiner(),
                KNNModel.Backend.THREADPOOL,
                NeighboursQueryFactoryType.BRUTE_FORCE);

        // use IBCS for FS
        var optimizer = new CuckooSearchOptimizer(learner,
                TransferFunction.V2,
                30,
                2d,
                2d,
                0.2d,
                1.5d,
                20,
                12345);


        /*// use mRMR filter-based FS
        var sDate = System.currentTimeMillis();
        var SFS = new mRMR(500,
                10,
                Runtime.getRuntime().availableProcessors())
                .select(trainPart);
        var eDate = System.currentTimeMillis();
        var SFDS = new SelectedFeatureDataset<>(trainPart, SFS);*/

        var sDate = System.currentTimeMillis();
        var SFS = optimizer.select(trainData);
        var eDate = System.currentTimeMillis();
        var SFDS = new SelectedFeatureDataset<>(trainData, SFS);

        // save the selected subset of features
        new CSVSaver().save(Paths.get(System.getProperty("user.dir") + "\\xxxx data After FS.csv"),
                SFDS,
                "Class");

        System.out.printf("The FS duration time is : %s\nThe number of selected features is : %d\nThe feature names are : %s\n",
                Util.formatDuration(sDate, eDate), SFDS.size(), SFDS.getFeatureSet().featureNames());

        /* Here you store the data after feature selection (FS) for the training part only.
         * To get the values or columns of the features in the generated training set after FS,
         * you should use the TableSaw library or another library to drop the unwanted features
         * and keep the ones that match the training part. */
    }
}