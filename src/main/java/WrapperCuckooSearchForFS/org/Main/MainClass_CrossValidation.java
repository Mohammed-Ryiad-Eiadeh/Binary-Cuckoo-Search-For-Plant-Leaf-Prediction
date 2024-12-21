package WrapperCuckooSearchForFS.org.Main;

import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.fm.FMClassificationTrainer;
import org.tribuo.classification.sgd.objectives.Hinge;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.evaluation.CrossValidation;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.util.Util;

import java.io.IOException;
import java.nio.file.Paths;

public class MainClass_CrossValidation {
    public static void main(String[] args) throws IOException {
        // read the entire dataset after the FS process
        var dataPath = "C:\\Users\\20187\\Desktop\\BCS-Leaf Classification\\Binary Cuckoo Search For Plant Leaf Prediction\\Entire Data Folder\\Swedish After IBCS-FS\\Swedish After FS.csv";
        var dataSource = new CSVLoader<>(new LabelFactory()).loadDataSource(Paths.get(dataPath), "Class");
        var Data = new MutableDataset<>(dataSource);

        // use FM classifier
        var FMTrainer = new FMClassificationTrainer(new Hinge(),
                new AdaGrad(0.1, 0.5),
                50,
                Trainer.DEFAULT_SEED,
                10,
                0.2D);

        // use crossvalidation
        var crossValidation = new CrossValidation<>(FMTrainer,
                Data,
                new LabelEvaluator(),
                3);

        // get outputs
        var avgAcc = 0D;
        var avgRecall = 0D;
        var avgF1 = 0D;
        var avgPrecision = 0D;
        var sTrain = System.currentTimeMillis();
        for (var performance : crossValidation.evaluate()) {
            avgAcc += performance.getA().accuracy();
            avgRecall += performance.getA().macroAveragedRecall();
            avgF1 += performance.getA().macroAveragedF1();
            avgPrecision += performance.getA().macroAveragedPrecision();
        }
        var eTrain = System.currentTimeMillis();

        System.out.println("The Training_Testing duration time is : " + Util.formatDuration(sTrain, eTrain));
        System.out.println("The average accuracy is : " + avgAcc / crossValidation.getK());
        System.out.println("The average recall is : " + avgRecall / crossValidation.getK());
        System.out.println("The average F1-Score is : " + avgF1 / crossValidation.getK());
        System.out.println("The average precision is : " +avgPrecision / crossValidation.getK());
    }
}