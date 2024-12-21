package WrapperCuckooSearchForFS.org.Main;

import org.tribuo.MutableDataset;
import org.tribuo.Trainer;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.ensemble.VotingCombiner;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.sgd.fm.FMClassificationTrainer;
import org.tribuo.classification.sgd.objectives.Hinge;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.ensemble.BaggingTrainer;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.util.Util;

import java.io.IOException;
import java.nio.file.Paths;

public class MainClass_Bagging {
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

        // use bagging for ensimple learning
        var trainer = new BaggingTrainer<>(FMTrainer,
                new VotingCombiner(),
                3,
                Trainer.DEFAULT_SEED);

        // train the model
        var sTrain = System.currentTimeMillis();
        var ensembleLearningTrainer = trainer.train(Data);
        var eTrain = System.currentTimeMillis();

        // define evaluater to test the model to get the output
        var labelEvaluator = new LabelEvaluator().evaluate(ensembleLearningTrainer, Data);

        System.out.println("The Training_Testing duration time is : " + Util.formatDuration(sTrain, eTrain));
        System.out.println("The average accuracy is : " + labelEvaluator.accuracy());
        System.out.println("The average recall is : " + labelEvaluator.macroAveragedRecall());
        System.out.println("The average F1-Score is : " + labelEvaluator.macroAveragedF1());
        System.out.println("The average precision is : " + labelEvaluator.macroAveragedPrecision());
    }
}
