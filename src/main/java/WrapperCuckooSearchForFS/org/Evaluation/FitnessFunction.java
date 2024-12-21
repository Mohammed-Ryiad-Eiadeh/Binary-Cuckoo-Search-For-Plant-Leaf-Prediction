package WrapperCuckooSearchForFS.org.Evaluation;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.FeatureSelector;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;
import org.tribuo.SelectedFeatureSet;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.dataset.SelectedFeatureDataset;
import org.tribuo.evaluation.CrossValidation;
import org.tribuo.provenance.FeatureSetProvenance;

import java.util.ArrayList;
import java.util.List;

/**
 * This class is used to evaluate each candidate solution which is nothing but a subset of features
 */
public final class FitnessFunction {
    private final Trainer<Label> trainer;

    /**
     * Default constructor of the fitness function utility
     * @param trainer The used trainer in the evaluation process
     */
    public FitnessFunction(Trainer<Label> trainer) {
        this.trainer = trainer;
    }
    /**
     * This method is used to compute the fitness score of each solution of the population
     * @param optimizer The optimizer that is used for FS
     * @param dataset The dataset to use
     * @param Fmap The dataset feature map
     * @param solution The current subset of features
     * @return The fitness score of the given subset
     */
    public <T extends FeatureSelector<Label>> double EvaluateSolution(T optimizer, Dataset<Label> dataset, ImmutableFeatureMap Fmap, int[] solution) {
        SelectedFeatureDataset<Label> selectedFeatureDataset = new SelectedFeatureDataset<>(dataset, getSFS(optimizer, dataset, Fmap, solution));
        CrossValidation<Label, LabelEvaluation> crossValidation = new CrossValidation<>(trainer, selectedFeatureDataset, new LabelEvaluator(), 10);
        double avgAccuracy = 0D;
        for (Pair<LabelEvaluation, Model<Label>> ACC : crossValidation.evaluate()) {
            avgAccuracy += ACC.getA().accuracy();
        }
        avgAccuracy /= crossValidation.getK();
        int sizeOfSubset = selectedFeatureDataset.getSelectedFeatures().size();
        int sizeOfDataset = Fmap.size();
        return avgAccuracy + 0.001 * (1 - ((double) sizeOfSubset / sizeOfDataset));
    }

    /**
     * This methid is used to return the selected subset of features
     * @param optimizer The optimizer that is used for FS
     * @param dataset The dataset to use
     * @param Fmap The dataset feature map
     * @param solution The current subset of featurs
     * @return The selected feature set
     */
    public <T extends FeatureSelector<Label>> SelectedFeatureSet getSFS(T optimizer, Dataset<Label> dataset, ImmutableFeatureMap Fmap, int[] solution) {
        List<String> names = new ArrayList<>();
        List<Double> scores = new ArrayList<>();
        for (int i = 0; i < solution.length; i++) {
            if (solution[i] == 1) {
                names.add(Fmap.get(i).getName());
                scores.add(1D);
            }
        }
        FeatureSetProvenance provenance = new FeatureSetProvenance(SelectedFeatureSet.class.getName(),
                dataset.getProvenance(),
                optimizer.getProvenance());
        return new SelectedFeatureSet(names, scores, optimizer.isOrdered(), provenance);
    }
}

