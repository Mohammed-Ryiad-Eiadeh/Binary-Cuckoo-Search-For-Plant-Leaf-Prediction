package WrapperCuckooSearchForFS.org.Optimizers;

import WrapperCuckooSearchForFS.org.Discreeting.TransferFunction;
import WrapperCuckooSearchForFS.org.Evaluation.FitnessFunction;
import org.tribuo.Dataset;
import org.tribuo.FeatureSelector;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.SelectedFeatureSet;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.ensemble.VotingCombiner;
import org.tribuo.common.nearest.KNNModel;
import org.tribuo.common.nearest.KNNTrainer;
import org.tribuo.math.distance.L1Distance;
import org.tribuo.math.neighbour.NeighboursQueryFactoryType;
import org.tribuo.provenance.FeatureSelectorProvenance;
import org.tribuo.provenance.impl.FeatureSelectorProvenanceImpl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Select features based on Cuckoo Search algorithm with binary transfer functions, KNN classifier and 10-fold cross validation
 * <p>
 * see:
 * <pre>
 * Xin-She Yang and Suash Deb.
 * "Cuckoo Search via L´evy Flights", 2010.
 *
 * L. A. M. Pereira et al.
 * "A Binary Cuckoo Search and its Application for Feature Selection", 2014.
 * </pre>
 */
public  final class CuckooSearchOptimizer implements FeatureSelector<Label> {
    private final TransferFunction transferFunction;
    private final double stepSizeScaling;
    private final double lambda;
    private final double worstNestProbability;
    private final double delta;
    private final int populationSize;
    private int [][] setOfSolutions;
    private final FitnessFunction FN;
    private final int maxIteration;
    private final SplittableRandom rng;
    private final int seed;

    /**
     * The default constructor for feature selection based on Cuckoo Search Algorithm
     */
    public CuckooSearchOptimizer() {
        this.transferFunction = TransferFunction.V2;
        this.populationSize = 50;
        KNNTrainer<Label> KnnTrainer =  new KNNTrainer<>(1,
                new L1Distance(),
                Runtime.getRuntime().availableProcessors(),
                new VotingCombiner(),
                KNNModel.Backend.THREADPOOL,
                NeighboursQueryFactoryType.BRUTE_FORCE);
        FN = new FitnessFunction(KnnTrainer);
        this.stepSizeScaling = 2d;
        this.lambda = 2d;
        this.worstNestProbability = 0.1d;
        this.delta = 1.5d;
        this.maxIteration = 30;
        this.seed = 12345;
        this.rng = new SplittableRandom(seed);
    }

    /**
     * Constructs the wrapper feature selection based on cuckoo search algorithm
     * @param trainer The used trainer in the evaluation process
     * @param transferFunction The transfer function to convert continuous values to binary ones
     * @param populationSize The size of the solution in the initial population
     * @param maxIteration The number of times that is used to enhance generation
     * @param seed This seed is required for the SplittableRandom
     */
    public CuckooSearchOptimizer(Trainer<Label> trainer, TransferFunction transferFunction, int populationSize, int maxIteration, int seed) {
        this.transferFunction = transferFunction;
        this.populationSize = populationSize;
        FN = new FitnessFunction(trainer);
        this.stepSizeScaling = 2d;
        this.lambda = 2d;
        this.worstNestProbability = 1.5d;
        this.delta = 1.5d;
        this.maxIteration = maxIteration;
        this.seed = seed;
        this.rng = new SplittableRandom(seed);
    }

    /**
     * Constructs the wrapper feature selection based on cuckoo search algorithm
     * @param trainer The used trainer in the evaluation process
     * @param transferFunction The transfer function to convert continuous values to binary ones
     * @param populationSize The size of the solution in the initial population
     * @param stepSizeScaling The cuckoo step size
     * @param lambda The lambda of the levy flight function
     * @param worstNestProbability The fraction of the nests to be abandoned
     * @param delta The delta that is used in the abandon nest function
     * @param maxIteration The number of times that is used to enhance generation
     * @param seed This seed is required for the SplittableRandom
     */
    public CuckooSearchOptimizer(Trainer<Label> trainer, TransferFunction transferFunction, int populationSize, double stepSizeScaling, double lambda, double worstNestProbability, double delta, int maxIteration, int seed) {
        this.transferFunction = transferFunction;
        this.populationSize = populationSize;
        FN = new FitnessFunction(trainer);
        this.stepSizeScaling = stepSizeScaling;
        this.lambda = lambda;
        this.worstNestProbability = worstNestProbability;
        this.delta = delta;
        this.maxIteration = maxIteration;
        this.seed = seed;
        this.rng = new SplittableRandom(seed);
    }
    /**
     * This method is used to generate the initial population (set of solutions)
     * @param totalNumberOfFeatures The number of features in the given dataset
     * @return The population of subsets of selected features
     */
    private int[][] GeneratePopulation(int totalNumberOfFeatures) {
        setOfSolutions = new int[this.populationSize][totalNumberOfFeatures];
        for (int[] subSet : setOfSolutions) {
            int[] values = new int[subSet.length];
            for (int i = 0; i < values.length; i++) {
                values[i] = rng.nextInt(2);
            }
            System.arraycopy(values, 0, subSet, 0, setOfSolutions[0].length);
        }
        return setOfSolutions;
    }

    /**
     * Does this feature selection algorithm return an ordered feature set?
     *
     * @return True if the set is ordered.
     */
    @Override
    public boolean isOrdered() {
        return true;
    }

    /**
     * Selects features according to this selection algorithm from the specified dataset.
     * @param dataset The dataset to use.
     * @return A selected feature set.
     */
    @Override
    public SelectedFeatureSet select(Dataset<Label> dataset) {
        ImmutableFeatureMap FMap = new ImmutableFeatureMap(dataset.getFeatureMap());
        setOfSolutions = GeneratePopulation(dataset.getFeatureMap().size());
        List<CuckooSearchFeatureSet> subSet_fScores = new ArrayList<>();
        SelectedFeatureSet selectedFeatureSet = null;
        for (int i = 0; i < maxIteration; i++) {
            for (int solution = 0; solution < setOfSolutions.length; solution++) {
                AtomicInteger subSet = new AtomicInteger(solution);
                // Update the solution based on the levy flight function
                int[] evolvedSolution = Arrays.stream(setOfSolutions[subSet.get()]).map(x -> (int) transferFunction.
                        applyAsDouble(x + stepSizeScaling * Math.pow(subSet.get() + 1, -lambda))).toArray();
                int[] randomCuckoo = setOfSolutions[rng.nextInt(setOfSolutions.length)];
                keepBestAfterEvaluation(dataset, FMap, evolvedSolution, randomCuckoo);
                // Update the solution based on the abandone nest function
                if (new Random().nextDouble() < worstNestProbability) {
                    int r1 = rng.nextInt(setOfSolutions.length);
                    int r2 = rng.nextInt(setOfSolutions.length);
                    for (int j = 0; j < setOfSolutions[subSet.get()].length; j++) {
                        evolvedSolution[j] = (int) transferFunction.
                                applyAsDouble(setOfSolutions[subSet.get()][j] +
                                delta * (setOfSolutions[r1][j] - setOfSolutions[r2][j]));
                    }
                    keepBestAfterEvaluation(dataset, FMap, evolvedSolution, setOfSolutions[subSet.get()]);
                    }
            }
            Arrays.stream(setOfSolutions).map(subSet -> new CuckooSearchFeatureSet(subSet,
                    FN.EvaluateSolution(this, dataset, FMap, subSet))).
                    forEach(subSet_fScores::add);
            subSet_fScores.sort(Comparator.comparing(CuckooSearchFeatureSet::score).reversed());
            selectedFeatureSet = FN.getSFS(this, dataset, FMap, subSet_fScores.get(0).subSet);
        }
        return selectedFeatureSet;
    }

    @Override
    public FeatureSelectorProvenance getProvenance() {
        return new FeatureSelectorProvenanceImpl(this);
    }

    /**
     * @param dataset The dataset to use
     * @param FMap The map of selected features
     * @param alteredSolution The modified solution
     * @param oldSolution The old solution
     */
    private void keepBestAfterEvaluation(Dataset<Label> dataset, ImmutableFeatureMap FMap, int[] alteredSolution, int[] oldSolution) {
        double scoreOfModifiedSolution = FN.EvaluateSolution(this, dataset, FMap, alteredSolution);
        double scoreOfSolution = FN.EvaluateSolution(this, dataset, FMap, oldSolution);
        if (scoreOfModifiedSolution > scoreOfSolution) {
            System.arraycopy(alteredSolution, 0, oldSolution, 0, alteredSolution.length);
        }
    }

    /**
     * This record is used to hold subset of features with its corresponding fitness score
     */
    record CuckooSearchFeatureSet(int[] subSet, double score) { }
}