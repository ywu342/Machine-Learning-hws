package opt.test;

import java.io.IOException;
import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.*;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import shared.writer.CSVWriter;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int N = 200;
    /** The t value */
    private static final int T = N / 10;
    private static String OUTPUT_DIR = "outputs/";
    private int[] ranges = new int[N];
    private EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
    private Distribution odd = new DiscreteUniformDistribution(ranges);
    private NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
    private HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
    private int minIter = 50, maxIter = 5000, iterInterval = 100;

    public FourPeaksTest() {
        Arrays.fill(ranges, 2);
    }

    private void helper(String filename, int minIter, int maxIter, int iterInterval, OptimizationAlgorithm oa) throws IOException {
        String outputFile = OUTPUT_DIR+filename;
        CSVWriter writer = new CSVWriter(outputFile, new String[]{"Iterations", "Training Time", "Fitness"});
        writer.open();
        for (int i = minIter; i<=maxIter; i+=iterInterval) {
            double start = System.nanoTime(), trainingTime=0;
            FixedIterationTrainer fit = new FixedIterationTrainer(oa, i);
            fit.train();
            double end = System.nanoTime();
            double fitness = ef.value(oa.getOptimal());
            trainingTime = end-start;
            trainingTime /= Math.pow(10,9);
            String[] row = {i+"", trainingTime+"", fitness+""};
            writer.writeRow(Arrays.asList(row));
        }
        writer.close();
    }

    public void SATest() {
        try {
            helper("FP-SA.csv", minIter, maxIter, iterInterval, new SimulatedAnnealing(1E11, .95, hcp));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void GATest() {
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        try {
            helper("FP-GA.csv", minIter, maxIter, iterInterval, new StandardGeneticAlgorithm(200, 100, 10, gap));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void RHCTest() {
        try {
            helper("FP-RHC.csv", minIter, maxIter, iterInterval, new RandomizedHillClimbing(hcp));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void MIMICTest() {
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        try {
            helper("FP-MIMIC.csv", minIter, maxIter, iterInterval, new MIMIC(200, 20, pop));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        FourPeaksTest fpt = new FourPeaksTest();
        fpt.SATest();
        fpt.RHCTest();
        fpt.GATest();
        fpt.MIMICTest();
/*
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        fit.train();
        System.out.println("RHC: " + ef.value(rhc.getOptimal()));
        
        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
        fit = new FixedIterationTrainer(sa, 200000);
        fit.train();
        System.out.println("SA: " + ef.value(sa.getOptimal()));
        
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
        fit = new FixedIterationTrainer(ga, 1000);
        fit.train();
        System.out.println("GA: " + ef.value(ga.getOptimal()));
        
        MIMIC mimic = new MIMIC(200, 20, pop);
        fit = new FixedIterationTrainer(mimic, 1000);
        fit.train();
        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
*/
    }
}
