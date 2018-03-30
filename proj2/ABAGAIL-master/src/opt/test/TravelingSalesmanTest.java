package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import java.io.IOException;


import opt.*;
import shared.writer.CSVWriter;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
    private static final int N = 50;
    private static String OUTPUT_DIR = "outputs/";
    double[][] points = new double[N][2];
    private TravelingSalesmanEvaluationFunction ef;
    private Distribution odd;
    private NeighborFunction nf;
    private HillClimbingProblem hcp;
    private int minIter = 50, maxIter = 5000, iterInterval = 200;

    public TravelingSalesmanTest() {
        Random random = new Random();
        // create the random points
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();
        }
        ef = new TravelingSalesmanRouteEvaluationFunction(points);
        odd = new DiscretePermutationDistribution(N);
        nf = new SwapNeighbor();
        hcp = new GenericHillClimbingProblem(ef, odd, nf);
    }

    private void helper(String filename, int minIter, int maxIter, int iterInterval, OptimizationAlgorithm oa, TravelingSalesmanEvaluationFunction ef) throws IOException {
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
            helper("TS-SA.csv", minIter, maxIter, iterInterval, new SimulatedAnnealing(1E12, .95, hcp), this.ef);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void GATest() {
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        try {
            helper("TS-GA.csv", minIter, maxIter, iterInterval, new StandardGeneticAlgorithm(200, 150, 20, gap), this.ef);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void RHCTest() {
        try {
            helper("TS-RHC.csv", minIter, maxIter, iterInterval, new RandomizedHillClimbing(hcp), this.ef);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void MIMICTest() {
        ef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        try {
            helper("TS-MIMIC.csv", minIter, maxIter, iterInterval, new MIMIC(200, 100, pop), ef);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        TravelingSalesmanTest tst = new TravelingSalesmanTest();
        tst.SATest();
        tst.RHCTest();
        tst.GATest();
        tst.MIMICTest();
        // for rhc, sa, and ga we use a permutation based encoding
/*
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        fit.train();
        System.out.println(ef.value(rhc.getOptimal()));
        
        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
        fit = new FixedIterationTrainer(sa, 200000);
        fit.train();
        System.out.println(ef.value(sa.getOptimal()));
        
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
        fit = new FixedIterationTrainer(ga, 1000);
        fit.train();
        System.out.println(ef.value(ga.getOptimal()));
        
        // for mimic we use a sort encoding
        ef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        MIMIC mimic = new MIMIC(200, 100, pop);
        fit = new FixedIterationTrainer(mimic, 1000);
        fit.train();
        System.out.println(ef.value(mimic.getOptimal()));
        */
    }
}
