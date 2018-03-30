package opt.test;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Scanner;

import shared.filt.RandomOrderFilter;
import shared.filt.TestTrainSplitFilter;
import shared.writer.CSVWriter;


public class NN_RandOPT {
    private static String FILEPATH = "src/opt/test/car.data";
    private static String OUTPUT_DIR = "outputs/";
    private static int numOfAttr = 6, numOfInstances = 1728, numOfClasses = 4;
    //private static String FILEPATH = "src/opt/test/tic-tac-toe.data";
    //private static int numOfAttr = 9;
    //private static int numOfInstances = 958, numOfClasses = 2;

    private static HashMap<String, Double>[] attrMaps = new HashMap[numOfAttr];
    private static HashMap<String, Double> labelMap = new HashMap<>();
    private static Instance[] instances, trainInstances, testInstances;

    private static int inputLayer = numOfAttr, hiddenLayer = 1, outputLayer = numOfClasses, trainingIterations = 5000, iterationInterval = 100;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet trainSet, testSet;

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        /*
        FOR Car.data
         */
        labelMap.put("unacc", new Double(0));
        labelMap.put("acc", new Double(1));
        labelMap.put("good", new Double(2));
        labelMap.put("vgood", new Double(3));
        for(int i = 0; i<2; i++) {
            attrMaps[i] = new HashMap<>();
            attrMaps[i].put("vhigh", new Double(3));
            attrMaps[i].put("high", new Double(2));
            attrMaps[i].put("med", new Double(1));
            attrMaps[i].put("low", new Double(0));
        }
        attrMaps[2] = new HashMap<>();
        attrMaps[2].put("5more", new Double(3));
        attrMaps[2].put("4", new Double(2));
        attrMaps[2].put("3", new Double(1));
        attrMaps[2].put("2", new Double(0));
        attrMaps[3] = new HashMap<>();
        attrMaps[3].put("more", new Double(2));
        attrMaps[3].put("4", new Double(1));
        attrMaps[3].put("2", new Double(0));
        attrMaps[4] = new HashMap<>();
        attrMaps[4].put("big", new Double(2));
        attrMaps[4].put("med", new Double(1));
        attrMaps[4].put("small", new Double(0));
        attrMaps[5] = new HashMap<>();
        attrMaps[5].put("high", new Double(2));
        attrMaps[5].put("med", new Double(1));
        attrMaps[5].put("low", new Double(0));

        /*
        FOR tic-tac-toe.data
         */
       // labelMap.put("negative", new Double(0));
       // labelMap.put("positive", new Double(1));
       // HashMap<String, Double> map = new HashMap<>();
       // map.put("x", new Double(2));
       // map.put("o", new Double(1));
       // map.put("b", new Double(0));
       // for(int i = 0; i<numOfAttr; i++) {
       //     attrMaps[i] = map;
       // }

        instances = initializeInstances(FILEPATH, numOfInstances, numOfAttr);

        DataSet instanceSet = new DataSet(instances);
        RandomOrderFilter rof = new RandomOrderFilter();
        rof.filter(instanceSet);
        TestTrainSplitFilter ttsf = new TestTrainSplitFilter(80);
        ttsf.filter(instanceSet);
        trainSet = ttsf.getTrainingSet();
        testSet = ttsf.getTestingSet();
        trainInstances = trainSet.getInstances();
        testInstances = testSet.getInstances();
        runSA();
        runGA();
        runRHC();
    }

    private static void runRHC() {
            BackPropagationNetwork network = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(trainSet, network, measure);
            OptimizationAlgorithm oa = new RandomizedHillClimbing(nnop);
            String outputFile = OUTPUT_DIR+"RHC.csv";
            CSVWriter w = new CSVWriter(outputFile, new String[]{"Iterations", "Training Accuracy", "Testing Accuracy", "Training Time", "Testing Time"});
            try {
                train(oa, network, "RHC", trainingIterations, w);
            } catch (IOException e) {
                e.printStackTrace();
            }
    }

    private static void runSA() {
        double[] coolings = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        for(double cooling: coolings) {
            BackPropagationNetwork network = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(trainSet, network, measure);
            OptimizationAlgorithm oa = new SimulatedAnnealing(1E11, cooling, nnop);
            String outputFile = OUTPUT_DIR+"SA-"+cooling+".csv";
            CSVWriter w = new CSVWriter(outputFile, new String[]{"Iterations", "Training Accuracy", "Testing Accuracy", "Training Time", "Testing Time"});
            try {
                train(oa, network, "SA", trainingIterations, w);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private static void runGA() {
        int[] populations = {10, 50, 100, 200, 400};
        for(int population: populations) {
            BackPropagationNetwork network = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(trainSet, network, measure);
            OptimizationAlgorithm oa = new StandardGeneticAlgorithm(population, population/2, 10,nnop);
            String outputFile = OUTPUT_DIR+"GA-"+population+".csv";
            CSVWriter w = new CSVWriter(outputFile, new String[]{"Iterations", "Training Accuracy", "Testing Accuracy", "Training Time", "Testing Time"});
            try {
                train(oa, network, "GA", trainingIterations, w);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private static double[] computeOAStats(Instance[] instances, OptimizationAlgorithm oa, BackPropagationNetwork network) {
        double start, end, testingTime, correct = 0, incorrect = 0;
        Instance optimalInstance = oa.getOptimal();
        network.setWeights(optimalInstance.getData());

        double predicted, actual;
        start = System.nanoTime();
        for(int j = 0; j < instances.length; j++) {
            network.setInputValues(instances[j].getData());
            network.run();

            predicted = instances[j].getLabel().getData().argMax();
            actual = new Instance(network.getOutputValues()).getData().argMax();

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);
        return new double[]{testingTime, correct/(correct+incorrect)*100};
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int trainingIterations, CSVWriter csvWriter) throws IOException {
        System.out.println("\nTraining results for Optimization problem " + oaName + "\n---------------------------");
        double start = System.nanoTime(), trainingTime=0;
        csvWriter.open();
        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            if ((i+1)%iterationInterval == 0) {
                double end = System.nanoTime();
                trainingTime = end-start;
                trainingTime /= Math.pow(10,9);
                //double error = 0;
                //for(int j = 0; j < trainInstances.length; j++) {
                //    network.setInputValues(trainInstances[j].getData());
                //    network.run();
                //    Instance output = trainInstances[j].getLabel(), example = new Instance(network.getOutputValues());
                //    example.setLabel(new Instance(network.getOutputValues()));
                //    //error += measure.value(output, example);
                //}
                double[] trainStats = computeOAStats(trainInstances, oa, network);
                double[] testStats = computeOAStats(testInstances, oa, network);
                String[] row = {(i+1)+"", df.format(trainStats[1]), df.format(testStats[1]), df.format(trainingTime), df.format(trainStats[0])};
                csvWriter.writeRow(Arrays.asList(row));
                //System.out.println(df.format(error));

            }
        }
        csvWriter.close();
    }

    private static Instance[] initializeInstances(String path, int numOfIns, int numOfAttr) {

        double[][][] attributes = new double[numOfIns][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(path)));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[numOfAttr]; // 7 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < numOfAttr; j++) {
                    String attr = scan.next();
                    if(!attrMaps[j].containsKey(attr)) {
                        System.err.println("Some attribute values cannot be resolved: "+attr+" for attribute "+j);
                        System.exit(1);
                    }
                    attributes[i][0][j] = attrMaps[j].get(attr);
                }

                String label = scan.next();
                if(!labelMap.containsKey(label)) {
                    System.err.println("Some label values cannot be resolved");
                    System.exit(1);
                }
                attributes[i][1][0] = labelMap.get(label);
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // For multi-label
            int c = (int) attributes[i][1][0];
            double[] classes = new double[labelMap.size()];
            classes[c] = 1.0;
            instances[i].setLabel(new Instance(classes));
//            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }

    private static void allOAs() {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(trainSet, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            try {
                train(oa[i], networks[i], oaNames[i], trainingIterations, new CSVWriter("all.csv", new String[]{"Iterations", "Training Accuracy", "Testing Accuracy", "Training Time", "Testing Time"}));
            } catch (IOException e) {
                e.printStackTrace();
            } //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                //predicted = Double.parseDouble(instances[j].getLabel().toString());
                //actual = Double.parseDouble(networks[i].getOutputValues().toString());
                predicted = instances[j].getLabel().getData().argMax();
                actual = new Instance(networks[i].getOutputValues()).getData().argMax();
                //System.out.println("predicted: "+predicted+" actual: "+actual);

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }

        System.out.println(results);

    }
}

