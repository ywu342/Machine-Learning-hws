package mdps;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class EpisodeData {
    public List<Integer> numOfIters = new ArrayList<Integer>();
    public List<Double> rewards = new ArrayList<Double>();
    public List<Integer> steps = new ArrayList<Integer>();
    public List<Double> usedTime = new ArrayList<Double>();
    private static final String DELIMITER = ",";
    private static final String NEW_LINE_SEPARATOR = "\n";

    public void Reset(){
        numOfIters = new ArrayList<Integer>();
        rewards = new ArrayList<Double>();
        steps = new ArrayList<Integer>();
        usedTime = new ArrayList<Double>();
    }

    public void addTuple(int i, double r, int step, double time){
        numOfIters.add(i);
        rewards.add(r);
        steps.add(step);
        usedTime.add(time);
    }

    public void print(){
        System.out.println("\n");
        for (int i = 0; i < numOfIters.size(); i++){
            System.out.println(numOfIters.get(i) + ", " + rewards.get(i) +
                    ", " + steps.get(i) + ", " + usedTime.get(i));
        }
    }

    public void write(String pathName){
        FileWriter fileWriter = null;
        try{
            fileWriter = new FileWriter(pathName);
            fileWriter.append("iterations,rewards,steps,time\n");
            for (int i = 0; i < numOfIters.size(); i++){
                fileWriter.append(String.valueOf(numOfIters.get(i)));
                fileWriter.append(DELIMITER);
                fileWriter.append(String.valueOf(rewards.get(i)));
                fileWriter.append(DELIMITER);
                fileWriter.append(String.valueOf(steps.get(i)));
                fileWriter.append(DELIMITER);
                fileWriter.append(String.valueOf(usedTime.get(i)));
                fileWriter.append(NEW_LINE_SEPARATOR);
            }
        }
        catch (Exception e){
            System.out.println("Error when writing CSV");
            e.printStackTrace();
        }
        finally {
            try{
                fileWriter.flush();
                fileWriter.close();
            }
            catch (Exception e){
                System.out.println("Error while flushing or closing CSV");
                e.printStackTrace();
            }
        }
    }

}
